"""KernelSHAP approximation engine -- pure Python, zero external dependencies.

Implements the KernelSHAP algorithm (Lundberg & Lee, 2017) for computing
approximate Shapley values of a surrogate model's predictions.  Supports
exact enumeration for low-dimensional problems (d <= 11) and random subset
sampling for higher dimensions.

All linear-algebra operations use the helpers in
``optimization_copilot.backends._math.linalg`` so the module remains
dependency-free.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from optimization_copilot.backends._math.linalg import (
    mat_inv,
    mat_mul,
    mat_vec,
    transpose,
)

if TYPE_CHECKING:
    from optimization_copilot.visualization.models import SurrogateModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _comb(n: int, k: int) -> int:
    """Binomial coefficient C(n, k)."""
    return math.comb(n, k)


def _random_sample(population: list[int], k: int, rng: random.Random) -> list[int]:
    """Draw *k* unique items from *population* using *rng* for randomness."""
    # Replicate random.sample logic with a seeded RNG.
    pool = list(population)
    rng.shuffle(pool)
    return pool[:k]


def _shap_kernel_weight(subset_size: int, d: int) -> float:
    """SHAP kernel weight pi(|S|) = (d-1) / (C(d, |S|) * |S| * (d - |S|)).

    Parameters
    ----------
    subset_size : int
        Number of features in the coalition S.
    d : int
        Total number of features.

    Returns
    -------
    float
        Kernel weight.  Returns 0.0 for degenerate cases (|S| == 0 or |S| == d).
    """
    if subset_size == 0 or subset_size == d:
        return 0.0
    denom = _comb(d, subset_size) * subset_size * (d - subset_size)
    if denom == 0:
        return 0.0
    return (d - 1) / denom


def _evaluate_coalition(
    model: SurrogateModel,
    x: list[float],
    X_bg: list[list[float]],
    mask: list[bool],
    max_bg: int | None = None,
) -> float:
    """Compute E[f(x_S, X_{S^c})] by averaging model predictions.

    For features where ``mask[j]`` is True the value from *x* is used;
    for the remaining features the value comes from each background point.

    Parameters
    ----------
    model : SurrogateModel
        Any object satisfying the ``predict(x) -> (mean, unc)`` protocol.
    x : list[float]
        The foreground instance.
    X_bg : list[list[float]]
        Background data used to marginalise absent features.
    mask : list[bool]
        Boolean mask -- True means "feature is in coalition S".
    max_bg : int | None
        If set, use at most this many background points (random sub-sample).

    Returns
    -------
    float
        The averaged model prediction.
    """
    bg = X_bg
    if max_bg is not None and len(bg) > max_bg:
        # Deterministic sub-sampling (take first max_bg after list is formed).
        bg = bg[:max_bg]

    total = 0.0
    for bg_point in bg:
        xi = [x[j] if mask[j] else bg_point[j] for j in range(len(x))]
        mean, _ = model.predict(xi)
        total += mean
    return total / max(len(bg), 1)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class KernelSHAPApproximator:
    """Approximate Shapley values using the KernelSHAP weighted regression.

    Parameters
    ----------
    model : SurrogateModel
        Predictive model exposing ``predict(x) -> (mean, uncertainty)``.
    n_samples : int
        Number of random subsets to draw when d > 11 (sampled mode).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        model: SurrogateModel,
        n_samples: int = 1000,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.n_samples = n_samples
        self.seed = seed
        self._rng = random.Random(seed)

    # -- public API ----------------------------------------------------------

    def compute(
        self,
        x: list[float],
        X_background: list[list[float]],
    ) -> list[float]:
        """Compute SHAP values for a single instance *x*.

        The resulting values satisfy the *efficiency* property:

            sum(phi_i) ~= f(x) - E[f(X_background)]

        Parameters
        ----------
        x : list[float]
            Feature vector to explain.
        X_background : list[list[float]]
            Background dataset used for the expectation.

        Returns
        -------
        list[float]
            One SHAP value per feature.
        """
        d = len(x)
        if d == 0:
            return []
        if d == 1:
            # Trivial case -- the single feature gets the full marginal.
            fx = self.model.predict(x)[0]
            e_fx = sum(self.model.predict(bg)[0] for bg in X_background) / max(
                len(X_background), 1
            )
            return [fx - e_fx]
        if d <= 11:
            return self._exact_shap(x, X_background)
        return self._sampled_shap(x, X_background)

    # -- exact enumeration ---------------------------------------------------

    def _exact_shap(
        self,
        x: list[float],
        X_bg: list[list[float]],
    ) -> list[float]:
        """Exact KernelSHAP: enumerate all 2^d - 2 non-trivial coalitions."""
        d = len(x)
        n_coalitions = (1 << d) - 2  # exclude empty (0) and full (2^d - 1)

        Z: list[list[float]] = []
        y: list[float] = []
        w: list[float] = []

        # f(x) via full coalition
        fx = self.model.predict(x)[0]

        # E[f(X)] via empty coalition
        e_fx = sum(self.model.predict(bg)[0] for bg in X_bg) / max(len(X_bg), 1)

        for bits in range(1, (1 << d) - 1):
            mask = [(bits >> j) & 1 == 1 for j in range(d)]
            sz = sum(mask)
            weight = _shap_kernel_weight(sz, d)

            indicator = [1.0 if m else 0.0 for m in mask]
            val = _evaluate_coalition(self.model, x, X_bg, mask)

            Z.append(indicator)
            y.append(val - e_fx)
            w.append(weight)

        return self._weighted_regression(Z, y, w, d)

    # -- sampled mode --------------------------------------------------------

    def _sampled_shap(
        self,
        x: list[float],
        X_bg: list[list[float]],
    ) -> list[float]:
        """Sampled KernelSHAP for high-dimensional problems (d >= 12)."""
        d = len(x)
        max_bg = min(100, len(X_bg))

        # Limit background points for efficiency.
        bg_subset = X_bg[:max_bg]

        # E[f(X)] over background.
        e_fx = sum(self.model.predict(bg)[0] for bg in bg_subset) / max(
            len(bg_subset), 1
        )

        Z: list[list[float]] = []
        y: list[float] = []
        w: list[float] = []

        all_features = list(range(d))

        for _ in range(self.n_samples):
            # Random subset size from 1 to d-1.
            sz = self._rng.randint(1, d - 1)
            selected = set(_random_sample(all_features, sz, self._rng))

            mask = [j in selected for j in range(d)]
            weight = _shap_kernel_weight(sz, d)

            indicator = [1.0 if m else 0.0 for m in mask]
            val = _evaluate_coalition(self.model, x, bg_subset, mask, max_bg=max_bg)

            Z.append(indicator)
            y.append(val - e_fx)
            w.append(weight)

        return self._weighted_regression(Z, y, w, d)

    # -- weighted least-squares regression -----------------------------------

    def _weighted_regression(
        self,
        Z: list[list[float]],
        y: list[float],
        w: list[float],
        d: int,
    ) -> list[float]:
        """Solve the weighted regression phi = (Z^T W Z)^{-1} Z^T W y.

        Parameters
        ----------
        Z : list[list[float]]
            Indicator matrix, shape (n_coalitions, d).
        y : list[float]
            Target vector (coalition payoffs - baseline), length n_coalitions.
        w : list[float]
            SHAP kernel weights, length n_coalitions.
        d : int
            Number of features.

        Returns
        -------
        list[float]
            SHAP values (length d).
        """
        n = len(Z)
        if n == 0:
            return [0.0] * d

        # Build diagonal-weight-scaled Z: W_half * Z where W is diagonal.
        # More efficient: compute Z^T W Z and Z^T W y directly.

        # Z^T is (d, n).
        Zt = transpose(Z)

        # Apply weights: Z^T W  (d, n) where column j is scaled by w[j].
        ZtW: list[list[float]] = []
        for i in range(d):
            row = [Zt[i][j] * w[j] for j in range(n)]
            ZtW.append(row)

        # Z^T W Z  (d, d)
        ZtWZ = mat_mul(ZtW, Z)

        # Regularise for numerical stability.
        reg = 1e-8
        for i in range(d):
            ZtWZ[i][i] += reg

        # Z^T W y  (d,)
        Wy = [w[j] * y[j] for j in range(n)]
        ZtWy = mat_vec(ZtW, [1.0] * n)  # placeholder -- compute properly below
        # Actually: ZtWy[i] = sum_j ZtW[i][j] * y_j / w_j * w_j ... simpler:
        ZtWy = [sum(Zt[i][j] * w[j] * y[j] for j in range(n)) for i in range(d)]

        # Invert and solve.
        ZtWZ_inv = mat_inv(ZtWZ)
        phi = mat_vec(ZtWZ_inv, ZtWy)

        return phi
