"""Heteroscedastic Gaussian Process backend for Bayesian optimization.

Unlike the standard ``GaussianProcessBO`` which uses a single scalar noise
variance for all observations (K_y = K(X,X) + sigma^2 * I), this backend
supports **per-point noise variances**:

    K_y = K(X, X) + diag(sigma_1^2, ..., sigma_n^2)

This is critical for real experimental campaigns where measurement
uncertainty varies across the search space (e.g., some regions have
noisier instruments, higher fitting residuals, or fewer repetitions).

Implements both ``AlgorithmPlugin`` (for the meta-controller) and
``SurrogateModel`` (for the visualization layer).
"""

from __future__ import annotations

import math
import random
from functools import partial
from typing import Any

from optimization_copilot.core.models import Observation, ParameterSpec, VariableType
from optimization_copilot.plugins.base import AlgorithmPlugin
from optimization_copilot.backends._math.linalg import (
    cholesky,
    solve_cholesky,
    solve_lower,
    vec_dot,
    mat_vec,
    identity,
)
from optimization_copilot.backends._math.kernels import (
    rbf_kernel,
    matern52_kernel,
    kernel_matrix,
)
from optimization_copilot.backends._math.acquisition import expected_improvement


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_param(spec: ParameterSpec, rng: random.Random) -> Any:
    """Draw one random value for *spec*."""
    if spec.type == VariableType.CATEGORICAL:
        return rng.choice(spec.categories)
    if spec.type == VariableType.DISCRETE:
        return rng.randint(int(spec.lower), int(spec.upper))
    return rng.uniform(spec.lower, spec.upper)


def _clamp(value: float, spec: ParameterSpec) -> Any:
    """Clamp *value* to the bounds of *spec*."""
    if spec.type == VariableType.CATEGORICAL:
        return value
    if spec.type == VariableType.DISCRETE:
        return max(int(spec.lower), min(int(spec.upper), int(round(value))))
    return max(spec.lower, min(spec.upper, value))


def _build_kernel_matrix_heteroscedastic(
    X: list[list[float]],
    noise_vars: list[float],
    kernel_fn: Any,
    signal_variance: float,
) -> list[list[float]]:
    """Build K_y = signal_var * K(X,X) + diag(noise_vars).

    We pass ``noise=0`` to ``kernel_matrix`` and add the heteroscedastic
    diagonal ourselves so each point gets its own noise level.
    """
    n = len(X)
    K = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            k = signal_variance * kernel_fn(X[i], X[j])
            if i == j:
                k += noise_vars[i]
            K[i][j] = k
            K[j][i] = k
    return K


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HeteroscedasticGP(AlgorithmPlugin):
    """Gaussian Process with per-point (heteroscedastic) noise variance.

    Parameters
    ----------
    kernel : str
        Kernel function name: ``"matern52"`` (default) or ``"rbf"``.
    lengthscale : float
        Kernel length-scale hyperparameter.
    signal_variance : float
        Signal (output) variance multiplier on the kernel.
    default_noise : float
        Fallback noise variance when an observation does not carry one.
    """

    # -- construction --------------------------------------------------------

    def __init__(
        self,
        kernel: str = "matern52",
        lengthscale: float = 1.0,
        signal_variance: float = 1.0,
        default_noise: float = 0.01,
    ) -> None:
        self._kernel_name = kernel
        self._lengthscale = lengthscale
        self._signal_variance = signal_variance
        self._default_noise = default_noise

        # Training data (populated via observe() or fit())
        self._X: list[list[float]] = []
        self._y: list[float] = []
        self._noise_vars: list[float] = []
        self._metadata_list: list[dict[str, Any]] = []

        # Parameter specs (populated via fit())
        self._specs: list[ParameterSpec] = []

        # Cached Cholesky and alpha (invalidated on new data)
        self._L: list[list[float]] | None = None
        self._alpha: list[float] | None = None

    # -- AlgorithmPlugin interface -------------------------------------------

    def name(self) -> str:
        return "heteroscedastic_gp"

    def fit(
        self,
        observations: list[Observation],
        parameter_specs: list[ParameterSpec],
    ) -> None:
        """Ingest observations. Extracts per-point noise from metadata."""
        self._specs = list(parameter_specs)
        self._X = []
        self._y = []
        self._noise_vars = []
        self._metadata_list = []
        self._L = None
        self._alpha = None

        for obs in observations:
            if obs.is_failure:
                continue
            x = self._to_vec(obs.parameters)
            y = list(obs.kpi_values.values())[0]
            noise_var = obs.metadata.get("noise_variance", self._default_noise)
            self._X.append(x)
            self._y.append(y)
            self._noise_vars.append(float(noise_var))
            self._metadata_list.append(dict(obs.metadata))

    def suggest(
        self,
        n_suggestions: int = 1,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Return *n_suggestions* candidates via EI acquisition.

        Generates Latin Hypercube candidates, evaluates the GP posterior
        at each, computes Expected Improvement, and returns the top-n.
        """
        rng = random.Random(seed)

        # Fallback when insufficient data
        if len(self._X) < 1 or not self._specs:
            return [
                {s.name: _sample_param(s, rng) for s in self._specs}
                for _ in range(n_suggestions)
            ]

        # Ensure GP cache is built
        self._ensure_cache()

        numeric_specs = self._numeric_specs()
        n_dims = len(numeric_specs) if numeric_specs else 1

        best_y = min(self._y)

        # Generate Latin Hypercube candidates
        n_candidates = max(200, 50 * n_dims)
        candidates = self._latin_hypercube(n_candidates, rng)

        # Evaluate EI at each candidate
        scored: list[tuple[float, dict[str, Any]]] = []
        for cand_point, cand_vec in candidates:
            mu, var = self._predict_raw(cand_vec)
            sigma = math.sqrt(max(var, 1e-12))
            ei = expected_improvement(mu, sigma, best_y)
            scored.append((ei, cand_point))

        # Sort by EI descending and take top-n
        scored.sort(key=lambda t: t[0], reverse=True)
        return [point for _, point in scored[:n_suggestions]]

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_heteroscedastic_noise": True,
            "supports_categorical": False,
            "supports_continuous": True,
            "supports_discrete": True,
            "supports_batch": True,
            "requires_observations": True,
            "max_dimensions": None,
        }

    # -- SurrogateModel interface --------------------------------------------

    def predict(self, x: list[float]) -> tuple[float, float]:
        """Return ``(mean, variance)`` for a new input *x*.

        Satisfies the ``SurrogateModel`` protocol.
        """
        if len(self._X) == 0:
            return 0.0, self._signal_variance

        self._ensure_cache()
        return self._predict_raw(x)

    # -- Additional public API -----------------------------------------------

    def observe(
        self,
        x: list[float],
        y: float,
        noise_var: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a single observation with per-point noise.

        Parameters
        ----------
        x : list[float]
            Input vector.
        y : float
            Observed objective value.
        noise_var : float | None
            Noise variance for this point. Uses ``default_noise`` if None.
        metadata : dict | None
            Arbitrary metadata attached to this observation.
        """
        self._X.append(list(x))
        self._y.append(y)
        self._noise_vars.append(noise_var if noise_var is not None else self._default_noise)
        self._metadata_list.append(metadata or {})
        # Invalidate cache
        self._L = None
        self._alpha = None

    def get_model_state(self) -> dict[str, Any]:
        """Export GP state for Agent consumption.

        Returns a dictionary containing all relevant model state
        including training data, hyperparameters, and diagnostics.
        """
        self._ensure_cache()
        return {
            "kernel": self._kernel_name,
            "lengthscale": self._lengthscale,
            "signal_variance": self._signal_variance,
            "default_noise": self._default_noise,
            "n_observations": len(self._X),
            "X_train": [list(x) for x in self._X],
            "y_train": list(self._y),
            "noise_vars": list(self._noise_vars),
            "best_y": min(self._y) if self._y else None,
            "noise_range": (
                (min(self._noise_vars), max(self._noise_vars))
                if self._noise_vars
                else (None, None)
            ),
        }

    def compute_noise_impact(self) -> list[dict[str, Any]]:
        """Diagnostic: per-point weight analysis comparing homo vs hetero.

        For each training point, computes the effective weight (influence)
        under both the heteroscedastic model and a hypothetical
        homoscedastic model with the mean noise variance.  This helps
        identify which points are being up-weighted or down-weighted
        due to their individual noise levels.

        Returns
        -------
        list[dict]
            One entry per training point with keys:
            ``"index"``, ``"x"``, ``"y"``, ``"noise_variance"``,
            ``"hetero_weight"``, ``"homo_weight"``, ``"weight_ratio"``.
        """
        if len(self._X) == 0:
            return []

        n = len(self._X)
        kernel_fn = self._get_kernel_fn()

        # --- Heteroscedastic weights ---
        K_hetero = _build_kernel_matrix_heteroscedastic(
            self._X, self._noise_vars, kernel_fn, self._signal_variance,
        )
        L_hetero = cholesky(K_hetero)
        # Weight of each point = (K_y^{-1} y)_i * y_i  is one view,
        # but a cleaner diagnostic is the diagonal of K_y^{-1}.
        # We compute alpha = K_y^{-1} y for the effective weight.
        alpha_hetero = solve_cholesky(L_hetero, self._y)

        # --- Homoscedastic weights (mean noise) ---
        mean_noise = sum(self._noise_vars) / n if n > 0 else self._default_noise
        homo_noise_vars = [mean_noise] * n
        K_homo = _build_kernel_matrix_heteroscedastic(
            self._X, homo_noise_vars, kernel_fn, self._signal_variance,
        )
        L_homo = cholesky(K_homo)
        alpha_homo = solve_cholesky(L_homo, self._y)

        diagnostics: list[dict[str, Any]] = []
        for i in range(n):
            hetero_w = abs(alpha_hetero[i])
            homo_w = abs(alpha_homo[i])
            ratio = hetero_w / max(homo_w, 1e-30)
            diagnostics.append({
                "index": i,
                "x": list(self._X[i]),
                "y": self._y[i],
                "noise_variance": self._noise_vars[i],
                "hetero_weight": hetero_w,
                "homo_weight": homo_w,
                "weight_ratio": ratio,
            })

        return diagnostics

    # -- Private helpers -----------------------------------------------------

    def _get_kernel_fn(self) -> Any:
        """Return the kernel callable with lengthscale baked in."""
        if self._kernel_name == "rbf":
            return partial(rbf_kernel, length_scale=self._lengthscale)
        # Default: Matern 5/2
        return partial(matern52_kernel, length_scale=self._lengthscale)

    def _numeric_specs(self) -> list[ParameterSpec]:
        return [s for s in self._specs if s.type != VariableType.CATEGORICAL]

    def _to_vec(self, params: dict[str, Any]) -> list[float]:
        """Convert parameter dict to numeric vector (skip categoricals)."""
        vec: list[float] = []
        for s in self._specs:
            if s.type == VariableType.CATEGORICAL:
                continue
            vec.append(float(params.get(s.name, 0.0)))
        return vec

    def _ensure_cache(self) -> None:
        """Build Cholesky factor and alpha if not cached."""
        if self._L is not None and self._alpha is not None:
            return
        if len(self._X) == 0:
            return

        kernel_fn = self._get_kernel_fn()
        K = _build_kernel_matrix_heteroscedastic(
            self._X, self._noise_vars, kernel_fn, self._signal_variance,
        )
        self._L = cholesky(K)
        self._alpha = solve_cholesky(self._L, self._y)

    def _predict_raw(self, x_new: list[float]) -> tuple[float, float]:
        """GP posterior prediction at *x_new*.

        Returns ``(mean, variance)`` including the kernel self-variance
        minus the explained variance from training data.
        """
        kernel_fn = self._get_kernel_fn()

        # k_star: covariance between x_new and each training point
        k_star = [
            self._signal_variance * kernel_fn(x_new, xi)
            for xi in self._X
        ]

        # Posterior mean: mu = k_star^T alpha
        mu = vec_dot(k_star, self._alpha)

        # Posterior variance: var = k(x_new, x_new) - k_star^T K^{-1} k_star
        k_ss = self._signal_variance * kernel_fn(x_new, x_new)
        v = solve_lower(self._L, k_star)
        var = k_ss - vec_dot(v, v)
        var = max(var, 1e-12)

        return mu, var

    def _latin_hypercube(
        self,
        n_samples: int,
        rng: random.Random,
    ) -> list[tuple[dict[str, Any], list[float]]]:
        """Generate LHS candidates as ``(param_dict, numeric_vector)`` pairs."""
        numeric_specs = self._numeric_specs()
        n_dims = len(numeric_specs)

        if n_dims == 0:
            # No numeric dimensions: random sample categoricals
            results: list[tuple[dict[str, Any], list[float]]] = []
            for _ in range(n_samples):
                point = {s.name: _sample_param(s, rng) for s in self._specs}
                results.append((point, []))
            return results

        # Build LHS columns: for each dimension, divide [0, 1] into
        # n_samples strata and place one sample per stratum.
        columns: list[list[float]] = []
        for _ in range(n_dims):
            perm = list(range(n_samples))
            rng.shuffle(perm)
            col = [
                (perm[i] + rng.random()) / n_samples
                for i in range(n_samples)
            ]
            columns.append(col)

        results = []
        for i in range(n_samples):
            point: dict[str, Any] = {}
            vec: list[float] = []
            dim_idx = 0
            for s in self._specs:
                if s.type == VariableType.CATEGORICAL:
                    point[s.name] = rng.choice(s.categories)
                else:
                    u = columns[dim_idx][i]
                    lo, hi = float(s.lower), float(s.upper)
                    val = lo + u * (hi - lo)
                    point[s.name] = _clamp(val, s)
                    vec.append(float(point[s.name]))
                    dim_idx += 1
            results.append((point, vec))

        return results
