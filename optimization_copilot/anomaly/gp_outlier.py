"""Layer 3: GP-based statistical outlier detection.

Provides three detection methods:
1. Standardised residual check (given GP predictions).
2. Leave-one-out (LOO) cross-validation outlier detection.
3. Entropy-change detection for sudden shifts in local variance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from optimization_copilot.backends._math.kernels import rbf_kernel, kernel_matrix
from optimization_copilot.backends._math.linalg import mat_inv, identity


# ── Data types ─────────────────────────────────────────────────────────


@dataclass
class GPAnomaly:
    """An outlier detected by a GP-based method."""

    index: int
    detection_method: str  # "standardized_residual", "loo_cv", "entropy_change"
    score: float           # how anomalous (standardized residual value)
    threshold: float
    message: str


# ── Helpers ────────────────────────────────────────────────────────────


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(var)


# ── GPOutlierDetector ──────────────────────────────────────────────────


class GPOutlierDetector:
    """Statistical outlier detection using GP-based methods.

    Parameters
    ----------
    threshold_sigma : float
        Number of standard deviations beyond which a point is flagged
        as an outlier.  Default is 3.0.
    """

    def __init__(self, threshold_sigma: float = 3.0) -> None:
        self.threshold_sigma = threshold_sigma

    # ── Method 1: standardised residual ────────────────────────────

    def detect_standardized_residual(
        self,
        y_pred: list[float],
        y_std: list[float],
        y_actual: list[float],
    ) -> list[GPAnomaly]:
        """Flag points where |y_actual - y_pred| / y_std > threshold.

        Parameters
        ----------
        y_pred : list[float]
            GP predictive means.
        y_std : list[float]
            GP predictive standard deviations.
        y_actual : list[float]
            Observed values.

        Returns
        -------
        list[GPAnomaly]
            List of anomalies (may be empty).
        """
        n = min(len(y_pred), len(y_std), len(y_actual))
        anomalies: list[GPAnomaly] = []

        for i in range(n):
            if y_std[i] <= 0:
                continue
            residual = abs(y_actual[i] - y_pred[i]) / y_std[i]
            if residual > self.threshold_sigma:
                anomalies.append(GPAnomaly(
                    index=i,
                    detection_method="standardized_residual",
                    score=residual,
                    threshold=self.threshold_sigma,
                    message=(
                        f"Point {i}: |{y_actual[i]:.4g} - {y_pred[i]:.4g}| / "
                        f"{y_std[i]:.4g} = {residual:.2f} > {self.threshold_sigma}"
                    ),
                ))

        return anomalies

    # ── Method 2: leave-one-out cross-validation ───────────────────

    def detect_loo_outlier(
        self,
        X: list[list[float]],
        y: list[float],
        kernel_fn: Callable[[list[float], list[float]], float] | None = None,
        noise: float = 0.01,
    ) -> list[GPAnomaly]:
        """Leave-one-out cross-validation outlier detection.

        For each point *i*, fits a GP on all other points, predicts *i*,
        and flags it if the standardised residual exceeds the threshold.

        Parameters
        ----------
        X : list[list[float]]
            Input features, shape (n, d).
        y : list[float]
            Target values, length n.
        kernel_fn : callable | None
            Kernel function.  Defaults to ``rbf_kernel``.
        noise : float
            Noise variance for the GP (default 0.01).

        Returns
        -------
        list[GPAnomaly]
            Detected outliers.
        """
        n = len(X)
        if n < 3:
            return []

        if kernel_fn is None:
            kernel_fn = rbf_kernel

        # Build full kernel matrix
        K_full = kernel_matrix(X, kernel_fn, noise=noise)

        # Invert full kernel matrix
        K_inv = mat_inv(K_full)

        anomalies: list[GPAnomaly] = []

        # Use the LOO formula from the full inverse:
        # LOO predictive mean for point i:
        #   mu_i = y_i - (K_inv @ y)[i] / K_inv[i][i]
        # LOO predictive variance for point i:
        #   var_i = 1.0 / K_inv[i][i]

        # Compute K_inv @ y
        alpha = [0.0] * n
        for i in range(n):
            for j in range(n):
                alpha[i] += K_inv[i][j] * y[j]

        for i in range(n):
            if K_inv[i][i] <= 0:
                continue
            var_i = 1.0 / K_inv[i][i]
            mu_i = y[i] - alpha[i] / K_inv[i][i]
            std_i = math.sqrt(max(var_i, 1e-12))
            residual = abs(y[i] - mu_i) / std_i

            if residual > self.threshold_sigma:
                anomalies.append(GPAnomaly(
                    index=i,
                    detection_method="loo_cv",
                    score=residual,
                    threshold=self.threshold_sigma,
                    message=(
                        f"LOO point {i}: residual {residual:.2f} > "
                        f"{self.threshold_sigma}"
                    ),
                ))

        return anomalies

    # ── Method 3: entropy (variance) change detection ──────────────

    def detect_entropy_change(
        self,
        y_sequence: list[float],
        window: int = 5,
    ) -> list[GPAnomaly]:
        """Detect sudden changes in local entropy (rolling variance).

        Computes the variance of each rolling window of size *window*,
        then flags where the change in entropy between consecutive windows
        exceeds 2 * overall std of entropy changes.

        Parameters
        ----------
        y_sequence : list[float]
            Sequential observations.
        window : int
            Rolling window size (default 5).

        Returns
        -------
        list[GPAnomaly]
            Detected change points.
        """
        n = len(y_sequence)
        if n < window + 1:
            return []

        # Compute rolling variances
        entropies: list[float] = []
        for i in range(n - window + 1):
            segment = y_sequence[i : i + window]
            m = _mean(segment)
            var = sum((v - m) ** 2 for v in segment) / len(segment)
            entropies.append(var)

        if len(entropies) < 2:
            return []

        # Compute changes between consecutive windows
        changes: list[float] = []
        for i in range(1, len(entropies)):
            changes.append(abs(entropies[i] - entropies[i - 1]))

        overall_std = _std(changes)
        if overall_std <= 0:
            return []

        threshold = 2.0 * overall_std
        anomalies: list[GPAnomaly] = []

        for i, change in enumerate(changes):
            if change > threshold:
                # The change occurs between window starting at i and i+1,
                # which corresponds to the data point at index i + window
                data_index = i + window
                anomalies.append(GPAnomaly(
                    index=data_index,
                    detection_method="entropy_change",
                    score=change / overall_std,
                    threshold=2.0,
                    message=(
                        f"Entropy change at index {data_index}: "
                        f"{change:.4g} > {threshold:.4g} "
                        f"(2\u00d7std of changes)"
                    ),
                ))

        return anomalies
