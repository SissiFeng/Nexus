"""Lightweight Gaussian Process proxy model for intermediate KPIs.

Uses the existing pure-Python math utilities to provide a simple GP
surrogate that can predict mean and variance for unobserved KPI values
at intermediate experiment stages.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any

from optimization_copilot.backends._math import (
    rbf_kernel,
    kernel_matrix,
    cholesky,
    solve_cholesky,
    norm_pdf,
    norm_cdf,
)


def _median_distance(X: list[list[float]]) -> float:
    """Estimate length scale using the median heuristic.

    Computes the median of all pairwise Euclidean distances. Falls back
    to 1.0 if there are fewer than 2 points.

    Parameters
    ----------
    X : list[list[float]]
        Data matrix of shape (n, d).

    Returns
    -------
    float
        Estimated length scale.
    """
    n = len(X)
    if n < 2:
        return 1.0
    distances: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt(sum((X[i][k] - X[j][k]) ** 2 for k in range(len(X[i]))))
            distances.append(d)
    if not distances:
        return 1.0
    distances.sort()
    mid = len(distances) // 2
    median = distances[mid]
    return max(median, 1e-6)


class ProxyModel:
    """Gaussian Process proxy model for a single KPI.

    Fits a GP with an RBF kernel to observations and provides mean/variance
    predictions for new points. The length scale is automatically estimated
    using the median heuristic on the training data.

    Parameters
    ----------
    length_scale : float | None
        Fixed length scale for the RBF kernel. If None, the median
        heuristic is used to estimate it from training data.
    signal_variance : float
        Signal variance (output scale) for the kernel (default 1.0).
    """

    def __init__(
        self,
        length_scale: float | None = None,
        signal_variance: float = 1.0,
    ) -> None:
        self._length_scale_fixed = length_scale
        self._length_scale: float = length_scale if length_scale is not None else 1.0
        self._signal_variance = signal_variance
        self._X: list[list[float]] = []
        self._y: list[float] = []
        self._L: list[list[float]] | None = None
        self._alpha: list[float] | None = None
        self._noise: float = 1e-6
        self._fitted: bool = False

    @property
    def length_scale(self) -> float:
        """Current length scale value."""
        return self._length_scale

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted to data."""
        return self._fitted

    @property
    def n_train(self) -> int:
        """Number of training points."""
        return len(self._X)

    def _kernel_fn(self, x1: list[float], x2: list[float]) -> float:
        """Compute the scaled RBF kernel between two points."""
        return self._signal_variance * rbf_kernel(x1, x2, self._length_scale)

    def fit(
        self,
        X: list[list[float]],
        y: list[float],
        noise: float = 1e-6,
    ) -> None:
        """Fit the GP to training data.

        Parameters
        ----------
        X : list[list[float]]
            Training inputs of shape (n, d).
        y : list[float]
            Training targets of length n.
        noise : float
            Observation noise variance (default 1e-6).

        Raises
        ------
        ValueError
            If X and y have inconsistent lengths.
        """
        if len(X) != len(y):
            raise ValueError(
                f"X has {len(X)} rows but y has {len(y)} elements"
            )
        if len(X) == 0:
            self._X = []
            self._y = []
            self._L = None
            self._alpha = None
            self._fitted = False
            return

        self._X = [list(row) for row in X]
        self._y = list(y)
        self._noise = noise

        # Estimate length scale if not fixed
        if self._length_scale_fixed is None:
            self._length_scale = _median_distance(self._X)

        # Build kernel matrix and decompose
        K = kernel_matrix(self._X, self._kernel_fn, noise=noise)
        self._L = cholesky(K)
        self._alpha = solve_cholesky(self._L, self._y)
        self._fitted = True

    def predict(
        self,
        X_new: list[list[float]],
    ) -> tuple[list[float], list[float]]:
        """Predict mean and variance at new points.

        Parameters
        ----------
        X_new : list[list[float]]
            Test inputs of shape (m, d).

        Returns
        -------
        tuple[list[float], list[float]]
            (means, variances) each of length m.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("ProxyModel has not been fitted; call fit() first")

        means: list[float] = []
        variances: list[float] = []
        for x_new in X_new:
            mu, var = self.predict_single(x_new)
            means.append(mu)
            variances.append(var)
        return means, variances

    def predict_single(self, x: list[float]) -> tuple[float, float]:
        """Predict mean and variance for a single point.

        Parameters
        ----------
        x : list[float]
            Input point of dimension d.

        Returns
        -------
        tuple[float, float]
            (mean, variance) at the point.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("ProxyModel has not been fitted; call fit() first")

        assert self._alpha is not None
        assert self._L is not None

        # k_star: kernel between x and each training point
        k_star = [self._kernel_fn(x, xi) for xi in self._X]

        # Mean: k_star^T @ alpha
        mu = sum(k * a for k, a in zip(k_star, self._alpha))

        # Variance: k(x, x) - k_star^T @ K^{-1} @ k_star
        k_xx = self._kernel_fn(x, x)
        v = solve_cholesky(self._L, k_star)
        var = k_xx - sum(k * vi for k, vi in zip(k_star, v))
        var = max(var, 0.0)  # Numerical stability

        return mu, var

    def y_mean(self) -> float:
        """Return the mean of training targets.

        Returns
        -------
        float
            Mean of training y values, or 0.0 if no data.
        """
        if not self._y:
            return 0.0
        return sum(self._y) / len(self._y)

    def y_best(self) -> float:
        """Return the best (minimum) training target.

        Returns
        -------
        float
            Minimum of training y values.

        Raises
        ------
        RuntimeError
            If no training data is available.
        """
        if not self._y:
            raise RuntimeError("No training data available")
        return min(self._y)
