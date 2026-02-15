"""Residual Gaussian Process for hybrid theory-data models.

Fits a lightweight GP on the residuals between observed data and a
deterministic theory model.  Reuses the Cholesky-based linear algebra
from ``backends._math.linalg`` and kernel functions from
``backends._math.kernels``.
"""

from __future__ import annotations

import math
from typing import Callable

from optimization_copilot.hybrid.theory import TheoryModel


class ResidualGP:
    """GP fitted on residuals: r(x) = y_observed - theory(x).

    This is a lightweight GP that reuses the math from
    ``backends/_math/linalg``.  It is **not** a full
    :class:`~optimization_copilot.backends.builtin.GaussianProcessBO`
    --- just the core predict functionality for hybrid modeling.

    Parameters
    ----------
    theory_model : TheoryModel
        The deterministic theory component.
    kernel_fn : callable or None
        Kernel function ``(x1, x2) -> float``.  Defaults to
        :func:`rbf_kernel` with ``length_scale=1.0``.
    noise : float
        Observation noise variance added to the kernel diagonal
        (default 1e-6).
    """

    def __init__(
        self,
        theory_model: TheoryModel,
        kernel_fn: Callable[[list[float], list[float]], float] | None = None,
        noise: float = 1e-6,
    ) -> None:
        self._theory = theory_model
        self._kernel_fn = kernel_fn
        self._noise = noise
        self._X_train: list[list[float]] = []
        self._residuals: list[float] = []
        self._alpha: list[float] = []  # K^{-1} @ r
        self._L: list[list[float]] = []  # Cholesky of K
        self._fitted = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_kernel_fn(self) -> Callable[[list[float], list[float]], float]:
        """Return the kernel function, defaulting to rbf_kernel."""
        if self._kernel_fn is not None:
            return self._kernel_fn
        from optimization_copilot.backends._math.kernels import rbf_kernel
        return rbf_kernel

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: list[list[float]], y: list[float]) -> None:
        """Fit the residual GP on training data.

        1. Compute theory predictions ``y_theory = theory.predict(X)``.
        2. Compute residuals ``r_i = y_i - y_theory_i``.
        3. Build kernel matrix ``K[i][j] = k(X[i], X[j]) + noise * delta(i,j)``.
        4. Compute Cholesky factor ``L`` of ``K``.
        5. Solve ``alpha = K^{-1} @ r`` via Cholesky.

        Parameters
        ----------
        X : list[list[float]]
            Training inputs (n rows, d columns).
        y : list[float]
            Observed outputs (length n).
        """
        from optimization_copilot.backends._math.linalg import (
            cholesky,
            solve_cholesky,
        )

        n = len(X)
        self._X_train = [list(row) for row in X]

        # Step 1-2: compute residuals
        y_theory = self._theory.predict(X)
        self._residuals = [y[i] - y_theory[i] for i in range(n)]

        # Step 3: build kernel matrix with noise on diagonal
        kfn = self._get_kernel_fn()
        K: list[list[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                k = kfn(X[i], X[j])
                if i == j:
                    k += self._noise
                K[i][j] = k
                K[j][i] = k

        # Step 4-5: Cholesky factorization and solve
        self._L = cholesky(K)
        self._alpha = solve_cholesky(self._L, self._residuals)
        self._fitted = True

    def predict(
        self, X_new: list[list[float]]
    ) -> tuple[list[float], list[float]]:
        """Predict residual mean and standard deviation at new points.

        Parameters
        ----------
        X_new : list[list[float]]
            Query inputs (m rows).

        Returns
        -------
        tuple[list[float], list[float]]
            ``(mean_residual, std_residual)`` each of length m.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if not self._fitted:
            raise RuntimeError("ResidualGP.predict() called before fit()")

        from optimization_copilot.backends._math.linalg import solve_lower

        kfn = self._get_kernel_fn()
        means: list[float] = []
        stds: list[float] = []
        n = len(self._X_train)

        for x_star in X_new:
            # k_star[j] = k(x*, X_train[j])
            k_star = [kfn(x_star, self._X_train[j]) for j in range(n)]

            # Predictive mean: k_star^T @ alpha
            mu = sum(k_star[j] * self._alpha[j] for j in range(n))

            # Predictive variance: k(x*,x*) - v^T v  where L v = k_star
            k_ss = kfn(x_star, x_star) + self._noise
            v = solve_lower(self._L, k_star)
            var = k_ss - sum(vi * vi for vi in v)
            var = max(var, 1e-12)

            means.append(mu)
            stds.append(math.sqrt(var))

        return means, stds

    def residual_summary(self) -> dict:
        """Summary statistics of the training residuals.

        Returns
        -------
        dict
            Keys: ``mean``, ``std``, ``max_abs``,
            ``has_systematic_bias`` (True if |mean| > 2 * std/sqrt(n)).
        """
        if not self._residuals:
            return {
                "mean": 0.0,
                "std": 0.0,
                "max_abs": 0.0,
                "has_systematic_bias": False,
            }
        n = len(self._residuals)
        mean_r = sum(self._residuals) / n
        var_r = sum((r - mean_r) ** 2 for r in self._residuals) / max(n - 1, 1)
        std_r = math.sqrt(var_r)
        max_abs = max(abs(r) for r in self._residuals)

        # Simple bias test: |mean| > 2 * SE  where SE = std / sqrt(n)
        se = std_r / math.sqrt(n) if n > 0 else 0.0
        has_bias = abs(mean_r) > 2.0 * se if se > 1e-15 else (abs(mean_r) > 1e-15)

        return {
            "mean": mean_r,
            "std": std_r,
            "max_abs": max_abs,
            "has_systematic_bias": has_bias,
        }

    @property
    def fitted(self) -> bool:
        """Whether the GP has been fitted."""
        return self._fitted

    @property
    def residuals(self) -> list[float]:
        """Training residuals (empty before fit)."""
        return list(self._residuals)
