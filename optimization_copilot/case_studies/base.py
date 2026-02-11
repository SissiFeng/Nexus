"""Base classes for real experimental case study benchmarks.

Provides:
- SimpleSurrogate: lightweight GP for offline replay (RBF kernel + Cholesky)
- ExperimentalBenchmark: ABC for real experimental data benchmarks
- ReplayBenchmark: offline replay using GP surrogates built from data
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from functools import partial
from typing import Any

from optimization_copilot.backends._math.kernels import rbf_kernel, kernel_matrix
from optimization_copilot.backends._math.linalg import (
    cholesky,
    solve_cholesky,
    solve_lower,
    vec_dot,
)
from optimization_copilot.domain_knowledge.loader import DomainConfig


# ---------------------------------------------------------------------------
# SimpleSurrogate -- lightweight GP for offline replay
# ---------------------------------------------------------------------------


class SimpleSurrogate:
    """Lightweight GP surrogate for offline replay benchmarks.

    Uses RBF kernel + Cholesky from backends/_math/.
    No optimisation of hyperparameters -- uses fixed lengthscale and signal
    variance.

    Parameters
    ----------
    lengthscale : float
        RBF kernel length-scale.
    signal_variance : float
        Signal variance (output scale).
    noise : float
        Observation noise variance added to the diagonal.
    """

    def __init__(
        self,
        lengthscale: float = 1.0,
        signal_variance: float = 1.0,
        noise: float = 0.01,
    ) -> None:
        self.lengthscale = lengthscale
        self.signal_variance = signal_variance
        self.noise = noise

        # Populated by fit()
        self._X_train: list[list[float]] = []
        self._y_train: list[float] = []
        self._L: list[list[float]] = []
        self._alpha: list[float] = []
        self._fitted: bool = False

    # -- kernel helper -----------------------------------------------------

    def _kernel(self, x1: list[float], x2: list[float]) -> float:
        """Scaled RBF kernel: signal_variance * rbf(x1, x2, lengthscale)."""
        return self.signal_variance * rbf_kernel(
            x1, x2, length_scale=self.lengthscale
        )

    # -- public API --------------------------------------------------------

    def fit(self, X: list[list[float]], y: list[float]) -> None:
        """Fit GP to training data using Cholesky factorisation.

        Parameters
        ----------
        X : list[list[float]]
            Training inputs, each a list of floats.
        y : list[float]
            Training targets.
        """
        self._X_train = X
        self._y_train = y
        n = len(X)

        # Build kernel matrix K using kernel_matrix from _math/kernels.py
        # kernel_matrix already adds a small noise to the diagonal; we pass
        # our own noise level (signal_variance-scaled noise).
        K = kernel_matrix(X, self._kernel, noise=self.noise)

        # Cholesky factor L  (K = L L^T)
        self._L = cholesky(K)

        # alpha = K^{-1} y  via  solve_cholesky(L, y)
        self._alpha = solve_cholesky(self._L, y)
        self._fitted = True

    def predict(self, x: list[float]) -> tuple[float, float]:
        """Predict mean and variance at a new point.

        Parameters
        ----------
        x : list[float]
            Query point.

        Returns
        -------
        tuple[float, float]
            (mean, variance) at *x*.
        """
        if not self._fitted:
            raise RuntimeError("SimpleSurrogate.predict called before fit()")

        # k_star[i] = kernel(x, X_train[i])
        k_star = [self._kernel(x, xi) for xi in self._X_train]

        # Predictive mean: mu = k_star^T @ alpha
        mu = vec_dot(k_star, self._alpha)

        # Predictive variance via Cholesky: v = L^{-1} k_star
        v = solve_lower(self._L, k_star)
        var = self.signal_variance - vec_dot(v, v)

        return mu, max(var, 0.0)


# ---------------------------------------------------------------------------
# ExperimentalBenchmark -- ABC for real experimental benchmarks
# ---------------------------------------------------------------------------


class ExperimentalBenchmark(ABC):
    """Base class for real experimental data benchmarks.

    Key differences from ``BenchmarkFunction``:
    - ``evaluate()`` can return ``None`` (constraint violation / experiment
      failure).
    - Returns ``{obj_name: {"value": float, "variance": float}}``
      (noise-aware).
    - Supports mixed parameter spaces (continuous + categorical).
    - Associated ``DomainConfig`` (optional).

    Parameters
    ----------
    domain_name : str | None
        If given, loads the corresponding ``DomainConfig``.
    """

    def __init__(self, domain_name: str | None = None) -> None:
        self.domain_config: DomainConfig | None = (
            DomainConfig(domain_name) if domain_name else None
        )

    @abstractmethod
    def evaluate(self, x: dict) -> dict | None:
        """Evaluate experiment at point *x*.

        Returns
        -------
        dict | None
            ``{obj_name: {"value": float, "variance": float}}`` on success,
            or ``None`` if a constraint was violated / experiment failed.
        """
        ...

    @abstractmethod
    def get_search_space(self) -> dict[str, dict]:
        """Parameter space definition.

        Returns
        -------
        dict[str, dict]
            ``{name: {"type": "continuous"|"categorical",
                       "range": [lo, hi], ...}}``
        """
        ...

    @abstractmethod
    def get_objectives(self) -> dict[str, dict]:
        """Objective definitions.

        Returns
        -------
        dict[str, dict]
            ``{name: {"direction": "maximize"|"minimize", "unit": str}}``
        """
        ...

    def get_known_constraints(self) -> list[dict]:
        """Known constraints (from domain config if available)."""
        if self.domain_config is not None:
            return self.domain_config.get_known_incompatibilities()
        return []

    def get_evaluation_cost(self, x: dict) -> float:
        """Evaluation cost (normalised, default 1.0)."""
        return 1.0

    def is_feasible(self, x: dict) -> bool:
        """True feasibility (for offline evaluation, not exposed to optimiser)."""
        return True

    def get_domain_config(self) -> DomainConfig | None:
        """Return the associated ``DomainConfig``, if any."""
        return self.domain_config


# ---------------------------------------------------------------------------
# ReplayBenchmark -- offline replay using GP surrogates
# ---------------------------------------------------------------------------


class ReplayBenchmark(ExperimentalBenchmark):
    """Offline replay benchmark using a GP surrogate built from real/synthetic data.

    Subclasses implement ``_generate_data()`` to provide training data.
    The base class fits ``SimpleSurrogate``(s) and handles ``evaluate()``.

    Parameters
    ----------
    domain_name : str | None
        Passed to ``ExperimentalBenchmark.__init__``.
    n_train : int
        Number of training points to generate.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        domain_name: str | None = None,
        n_train: int = 50,
        seed: int = 42,
    ) -> None:
        super().__init__(domain_name)
        self._seed = seed
        self._n_train = n_train
        self._rng = random.Random(seed)
        self._surrogates: dict[str, SimpleSurrogate] = {}
        self._noise_levels: dict[str, float] = {}
        self._X_train: list[list[float]] = []
        self._initialize()

    def _initialize(self) -> None:
        """Generate data and fit surrogates."""
        data = self._generate_data()
        self._X_train = data["X"]
        for obj_name in self.get_objectives():
            surrogate = SimpleSurrogate()
            surrogate.fit(data["X"], data["Y"][obj_name])
            self._surrogates[obj_name] = surrogate
            self._noise_levels[obj_name] = data.get(
                "noise_levels", {}
            ).get(obj_name, 0.01)

    @abstractmethod
    def _generate_data(self) -> dict:
        """Generate or load training data.

        Returns
        -------
        dict
            ``{"X": list[list[float]],
              "Y": {obj_name: list[float]},
              "noise_levels": {obj_name: float}}``
        """
        ...

    def evaluate(self, x: dict) -> dict | None:
        """Evaluate the surrogate at point *x*.

        Returns ``None`` if the point is infeasible.
        """
        if not self.is_feasible(x):
            return None

        encoded = self._encode(x)
        results: dict[str, dict[str, float]] = {}
        for obj_name, surrogate in self._surrogates.items():
            mu, var = surrogate.predict(encoded)
            noise_var = self._noise_levels[obj_name]
            y = self._rng.gauss(mu, max((var + noise_var), 1e-12) ** 0.5)
            results[obj_name] = {"value": y, "variance": noise_var}
        return results

    def _encode(self, x: dict) -> list[float]:
        """Default encoding: extract continuous values in search-space order.

        Categorical parameters are one-hot encoded.
        """
        space = self.get_search_space()
        encoded: list[float] = []
        for name, spec in space.items():
            ptype = spec.get("type", "continuous")
            if ptype == "continuous":
                encoded.append(float(x[name]))
            elif ptype == "categorical":
                categories = spec["categories"]
                val = x[name]
                for cat in categories:
                    encoded.append(1.0 if val == cat else 0.0)
        return encoded

    def get_known_optimum(self) -> dict[str, float] | None:
        """Known best objective values (for regret calculation).

        Override in subclass to provide known optima.
        """
        return None
