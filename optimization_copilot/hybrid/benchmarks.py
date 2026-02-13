"""Sample efficiency benchmark: hybrid vs pure GP vs theory-only.

Demonstrates quantitatively that incorporating physics theory into
the GP prior (via HybridModel) reduces the number of samples needed
to achieve a given prediction accuracy compared to a pure data-driven GP.
"""

from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass, field
from typing import Callable

from optimization_copilot.hybrid.theory import TheoryModel
from optimization_copilot.hybrid.residual import ResidualGP
from optimization_copilot.hybrid.composite import HybridModel


class _ZeroTheory(TheoryModel):
    """Theory model returning zero — makes HybridModel behave as pure GP."""

    def predict(self, X: list[list[float]]) -> list[float]:
        return [0.0] * len(X)

    def n_parameters(self) -> int:
        return 0

    def parameter_names(self) -> list[str]:
        return []


@dataclass
class SampleEfficiencyCurve:
    """RMSE values at each sample size for one method."""
    method: str  # "hybrid", "pure_gp", "theory_only"
    sample_sizes: list[int] = field(default_factory=list)
    rmse_values: list[float] = field(default_factory=list)


@dataclass
class SampleEfficiencyResult:
    """Complete sample efficiency benchmark result."""
    curves: list[SampleEfficiencyCurve]
    efficiency_ratio: float  # n_pure / n_hybrid for matched RMSE
    hybrid_advantage_at_n: dict[int, float]  # sample_size -> % improvement
    minimum_n_hybrid: int | None  # smallest n where hybrid RMSE < threshold
    minimum_n_pure: int | None    # smallest n where pure GP RMSE < threshold
    rmse_threshold: float         # threshold used for minimum_n computation


class SampleEfficiencyBenchmark:
    """Compare hybrid vs pure GP sample efficiency.

    Parameters
    ----------
    sample_sizes : list[int]
        Sample sizes to evaluate at.
    n_test : int
        Number of test points for RMSE evaluation.
    rmse_threshold : float
        RMSE threshold for computing minimum_n values.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        sample_sizes: list[int] | None = None,
        n_test: int = 50,
        rmse_threshold: float = 0.5,
        seed: int = 42,
    ) -> None:
        self._sample_sizes = sample_sizes or [5, 10, 20, 50, 100]
        self._n_test = n_test
        self._rmse_threshold = rmse_threshold
        self._seed = seed

    def compare(
        self,
        theory: TheoryModel,
        data_generator: Callable[[list[list[float]]], list[float]],
        X_domain: list[tuple[float, float]],
        noise_std: float = 0.1,
        kernel_fn: Callable | None = None,
    ) -> SampleEfficiencyResult:
        """Run the sample efficiency comparison.

        Parameters
        ----------
        theory : TheoryModel
            The physics theory model.
        data_generator : callable
            True function: X -> y (without noise).
        X_domain : list[tuple[float, float]]
            (lower, upper) bounds for each dimension.
        noise_std : float
            Observation noise standard deviation.
        kernel_fn : callable or None
            Kernel function for both GP and HybridModel.

        Returns
        -------
        SampleEfficiencyResult
        """
        rng = _random.Random(self._seed)
        n_dims = len(X_domain)

        # Generate test set (fixed)
        X_test = self._random_points(n_dims, X_domain, self._n_test, rng)
        y_test = data_generator(X_test)

        # Curves for each method
        hybrid_curve = SampleEfficiencyCurve(method="hybrid")
        pure_gp_curve = SampleEfficiencyCurve(method="pure_gp")
        theory_curve = SampleEfficiencyCurve(method="theory_only")

        hybrid_advantage: dict[int, float] = {}

        for n in self._sample_sizes:
            # Generate training data with noise
            X_train = self._random_points(n_dims, X_domain, n, rng)
            y_clean = data_generator(X_train)
            y_train = [y + rng.gauss(0, noise_std) for y in y_clean]

            # 1. Hybrid model (theory + GP)
            hybrid_rmse = self._eval_hybrid(
                theory, X_train, y_train, X_test, y_test, kernel_fn
            )
            hybrid_curve.sample_sizes.append(n)
            hybrid_curve.rmse_values.append(hybrid_rmse)

            # 2. Pure GP (zero theory + GP)
            zero_theory = _ZeroTheory()
            pure_rmse = self._eval_hybrid(
                zero_theory, X_train, y_train, X_test, y_test, kernel_fn
            )
            pure_gp_curve.sample_sizes.append(n)
            pure_gp_curve.rmse_values.append(pure_rmse)

            # 3. Theory only (no data)
            y_theory = theory.predict(X_test)
            theory_rmse = self._compute_rmse(y_test, y_theory)
            theory_curve.sample_sizes.append(n)
            theory_curve.rmse_values.append(theory_rmse)

            # Hybrid advantage (% improvement over pure GP)
            if pure_rmse > 1e-15:
                advantage = (pure_rmse - hybrid_rmse) / pure_rmse * 100.0
            else:
                advantage = 0.0
            hybrid_advantage[n] = advantage

        # Compute efficiency ratio and minimum_n
        efficiency_ratio = self._compute_efficiency_ratio(
            hybrid_curve, pure_gp_curve
        )
        min_n_hybrid = self._find_minimum_n(hybrid_curve, self._rmse_threshold)
        min_n_pure = self._find_minimum_n(pure_gp_curve, self._rmse_threshold)

        return SampleEfficiencyResult(
            curves=[hybrid_curve, pure_gp_curve, theory_curve],
            efficiency_ratio=efficiency_ratio,
            hybrid_advantage_at_n=hybrid_advantage,
            minimum_n_hybrid=min_n_hybrid,
            minimum_n_pure=min_n_pure,
            rmse_threshold=self._rmse_threshold,
        )

    def _eval_hybrid(
        self,
        theory: TheoryModel,
        X_train: list[list[float]],
        y_train: list[float],
        X_test: list[list[float]],
        y_test: list[float],
        kernel_fn: Callable | None,
    ) -> float:
        """Fit a HybridModel and compute test RMSE."""
        gp = ResidualGP(theory, kernel_fn=kernel_fn, noise=1e-4)
        model = HybridModel(theory, gp)
        model.fit(X_train, y_train)
        y_pred, _ = model.predict_with_uncertainty(X_test)
        return self._compute_rmse(y_test, y_pred)

    @staticmethod
    def _compute_rmse(y_true: list[float], y_pred: list[float]) -> float:
        """Compute root mean squared error."""
        n = len(y_true)
        if n == 0:
            return 0.0
        se = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n))
        return math.sqrt(se / n)

    @staticmethod
    def _random_points(
        n_dims: int,
        domain: list[tuple[float, float]],
        n_points: int,
        rng: _random.Random,
    ) -> list[list[float]]:
        """Generate random points within domain bounds."""
        points: list[list[float]] = []
        for _ in range(n_points):
            point = [rng.uniform(lo, hi) for lo, hi in domain]
            points.append(point)
        return points

    @staticmethod
    def _compute_efficiency_ratio(
        hybrid_curve: SampleEfficiencyCurve,
        pure_curve: SampleEfficiencyCurve,
    ) -> float:
        """Estimate n_pure / n_hybrid for matched RMSE.

        Uses the best hybrid RMSE and finds what sample size
        the pure GP needs to match it (via linear interpolation).
        Returns ratio > 1 if hybrid is more efficient.
        """
        if not hybrid_curve.rmse_values or not pure_curve.rmse_values:
            return 1.0

        # Target: the hybrid RMSE at its largest sample size
        target_rmse = hybrid_curve.rmse_values[-1]
        n_hybrid = hybrid_curve.sample_sizes[-1]

        # Find where pure GP crosses this RMSE
        n_pure = pure_curve.sample_sizes[-1]  # default: needs all samples
        for i in range(len(pure_curve.rmse_values)):
            if pure_curve.rmse_values[i] <= target_rmse:
                n_pure = pure_curve.sample_sizes[i]
                break

        if n_hybrid > 0:
            return n_pure / n_hybrid
        return 1.0

    @staticmethod
    def _find_minimum_n(
        curve: SampleEfficiencyCurve, threshold: float
    ) -> int | None:
        """Find the smallest sample size where RMSE drops below threshold."""
        for i, rmse in enumerate(curve.rmse_values):
            if rmse <= threshold:
                return curve.sample_sizes[i]
        return None
