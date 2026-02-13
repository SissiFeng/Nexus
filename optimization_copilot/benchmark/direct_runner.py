"""Direct closed-loop benchmark runner for evaluating optimization backends.

Bridges ``BenchmarkFunction`` and ``AlgorithmPlugin`` by executing a real
suggest -> evaluate -> observe loop, unlike the replay-based ``BenchmarkRunner``.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.benchmark.functions import BenchmarkFunction
from optimization_copilot.core.models import Observation, ParameterSpec, VariableType


@dataclass
class DirectBenchmarkResult:
    """Result of a single direct benchmark run."""

    backend_name: str
    benchmark_name: str
    seed: int
    budget: int
    observations: list[Observation] = field(default_factory=list)
    wall_time_s: float = 0.0

    # ── Derived metrics ───────────────────────────────────────

    @property
    def best_objective(self) -> float:
        """Best (lowest) objective value found."""
        if not self.observations:
            return float("inf")
        return min(
            obs.kpi_values.get("objective", float("inf"))
            for obs in self.observations
        )

    @property
    def simple_regret(self) -> float:
        """Simple regret: f(x_best) - f(x*)."""
        return self.best_objective - self._known_optimum

    @property
    def log10_regret(self) -> float:
        """Log10 of simple regret (clamped to avoid log(0))."""
        sr = max(self.simple_regret, 1e-10)
        return math.log10(sr)

    @property
    def best_so_far(self) -> list[float]:
        """Cumulative best objective at each iteration."""
        result: list[float] = []
        running_best = float("inf")
        for obs in self.observations:
            val = obs.kpi_values.get("objective", float("inf"))
            running_best = min(running_best, val)
            result.append(running_best)
        return result

    @property
    def auc_normalized(self) -> float:
        """Normalized AUC of best-so-far curve (lower is better).

        Computed as average best-so-far minus the known optimum,
        then divided by the range (first value - optimum) for normalization.
        Returns 0.0-1.0 where 0 = perfect convergence, 1 = no improvement.
        """
        bsf = self.best_so_far
        if not bsf:
            return 1.0
        opt = self._known_optimum
        # Average gap from optimum
        avg_gap = sum(v - opt for v in bsf) / len(bsf)
        # Normalize by initial gap
        initial_gap = bsf[0] - opt
        if initial_gap <= 1e-10:
            return 0.0  # Already at optimum from start
        return min(1.0, max(0.0, avg_gap / initial_gap))

    @property
    def convergence_iteration(self) -> int | None:
        """First iteration where regret < 1% of initial gap.

        Returns None if convergence was not achieved.
        """
        bsf = self.best_so_far
        if not bsf:
            return None
        opt = self._known_optimum
        initial_gap = bsf[0] - opt
        if initial_gap <= 1e-10:
            return 0
        threshold = 0.01 * initial_gap
        for i, val in enumerate(bsf):
            if val - opt <= threshold:
                return i
        return None

    _known_optimum: float = 0.0  # Set by runner


def _compute_budget(n_dims: int) -> int:
    """Compute evaluation budget: clamp(10 * n_dims, 20, 200)."""
    return max(20, min(200, 10 * n_dims))


def _benchmark_to_param_specs(benchmark: BenchmarkFunction) -> list[ParameterSpec]:
    """Convert benchmark parameter_specs dicts to ParameterSpec objects."""
    specs: list[ParameterSpec] = []
    for ps in benchmark.parameter_specs:
        bounds = ps.get("bounds", [0.0, 1.0])
        specs.append(ParameterSpec(
            name=ps["name"],
            type=VariableType.CONTINUOUS,
            lower=bounds[0],
            upper=bounds[1],
        ))
    return specs


def _generate_random_point(
    param_specs: list[ParameterSpec], rng: random.Random
) -> dict[str, float]:
    """Generate a random point within parameter bounds."""
    return {
        ps.name: rng.uniform(
            ps.lower if ps.lower is not None else 0.0,
            ps.upper if ps.upper is not None else 1.0,
        )
        for ps in param_specs
    }


class DirectBenchmarkRunner:
    """Closed-loop benchmark runner for optimization backends.

    Executes a real suggest -> evaluate -> observe loop:
    1. Generate n_init random initial points
    2. Fit the backend model on all observations
    3. Get suggestion from backend
    4. Evaluate suggestion on benchmark function
    5. Add observation, repeat until budget exhausted
    """

    def __init__(self, n_init_fraction: float = 0.2) -> None:
        """Initialize runner.

        Parameters
        ----------
        n_init_fraction
            Fraction of budget used for initial random exploration.
            Default 0.2 (20% of budget).
        """
        self._n_init_fraction = n_init_fraction

    def run(
        self,
        plugin_factory: type,
        benchmark: BenchmarkFunction,
        budget: int | None = None,
        seed: int = 42,
    ) -> DirectBenchmarkResult:
        """Run a single benchmark evaluation.

        Parameters
        ----------
        plugin_factory
            AlgorithmPlugin *class* (not instance). A fresh instance is
            created per run to avoid stale state.
        benchmark
            The benchmark function to evaluate on.
        budget
            Total evaluation budget. If None, auto-computed from dimensionality.
        seed
            Random seed for reproducibility.

        Returns
        -------
        DirectBenchmarkResult
            Complete result with observations and derived metrics.
        """
        rng = random.Random(seed)

        # Auto budget
        n_dims = len(benchmark.parameter_specs)
        if budget is None:
            budget = _compute_budget(n_dims)

        # Convert specs
        param_specs = _benchmark_to_param_specs(benchmark)

        # Create fresh plugin instance
        plugin = plugin_factory()

        # Compute n_init
        n_init = max(2, int(self._n_init_fraction * budget))

        # Known optimum for regret
        known_opt = benchmark.known_optimum.get("objective", 0.0)

        # Result container
        result = DirectBenchmarkResult(
            backend_name=plugin.name(),
            benchmark_name=benchmark.name,
            seed=seed,
            budget=budget,
        )
        result._known_optimum = known_opt

        observations: list[Observation] = []
        t_start = time.monotonic()

        # Phase 1: Random initial exploration
        for i in range(min(n_init, budget)):
            params = _generate_random_point(param_specs, rng)
            obj_values = benchmark.evaluate(params)
            obs = Observation(
                iteration=i,
                parameters=params,
                kpi_values=obj_values,
            )
            observations.append(obs)

        # Phase 2: Model-based optimization loop
        for i in range(n_init, budget):
            try:
                # Fit model on all observations so far
                plugin.fit(observations, param_specs)

                # Get suggestion
                suggestions = plugin.suggest(n_suggestions=1, seed=seed + i)
                if suggestions and len(suggestions) > 0:
                    params = suggestions[0]
                    # Clamp to bounds
                    for ps in param_specs:
                        if ps.name in params:
                            lo = ps.lower if ps.lower is not None else 0.0
                            hi = ps.upper if ps.upper is not None else 1.0
                            params[ps.name] = max(lo, min(hi, params[ps.name]))
                else:
                    # Fallback to random if suggest returns nothing
                    params = _generate_random_point(param_specs, rng)
            except Exception:
                # If backend fails, fall back to random
                params = _generate_random_point(param_specs, rng)

            obj_values = benchmark.evaluate(params)
            obs = Observation(
                iteration=i,
                parameters=params,
                kpi_values=obj_values,
            )
            observations.append(obs)

        result.observations = observations
        result.wall_time_s = time.monotonic() - t_start
        return result

    def run_multi_seed(
        self,
        plugin_factory: type,
        benchmark: BenchmarkFunction,
        seeds: list[int],
        budget: int | None = None,
    ) -> list[DirectBenchmarkResult]:
        """Run benchmark across multiple seeds.

        Returns a list of DirectBenchmarkResult, one per seed.
        """
        return [
            self.run(plugin_factory, benchmark, budget=budget, seed=s)
            for s in seeds
        ]
