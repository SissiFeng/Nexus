"""Unified case study evaluation framework.

Provides:
- PerformanceMetrics: standardised metrics for a single optimisation run
- ComparisonResult: result of comparing multiple strategies
- CaseStudyEvaluator: runs and evaluates optimisation strategies on benchmarks
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.case_studies.base import ExperimentalBenchmark
from optimization_copilot.core.models import (
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.plugins.base import AlgorithmPlugin


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PerformanceMetrics:
    """Standardised performance metrics for a single optimisation run."""

    best_value: float
    """Best found objective value."""

    simple_regret: float
    """``|best_found - global_opt|``."""

    convergence_iteration: int
    """Iteration reaching 90 % of optimum."""

    area_under_curve: float
    """Area under convergence curve (lower = better for minimisation)."""

    feasibility_rate: float
    """Fraction of feasible experiments."""

    constraint_violations: int
    """Total constraint violations."""

    total_cost: float
    """Total evaluation cost."""

    cost_adjusted_regret: float
    """``regret / cost``."""

    hypervolume: float | None = None
    """Multi-objective only."""

    pareto_front_size: int | None = None
    """Multi-objective only."""

    std_across_repeats: float = 0.0
    mean_confidence: float = 0.0
    noise_calibration_error: float = 0.0


@dataclass
class ComparisonResult:
    """Result of comparing multiple optimisation strategies."""

    strategy_names: list[str]
    """Names of the compared strategies."""

    metrics: dict[str, list[PerformanceMetrics]]
    """strategy_name -> list of per-repeat metrics."""

    convergence_curves: dict[str, list[list[float]]]
    """strategy_name -> list of convergence curves (per repeat)."""

    budget: int
    n_repeats: int
    benchmark_name: str


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class CaseStudyEvaluator:
    """Unified case study evaluation framework.

    Runs *n_repeats* independent optimisation runs for each strategy,
    records standardised metrics, and enables statistical comparison.

    Parameters
    ----------
    benchmark : ExperimentalBenchmark
        The benchmark to evaluate on.
    """

    def __init__(self, benchmark: ExperimentalBenchmark) -> None:
        self.benchmark = benchmark

    # -- public API --------------------------------------------------------

    def run_single(
        self,
        strategy: AlgorithmPlugin,
        budget: int,
        seed: int = 42,
    ) -> tuple[list[dict], PerformanceMetrics]:
        """Run a single optimisation trial.

        Parameters
        ----------
        strategy : AlgorithmPlugin
            The optimisation strategy to evaluate.
        budget : int
            Number of evaluation iterations.
        seed : int
            Random seed for the strategy.

        Returns
        -------
        tuple[list[dict], PerformanceMetrics]
            ``(history, metrics)`` where *history* is a list of
            ``{"x": dict, "result": dict|None, "iteration": int}`` entries.
        """
        specs = self._build_parameter_specs()
        objectives = self.benchmark.get_objectives()
        # Pick the first objective for single-objective metrics
        obj_name = next(iter(objectives))
        direction = objectives[obj_name].get("direction", "minimize")
        known_opt = None
        if hasattr(self.benchmark, "get_known_optimum"):
            known_opt = self.benchmark.get_known_optimum()

        history: list[dict] = []
        observations: list[Observation] = []
        rng = random.Random(seed)

        for i in range(budget):
            # 1. Get suggestion(s) from strategy
            try:
                suggestions = strategy.suggest(1, seed=seed + i)
            except Exception:
                # Strategy may fail on first call if it requires observations
                # Fall back to random sampling
                suggestions = [self._random_point(specs, rng)]

            if not suggestions:
                suggestions = [self._random_point(specs, rng)]

            x = suggestions[0]

            # 2. Evaluate
            result = self.benchmark.evaluate(x)
            is_failure = result is None

            # 3. Record in history
            history.append({"x": x, "result": result, "iteration": i})

            # 4. Build Observation for strategy.fit()
            if is_failure:
                kpi_values: dict[str, float] = {}
                obs = Observation(
                    iteration=i,
                    parameters=x,
                    kpi_values=kpi_values,
                    qc_passed=False,
                    is_failure=True,
                    failure_reason="infeasible",
                )
            else:
                kpi_values = {
                    k: v["value"] for k, v in result.items()
                }
                obs = Observation(
                    iteration=i,
                    parameters=x,
                    kpi_values=kpi_values,
                    qc_passed=True,
                    is_failure=False,
                )

            observations.append(obs)

            # 5. Fit strategy with updated observations
            try:
                strategy.fit(observations, specs)
            except Exception:
                pass  # Strategy may not support all observation types

        # 6. Compute metrics
        metrics = self._compute_metrics(history, known_opt, obj_name, direction)
        return history, metrics

    def run_comparison(
        self,
        strategies: dict[str, AlgorithmPlugin],
        budget: int,
        n_repeats: int = 30,
    ) -> ComparisonResult:
        """Compare multiple strategies across *n_repeats* independent runs.

        Parameters
        ----------
        strategies : dict[str, AlgorithmPlugin]
            ``{name: plugin}`` mapping.
        budget : int
            Evaluation budget per run.
        n_repeats : int
            Number of independent repetitions.

        Returns
        -------
        ComparisonResult
        """
        all_metrics: dict[str, list[PerformanceMetrics]] = {}
        all_curves: dict[str, list[list[float]]] = {}

        for name, strategy in strategies.items():
            all_metrics[name] = []
            all_curves[name] = []

            for repeat in range(n_repeats):
                run_seed = 1000 * repeat + 42
                history, metrics = self.run_single(strategy, budget, seed=run_seed)

                all_metrics[name].append(metrics)

                # Extract convergence curve
                objectives = self.benchmark.get_objectives()
                obj_name = next(iter(objectives))
                direction = objectives[obj_name].get("direction", "minimize")
                curve = self._extract_convergence_curve(
                    history, obj_name, direction
                )
                all_curves[name].append(curve)

        benchmark_name = type(self.benchmark).__name__
        return ComparisonResult(
            strategy_names=list(strategies.keys()),
            metrics=all_metrics,
            convergence_curves=all_curves,
            budget=budget,
            n_repeats=n_repeats,
            benchmark_name=benchmark_name,
        )

    # -- internal helpers --------------------------------------------------

    def _build_parameter_specs(self) -> list[ParameterSpec]:
        """Convert benchmark search space to ``ParameterSpec`` list."""
        specs: list[ParameterSpec] = []
        for name, space_def in self.benchmark.get_search_space().items():
            ptype = space_def.get("type", "continuous")
            if ptype == "continuous":
                lo, hi = space_def["range"]
                specs.append(
                    ParameterSpec(
                        name=name,
                        type=VariableType.CONTINUOUS,
                        lower=lo,
                        upper=hi,
                    )
                )
            elif ptype == "categorical":
                specs.append(
                    ParameterSpec(
                        name=name,
                        type=VariableType.CATEGORICAL,
                        categories=space_def["categories"],
                    )
                )
            elif ptype == "discrete":
                lo, hi = space_def["range"]
                specs.append(
                    ParameterSpec(
                        name=name,
                        type=VariableType.DISCRETE,
                        lower=lo,
                        upper=hi,
                    )
                )
        return specs

    def _compute_metrics(
        self,
        history: list[dict],
        known_opt: dict[str, float] | None,
        objective_name: str,
        direction: str,
    ) -> PerformanceMetrics:
        """Compute standardised metrics from optimisation history."""
        # Collect feasible values
        feasible_values: list[float] = []
        total_cost = 0.0
        constraint_violations = 0
        n_feasible = 0
        convergence_curve: list[float] = []

        is_minimize = direction == "minimize"

        for entry in history:
            result = entry["result"]
            x = entry["x"]
            total_cost += self.benchmark.get_evaluation_cost(x)

            if result is None:
                constraint_violations += 1
                # Carry forward last best
                if convergence_curve:
                    convergence_curve.append(convergence_curve[-1])
                else:
                    convergence_curve.append(float("inf") if is_minimize else float("-inf"))
                continue

            n_feasible += 1
            val = result[objective_name]["value"]
            feasible_values.append(val)

            # Update best-so-far
            if not convergence_curve:
                convergence_curve.append(val)
            else:
                prev_best = convergence_curve[-1]
                if is_minimize:
                    convergence_curve.append(min(prev_best, val))
                else:
                    convergence_curve.append(max(prev_best, val))

        n_total = len(history)
        feasibility_rate = n_feasible / max(n_total, 1)

        # Best value
        if not feasible_values:
            best_value = float("inf") if is_minimize else float("-inf")
        else:
            best_value = min(feasible_values) if is_minimize else max(feasible_values)

        # Simple regret
        if known_opt is not None and objective_name in known_opt:
            simple_regret = abs(best_value - known_opt[objective_name])
        else:
            simple_regret = 0.0

        # Convergence iteration (first iteration reaching 90% of optimum gap)
        convergence_iteration = n_total  # default: never converged
        if known_opt is not None and objective_name in known_opt and feasible_values:
            opt_val = known_opt[objective_name]
            # 90% threshold
            first_val = convergence_curve[0] if convergence_curve else best_value
            gap = abs(first_val - opt_val)
            threshold_gap = 0.1 * gap  # 90% closed means 10% remaining
            for idx, cv in enumerate(convergence_curve):
                if abs(cv - opt_val) <= threshold_gap + 1e-12:
                    convergence_iteration = idx
                    break

        # Area under convergence curve
        area_under_curve = 0.0
        for val in convergence_curve:
            # Clamp infinities for AUC calculation
            clamped = val if abs(val) < 1e15 else 0.0
            area_under_curve += clamped
        if convergence_curve:
            area_under_curve /= len(convergence_curve)

        # Cost-adjusted regret
        cost_adjusted_regret = (
            simple_regret / max(total_cost, 1e-12)
        )

        return PerformanceMetrics(
            best_value=best_value,
            simple_regret=simple_regret,
            convergence_iteration=convergence_iteration,
            area_under_curve=area_under_curve,
            feasibility_rate=feasibility_rate,
            constraint_violations=constraint_violations,
            total_cost=total_cost,
            cost_adjusted_regret=cost_adjusted_regret,
        )

    def _extract_convergence_curve(
        self,
        history: list[dict],
        objective_name: str,
        direction: str,
    ) -> list[float]:
        """Extract best-so-far convergence curve from history."""
        is_minimize = direction == "minimize"
        curve: list[float] = []
        for entry in history:
            result = entry["result"]
            if result is None:
                if curve:
                    curve.append(curve[-1])
                else:
                    curve.append(float("inf") if is_minimize else float("-inf"))
                continue
            val = result[objective_name]["value"]
            if not curve:
                curve.append(val)
            else:
                prev = curve[-1]
                if is_minimize:
                    curve.append(min(prev, val))
                else:
                    curve.append(max(prev, val))
        return curve

    @staticmethod
    def _random_point(
        specs: list[ParameterSpec], rng: random.Random
    ) -> dict[str, Any]:
        """Generate a random point within the parameter specs."""
        point: dict[str, Any] = {}
        for spec in specs:
            if spec.type == VariableType.CONTINUOUS:
                lo = spec.lower if spec.lower is not None else 0.0
                hi = spec.upper if spec.upper is not None else 1.0
                point[spec.name] = rng.uniform(lo, hi)
            elif spec.type == VariableType.CATEGORICAL:
                if spec.categories:
                    point[spec.name] = rng.choice(spec.categories)
                else:
                    point[spec.name] = ""
            elif spec.type == VariableType.DISCRETE:
                lo = int(spec.lower) if spec.lower is not None else 0
                hi = int(spec.upper) if spec.upper is not None else 10
                point[spec.name] = rng.randint(lo, hi)
        return point
