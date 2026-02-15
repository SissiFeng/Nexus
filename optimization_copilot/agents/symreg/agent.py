"""SymbolicRegressionAgent -- discovers interpretable equations from optimization history.

Uses the pure-Python genetic-programming engine in ``equation_discovery`` to find
simple mathematical relationships between process parameters and KPIs.
"""

from __future__ import annotations

import math
from typing import Any

from optimization_copilot.agents.base import (
    AgentContext,
    AgentMode,
    OptimizationFeedback,
    ScientificAgent,
    TriggerCondition,
)
from optimization_copilot.explain.equation_discovery import (
    EquationDiscovery,
    ParetoSolution,
)


# Minimum number of observations before symbolic regression is meaningful
_MIN_OBSERVATIONS = 10

# Maximum equation complexity for a "good" result
_MAX_GOOD_COMPLEXITY = 10

# MSE threshold above which we consider the fit poor
_MAX_REASONABLE_MSE = 1e6


class SymbolicRegressionAgent(ScientificAgent):
    """Agent that discovers symbolic equations from optimization history.

    Uses tree-based genetic programming to find interpretable mathematical
    relationships between process parameters and KPI values.

    Parameters
    ----------
    population_size : int
        Number of individuals per generation for the GP algorithm.
    generations : int
        Number of evolutionary generations.
    seed : int
        Random seed for reproducibility.
    mode : AgentMode
        Operational mode (default: PRAGMATIC).
    """

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 30,
        seed: int = 42,
        mode: AgentMode = AgentMode.PRAGMATIC,
    ) -> None:
        super().__init__(mode=mode)
        self._population_size = population_size
        self._generations = generations
        self._seed = seed

        # Register trigger conditions
        self._trigger_conditions = [
            TriggerCondition(
                name="sufficient_observations",
                check_fn_name="check_observation_count",
                priority=5,
                description=(
                    f"Activates when optimization history has >= {_MIN_OBSERVATIONS} observations"
                ),
            ),
            TriggerCondition(
                name="parameter_variation",
                check_fn_name="check_parameter_variation",
                priority=3,
                description="Activates when parameters show sufficient variation for regression",
            ),
        ]

    def name(self) -> str:
        """Return agent identifier."""
        return "symbolic_regression"

    def should_activate(self, context: AgentContext) -> bool:
        """Activate when optimization history has sufficient observations.

        Requires at least ``_MIN_OBSERVATIONS`` entries in the history,
        each containing a ``parameters`` dict and at least one KPI value.
        """
        if not context.optimization_history:
            return False
        return len(context.optimization_history) >= _MIN_OBSERVATIONS

    def validate_context(self, context: AgentContext) -> bool:
        """Validate that optimization_history contains parameters and KPI values.

        Each entry must have a ``parameters`` dict. At least one entry
        must have a KPI value (first numeric value outside ``parameters``).
        """
        if not context.optimization_history:
            return False

        for entry in context.optimization_history:
            if "parameters" not in entry or not isinstance(entry["parameters"], dict):
                return False

        # Check that at least one entry has a target value
        first = context.optimization_history[0]
        target_key = self._find_target_key(first)
        return target_key is not None

    def analyze(self, context: AgentContext) -> dict[str, Any]:
        """Run symbolic regression on the optimization history.

        Extracts parameter values as feature matrix X, the first KPI
        as target vector y, then runs EquationDiscovery to find
        interpretable equations.

        Returns
        -------
        dict[str, Any]
            Keys: ``equations``, ``best_equation``, ``pareto_front``,
            ``n_observations``, ``feature_names``, ``target_key``.
        """
        history = context.optimization_history
        if not history:
            return self._empty_result()

        # Extract parameter names from first entry
        first = history[0]
        if "parameters" not in first:
            return self._empty_result()

        param_names = list(first["parameters"].keys())
        if not param_names:
            return self._empty_result()

        # Find the target key
        target_key = self._find_target_key(first)
        if target_key is None:
            return self._empty_result()

        # Build X, y
        X: list[list[float]] = []
        y: list[float] = []

        for entry in history:
            params = entry.get("parameters", {})
            target_val = entry.get(target_key)

            if target_val is None:
                continue

            try:
                target_float = float(target_val)
            except (TypeError, ValueError):
                continue

            if not math.isfinite(target_float):
                continue

            row: list[float] = []
            valid = True
            for pname in param_names:
                val = params.get(pname)
                if val is None:
                    valid = False
                    break
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    valid = False
                    break
                if not math.isfinite(fval):
                    valid = False
                    break
                row.append(fval)

            if valid:
                X.append(row)
                y.append(target_float)

        if len(X) < 2:
            return self._empty_result(
                n_observations=len(X),
                feature_names=param_names,
                target_key=target_key,
            )

        # Check for constant y (no regression possible)
        y_set = set(y)
        if len(y_set) <= 1:
            return {
                "equations": [],
                "best_equation": None,
                "pareto_front": [],
                "n_observations": len(X),
                "feature_names": param_names,
                "target_key": target_key,
                "constant_target": True,
            }

        # Run equation discovery
        engine = EquationDiscovery(
            population_size=self._population_size,
            n_generations=self._generations,
            seed=self._seed,
        )

        pareto_front = engine.fit(X, y, var_names=param_names)
        best = engine.best_equation()

        # Build equations list
        equations: list[dict[str, Any]] = []
        for sol in pareto_front:
            equations.append({
                "equation": sol.equation_string,
                "mse": sol.mse,
                "complexity": sol.complexity,
            })

        best_eq: dict[str, Any] | None = None
        if best is not None:
            best_eq = {
                "equation": best.equation_string,
                "mse": best.mse,
                "complexity": best.complexity,
            }

        return {
            "equations": equations,
            "best_equation": best_eq,
            "pareto_front": pareto_front,
            "n_observations": len(X),
            "feature_names": param_names,
            "target_key": target_key,
        }

    def get_optimization_feedback(
        self, analysis_result: dict[str, Any]
    ) -> OptimizationFeedback | None:
        """Convert analysis results to optimization feedback.

        Returns feedback if a good equation is found (low complexity,
        reasonable MSE). Returns None otherwise.
        """
        best = analysis_result.get("best_equation")
        if best is None:
            return None

        complexity = best.get("complexity", float("inf"))
        mse = best.get("mse", float("inf"))
        equation_str = best.get("equation", "")

        if not equation_str:
            return None

        # Only report equations with reasonable complexity and fitness
        if complexity > _MAX_GOOD_COMPLEXITY:
            return None

        if not math.isfinite(mse) or mse > _MAX_REASONABLE_MSE:
            return None

        # Confidence based on MSE -- lower MSE = higher confidence
        # Use a sigmoid-like mapping: confidence = 1 / (1 + mse)
        # Cap between 0.3 and 0.95
        raw_conf = 1.0 / (1.0 + mse)
        confidence = max(0.3, min(0.95, raw_conf))

        # Penalize high complexity
        complexity_penalty = max(0.0, (complexity - 3) * 0.05)
        confidence = max(0.3, confidence - complexity_penalty)

        target_key = analysis_result.get("target_key", "y")
        feature_names = analysis_result.get("feature_names", [])
        n_obs = analysis_result.get("n_observations", 0)

        return OptimizationFeedback(
            agent_name=self.name(),
            feedback_type="hypothesis",
            confidence=confidence,
            payload={
                "equation": equation_str,
                "mse": mse,
                "complexity": complexity,
                "target": target_key,
                "features": feature_names,
                "n_observations": n_obs,
            },
            reasoning=(
                f"Discovered equation: {equation_str} "
                f"(MSE={mse:.4g}, complexity={complexity}) "
                f"relating {', '.join(feature_names)} to {target_key} "
                f"from {n_obs} observations."
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_target_key(entry: dict[str, Any]) -> str | None:
        """Find the first KPI/target key in a history entry.

        Looks for common target key names, then falls back to the first
        numeric value that is not in ``parameters``.
        """
        # Common target key names
        for key in ("y", "objective", "kpi", "target", "value", "score"):
            if key in entry and key != "parameters":
                return key

        # Check for a 'kpis' dict and return its first key
        if "kpis" in entry and isinstance(entry["kpis"], dict):
            kpi_keys = list(entry["kpis"].keys())
            if kpi_keys:
                return "kpis"

        # Fallback: first numeric key that is not 'parameters'
        for key, val in entry.items():
            if key == "parameters":
                continue
            if isinstance(val, (int, float)):
                return key

        return None

    @staticmethod
    def _empty_result(
        n_observations: int = 0,
        feature_names: list[str] | None = None,
        target_key: str | None = None,
    ) -> dict[str, Any]:
        """Return an empty analysis result."""
        return {
            "equations": [],
            "best_equation": None,
            "pareto_front": [],
            "n_observations": n_observations,
            "feature_names": feature_names or [],
            "target_key": target_key,
        }
