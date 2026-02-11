"""ProblemBuilder — fluent API that produces an OptimizationSpec.

Supports both manual specification and data-driven inference from an
ExperimentStore, with automatic type/bound detection and direction
heuristics.
"""

from __future__ import annotations

from typing import Any

from optimization_copilot.dsl.spec import (
    BudgetDef,
    Direction,
    DiversityStrategy,
    ObjectiveDef,
    OptimizationSpec,
    ParallelDef,
    ParameterDef,
    ParamType,
    RiskPreference,
)
from optimization_copilot.dsl.validation import DSLValidationError, SpecValidator
from optimization_copilot.problem_builder.models import BuilderError
from optimization_copilot.store.store import ExperimentStore


# ── Direction inference word lists ─────────────────────────────


_MINIMIZE_KEYWORDS = {
    "error", "loss", "cost", "waste", "deviation",
}

_MAXIMIZE_KEYWORDS = {
    "yield", "purity", "efficiency", "score", "performance",
    "conversion", "selectivity",
}


# ── ProblemBuilder ─────────────────────────────────────────────


class ProblemBuilder:
    """Fluent builder for constructing an OptimizationSpec.

    Usage::

        spec = (
            ProblemBuilder("my-campaign", store=store)
            .set_inputs(["temperature", "pressure"])
            .set_objective("yield")
            .set_budget(max_iterations=50)
            .build()
        )
    """

    def __init__(
        self,
        campaign_id: str,
        *,
        store: ExperimentStore | None = None,
    ) -> None:
        self._campaign_id = campaign_id
        self._store = store

        # Internal accumulators.
        self._parameters: dict[str, ParameterDef] = {}
        self._objectives: dict[str, ObjectiveDef] = {}
        self._budget = BudgetDef()
        self._risk_preference = RiskPreference.MODERATE
        self._parallel = ParallelDef()

    # ── Input / parameter methods ─────────────────────────

    def set_inputs(self, names: list[str]) -> ProblemBuilder:
        """Add multiple inputs, auto-inferring types and bounds from store data."""
        for name in names:
            self.set_input(name)
        return self

    def set_input(
        self,
        name: str,
        param_type: ParamType | None = None,
        lower: float | None = None,
        upper: float | None = None,
        categories: list[str] | None = None,
        step_size: float | None = None,
    ) -> ProblemBuilder:
        """Add a single input parameter with optional manual overrides.

        When connected to a store, unspecified fields are inferred from
        the campaign data.
        """
        # Infer from store if fields are not provided.
        if param_type is None:
            param_type = self._infer_param_type(name)
        if param_type == ParamType.CATEGORICAL:
            if categories is None:
                categories = self._infer_categories(name)
        else:
            if lower is None or upper is None:
                inferred_lower, inferred_upper = self._infer_bounds(name)
                if lower is None:
                    lower = inferred_lower
                if upper is None:
                    upper = inferred_upper

        self._parameters[name] = ParameterDef(
            name=name,
            type=param_type,
            lower=lower,
            upper=upper,
            categories=categories,
            step_size=step_size,
        )
        return self

    # ── Objective methods ─────────────────────────────────

    def set_objective(
        self,
        name: str,
        direction: Direction | None = None,
    ) -> ProblemBuilder:
        """Add an optimization objective.

        If *direction* is not specified, it is inferred from the
        objective name using keyword heuristics.
        """
        if direction is None:
            direction = self._infer_direction(name)
        self._objectives[name] = ObjectiveDef(
            name=name,
            direction=direction,
        )
        return self

    def add_constraint(
        self,
        name: str,
        lower: float | None = None,
        upper: float | None = None,
    ) -> ProblemBuilder:
        """Add a constraint to an existing objective."""
        if name not in self._objectives:
            raise BuilderError(
                [f"Cannot add constraint: objective '{name}' has not been set."]
            )
        obj = self._objectives[name]
        self._objectives[name] = ObjectiveDef(
            name=obj.name,
            direction=obj.direction,
            constraint_lower=lower,
            constraint_upper=upper,
            is_primary=obj.is_primary,
            weight=obj.weight,
        )
        return self

    # ── Budget / preference methods ───────────────────────

    def set_budget(
        self,
        max_iterations: int | None = None,
        max_samples: int | None = None,
        max_time_seconds: float | None = None,
        max_cost: float | None = None,
    ) -> ProblemBuilder:
        """Set budget constraints for the optimization campaign."""
        self._budget = BudgetDef(
            max_iterations=max_iterations,
            max_samples=max_samples,
            max_time_seconds=max_time_seconds,
            max_cost=max_cost,
        )
        return self

    def set_risk_preference(self, preference: str) -> ProblemBuilder:
        """Set risk preference: 'conservative', 'moderate', or 'aggressive'."""
        self._risk_preference = RiskPreference(preference)
        return self

    def set_parallel(
        self,
        batch_size: int | None = None,
        diversity_strategy: str | None = None,
    ) -> ProblemBuilder:
        """Configure parallel evaluation settings."""
        bs = batch_size if batch_size is not None else self._parallel.batch_size
        ds = (
            DiversityStrategy(diversity_strategy)
            if diversity_strategy is not None
            else self._parallel.diversity_strategy
        )
        self._parallel = ParallelDef(batch_size=bs, diversity_strategy=ds)
        return self

    # ── Build ─────────────────────────────────────────────

    def build(self) -> OptimizationSpec:
        """Validate and return the OptimizationSpec.

        Raises:
            BuilderError: If the spec fails validation.
        """
        spec = OptimizationSpec(
            campaign_id=self._campaign_id,
            parameters=list(self._parameters.values()),
            objectives=list(self._objectives.values()),
            budget=self._budget,
            risk_preference=self._risk_preference,
            parallel=self._parallel,
        )
        try:
            SpecValidator().validate_or_raise(spec)
        except DSLValidationError as exc:
            raise BuilderError(exc.errors) from exc
        return spec

    # ── Inference helpers ─────────────────────────────────

    def _infer_param_type(self, column_name: str) -> ParamType:
        """Infer parameter type from store data.

        Rules:
        - All values are int and unique_count < 20 -> DISCRETE
        - All values are strings -> CATEGORICAL
        - Otherwise -> CONTINUOUS
        """
        values = self._get_column_values(column_name)
        if not values:
            return ParamType.CONTINUOUS

        non_null = [v for v in values if v is not None]
        if not non_null:
            return ParamType.CONTINUOUS

        # Check if all values are strings.
        if all(isinstance(v, str) for v in non_null):
            return ParamType.CATEGORICAL

        # Check if all values are int-like and unique count < 20.
        all_int = True
        for v in non_null:
            if isinstance(v, bool):
                all_int = False
                break
            if isinstance(v, int):
                continue
            if isinstance(v, float) and v == int(v):
                continue
            all_int = False
            break

        if all_int:
            unique_count = len(set(int(v) if isinstance(v, float) else v for v in non_null))
            if unique_count < 20:
                return ParamType.DISCRETE

        return ParamType.CONTINUOUS

    def _infer_bounds(self, column_name: str) -> tuple[float, float]:
        """Infer min/max bounds from store data with 10% padding."""
        values = self._get_column_values(column_name)
        numeric: list[float] = []
        for v in values:
            if v is None:
                continue
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                numeric.append(float(v))

        if not numeric:
            return (0.0, 1.0)

        min_val = min(numeric)
        max_val = max(numeric)
        span = max_val - min_val

        if span == 0:
            # All values identical — create a small range around the value.
            if min_val == 0:
                return (-0.1, 0.1)
            padding = abs(min_val) * 0.1
            return (min_val - padding, max_val + padding)

        padding = span * 0.1
        return (min_val - padding, max_val + padding)

    def _infer_direction(self, name: str) -> Direction:
        """Infer optimization direction from the objective name.

        Keywords like 'error', 'loss', 'cost' map to MINIMIZE.
        Keywords like 'yield', 'purity', 'efficiency' map to MAXIMIZE.
        Default: MINIMIZE.
        """
        name_lower = name.lower()
        for keyword in _MAXIMIZE_KEYWORDS:
            if keyword in name_lower:
                return Direction.MAXIMIZE
        for keyword in _MINIMIZE_KEYWORDS:
            if keyword in name_lower:
                return Direction.MINIMIZE
        return Direction.MINIMIZE

    def _infer_categories(self, column_name: str) -> list[str]:
        """Infer unique string values from store for categorical parameters."""
        values = self._get_column_values(column_name)
        unique: list[str] = []
        seen: set[str] = set()
        for v in values:
            if v is None:
                continue
            s = str(v)
            if s not in seen:
                unique.append(s)
                seen.add(s)
        return sorted(unique)

    def _get_column_values(self, column_name: str) -> list[Any]:
        """Retrieve column values from the store for the current campaign."""
        if self._store is None:
            return []
        return self._store.column_values(self._campaign_id, column_name)
