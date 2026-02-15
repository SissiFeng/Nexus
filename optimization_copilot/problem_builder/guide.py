"""ProblemGuide — guided/suggestive mode for building optimization specs.

Analyzes store data using ColumnProfiler and RoleInferenceEngine to
suggest column roles, then lets the user accept or override before
building a final OptimizationSpec.
"""

from __future__ import annotations

from typing import Any

from optimization_copilot.dsl.spec import (
    Direction,
    OptimizationSpec,
    ParamType,
)
from optimization_copilot.ingestion.models import ColumnRole, DataType
from optimization_copilot.ingestion.profiler import ColumnProfiler, RoleInferenceEngine
from optimization_copilot.problem_builder.builder import ProblemBuilder
from optimization_copilot.problem_builder.models import (
    ColumnSuggestion,
    ProblemSuggestions,
)
from optimization_copilot.store.store import ExperimentStore


# ── Direction inference word lists (same as builder) ───────────


_MINIMIZE_KEYWORDS = {
    "error", "loss", "cost", "waste", "deviation",
}

_MAXIMIZE_KEYWORDS = {
    "yield", "purity", "efficiency", "score", "performance",
    "conversion", "selectivity",
}


# ── ProblemGuide ───────────────────────────────────────────────


class ProblemGuide:
    """Guided problem definition with role suggestions.

    Usage::

        guide = ProblemGuide(store, "my-campaign")
        suggestions = guide.suggest_roles()
        # User reviews suggestions, then accepts:
        spec = (
            guide
            .accept_inputs(["temperature", "pressure"])
            .accept_objective("yield")
            .set_budget(max_iterations=50)
            .build()
        )
    """

    def __init__(
        self,
        store: ExperimentStore,
        campaign_id: str,
    ) -> None:
        self._store = store
        self._campaign_id = campaign_id
        self._builder = ProblemBuilder(
            campaign_id, store=store,
        )

    # ── Suggestion generation ─────────────────────────────

    def suggest_roles(self) -> ProblemSuggestions:
        """Analyze store data and suggest column roles.

        Uses ColumnProfiler and RoleInferenceEngine on data extracted
        from the store, then enriches parameter suggestions with
        param_type/bounds and KPI suggestions with direction.
        """
        rows = self._build_rows()
        if not rows:
            return ProblemSuggestions(
                campaign_id=self._campaign_id,
                n_rows=0,
                n_columns=0,
                all_suggestions=[],
            )

        # Profile and infer roles.
        profiler = ColumnProfiler()
        profiles = profiler.profile_columns(rows)
        engine = RoleInferenceEngine()
        profiles = engine.infer_roles(profiles)

        # Convert profiles to suggestions.
        all_suggestions: list[ColumnSuggestion] = []
        input_suggestions: list[ColumnSuggestion] = []
        objective_suggestions: list[ColumnSuggestion] = []
        metadata_suggestions: list[ColumnSuggestion] = []

        for profile in profiles:
            suggestion = self._profile_to_suggestion(profile)
            all_suggestions.append(suggestion)

            if profile.inferred_role == ColumnRole.PARAMETER:
                input_suggestions.append(suggestion)
            elif profile.inferred_role == ColumnRole.KPI:
                objective_suggestions.append(suggestion)
            elif profile.inferred_role in (
                ColumnRole.METADATA,
                ColumnRole.TIMESTAMP,
                ColumnRole.ITERATION,
                ColumnRole.IDENTIFIER,
            ):
                metadata_suggestions.append(suggestion)

        n_columns = len(profiles)

        return ProblemSuggestions(
            campaign_id=self._campaign_id,
            n_rows=len(rows),
            n_columns=n_columns,
            all_suggestions=all_suggestions,
            input_suggestions=input_suggestions,
            objective_suggestions=objective_suggestions,
            metadata_suggestions=metadata_suggestions,
        )

    # ── Accept / override methods ─────────────────────────

    def accept_inputs(self, names: list[str]) -> ProblemGuide:
        """Accept specific columns as input parameters."""
        self._builder.set_inputs(names)
        return self

    def accept_objective(
        self,
        name: str,
        direction: Direction | None = None,
    ) -> ProblemGuide:
        """Accept a column as an optimization objective."""
        self._builder.set_objective(name, direction=direction)
        return self

    def set_budget(
        self,
        max_iterations: int | None = None,
        max_samples: int | None = None,
        max_time_seconds: float | None = None,
        max_cost: float | None = None,
    ) -> ProblemGuide:
        """Set budget constraints."""
        self._builder.set_budget(
            max_iterations=max_iterations,
            max_samples=max_samples,
            max_time_seconds=max_time_seconds,
            max_cost=max_cost,
        )
        return self

    def set_risk_preference(self, preference: str) -> ProblemGuide:
        """Set risk preference: 'conservative', 'moderate', or 'aggressive'."""
        self._builder.set_risk_preference(preference)
        return self

    def build(self) -> OptimizationSpec:
        """Build the OptimizationSpec by delegating to the internal ProblemBuilder."""
        return self._builder.build()

    # ── Internal helpers ──────────────────────────────────

    def _build_rows(self) -> list[dict[str, Any]]:
        """Construct tabular rows from store experiments for the profiler."""
        experiments = self._store.get_by_campaign(self._campaign_id)
        rows: list[dict[str, Any]] = []
        for exp in experiments:
            row: dict[str, Any] = {}
            row.update(exp.parameters)
            row.update(exp.kpi_values)
            row.update(exp.metadata)
            rows.append(row)
        return rows

    def _profile_to_suggestion(self, profile: Any) -> ColumnSuggestion:
        """Convert a ColumnProfile with inferred role to a ColumnSuggestion."""
        param_type: ParamType | None = None
        direction: Direction | None = None
        lower: float | None = None
        upper: float | None = None
        categories: list[str] | None = None
        reason = ""

        if profile.inferred_role == ColumnRole.PARAMETER:
            # Infer param_type.
            param_type = self._infer_param_type_from_profile(profile)
            reason = f"Numeric column suitable as {param_type.value} parameter"

            if param_type == ParamType.CATEGORICAL:
                categories = self._infer_categories(profile.name)
                reason = f"String column with {len(categories)} categories"
            else:
                lower, upper = self._infer_bounds(profile.name)
                reason = f"Numeric column with range [{lower:.4g}, {upper:.4g}]"

        elif profile.inferred_role == ColumnRole.KPI:
            direction = self._infer_direction(profile.name)
            reason = f"KPI column, suggested direction: {direction.value}"

        elif profile.inferred_role == ColumnRole.METADATA:
            reason = "Metadata column (non-parameter, non-KPI)"

        elif profile.inferred_role == ColumnRole.TIMESTAMP:
            reason = "Timestamp column"

        elif profile.inferred_role == ColumnRole.ITERATION:
            reason = "Iteration/index column"

        elif profile.inferred_role == ColumnRole.IDENTIFIER:
            reason = "Identifier column"

        else:
            reason = "Unknown role"

        return ColumnSuggestion(
            column_name=profile.name,
            suggested_role=profile.inferred_role,
            confidence=profile.role_confidence,
            param_type=param_type,
            direction=direction,
            lower=lower,
            upper=upper,
            categories=categories,
            reason=reason,
        )

    def _infer_param_type_from_profile(self, profile: Any) -> ParamType:
        """Infer ParamType from a ColumnProfile."""
        if profile.data_type == DataType.STRING:
            return ParamType.CATEGORICAL
        if profile.data_type == DataType.INTEGER and profile.n_unique < 20:
            return ParamType.DISCRETE
        return ParamType.CONTINUOUS

    def _infer_bounds(self, column_name: str) -> tuple[float, float]:
        """Infer bounds from store data with 10% padding. Delegates to builder."""
        return self._builder._infer_bounds(column_name)

    def _infer_categories(self, column_name: str) -> list[str]:
        """Infer categories from store data. Delegates to builder."""
        return self._builder._infer_categories(column_name)

    def _infer_direction(self, name: str) -> Direction:
        """Infer optimization direction from objective name."""
        name_lower = name.lower()
        for keyword in _MAXIMIZE_KEYWORDS:
            if keyword in name_lower:
                return Direction.MAXIMIZE
        for keyword in _MINIMIZE_KEYWORDS:
            if keyword in name_lower:
                return Direction.MINIMIZE
        return Direction.MINIMIZE
