"""Comprehensive tests for the Problem Builder module.

Covers BuilderError, ColumnSuggestion, ProblemSuggestions, ProblemBuilder,
and ProblemGuide with ~50 tests exercising inference, fluent API, validation,
and end-to-end guide workflows.
"""

from __future__ import annotations

import pytest

from optimization_copilot.problem_builder import (
    BuilderError,
    ColumnSuggestion,
    ProblemBuilder,
    ProblemGuide,
    ProblemSuggestions,
)
from optimization_copilot.store import ExperimentStore, Experiment
from optimization_copilot.dsl.spec import (
    Direction,
    DiversityStrategy,
    ParamType,
    RiskPreference,
)
from optimization_copilot.ingestion.models import ColumnRole


# -- Test helpers ----------------------------------------------------------


def _make_experiment(
    campaign_id: str,
    iteration: int,
    parameters: dict,
    kpi_values: dict,
) -> Experiment:
    """Create an Experiment with sensible defaults."""
    return Experiment(
        experiment_id=f"{campaign_id}-exp-{iteration}",
        campaign_id=campaign_id,
        iteration=iteration,
        parameters=parameters,
        kpi_values=kpi_values,
        metadata={},
    )


def _make_store() -> ExperimentStore:
    """Return an ExperimentStore with 5 experiments for campaign 'test'.

    Parameters:
      - x: continuous floats in [0.1, 0.9]
      - temperature: integer-like values in [50, 90]
      - catalyst: categorical strings "A", "B", "C"
    KPIs:
      - yield: floats in [50.0, 90.0]
      - error: floats in [0.01, 0.05]
    """
    store = ExperimentStore()
    data = [
        {"x": 0.1, "temperature": 50, "catalyst": "A", "yield": 50.0, "error": 0.05},
        {"x": 0.3, "temperature": 60, "catalyst": "B", "yield": 60.0, "error": 0.04},
        {"x": 0.5, "temperature": 70, "catalyst": "C", "yield": 70.0, "error": 0.03},
        {"x": 0.7, "temperature": 80, "catalyst": "A", "yield": 80.0, "error": 0.02},
        {"x": 0.9, "temperature": 90, "catalyst": "B", "yield": 90.0, "error": 0.01},
    ]
    for i, d in enumerate(data):
        exp = _make_experiment(
            campaign_id="test",
            iteration=i + 1,
            parameters={"x": d["x"], "temperature": d["temperature"], "catalyst": d["catalyst"]},
            kpi_values={"yield": d["yield"], "error": d["error"]},
        )
        store.add_experiment(exp)
    return store


# -- TestBuilderError ------------------------------------------------------


class TestBuilderError:
    """Tests for the BuilderError exception class."""

    def test_stores_errors_list(self):
        errors = ["error one", "error two"]
        exc = BuilderError(errors)
        assert exc.errors == errors

    def test_message_formatting(self):
        errors = ["missing param", "bad objective"]
        exc = BuilderError(errors)
        msg = str(exc)
        assert "2 error(s)" in msg
        assert "- missing param" in msg
        assert "- bad objective" in msg

    def test_is_exception_subclass(self):
        exc = BuilderError(["test"])
        assert isinstance(exc, Exception)


# -- TestColumnSuggestion --------------------------------------------------


class TestColumnSuggestion:
    """Tests for the ColumnSuggestion dataclass."""

    def test_create_with_all_fields(self):
        cs = ColumnSuggestion(
            column_name="x",
            suggested_role=ColumnRole.PARAMETER,
            confidence=0.9,
            param_type=ParamType.CONTINUOUS,
            direction=None,
            lower=0.0,
            upper=1.0,
            categories=None,
            reason="Numeric column",
        )
        assert cs.column_name == "x"
        assert cs.suggested_role == ColumnRole.PARAMETER
        assert cs.confidence == 0.9
        assert cs.param_type == ParamType.CONTINUOUS
        assert cs.lower == 0.0
        assert cs.upper == 1.0

    def test_to_dict_serialization(self):
        cs = ColumnSuggestion(
            column_name="yield",
            suggested_role=ColumnRole.KPI,
            confidence=0.85,
            param_type=None,
            direction=Direction.MAXIMIZE,
            lower=None,
            upper=None,
            categories=None,
            reason="KPI column",
        )
        d = cs.to_dict()
        assert d["column_name"] == "yield"
        assert d["suggested_role"] == "kpi"
        assert d["confidence"] == 0.85
        assert d["direction"] == "maximize"
        assert d["param_type"] is None

    def test_to_dict_with_none_optional_fields(self):
        cs = ColumnSuggestion(
            column_name="z",
            suggested_role=ColumnRole.METADATA,
            confidence=0.5,
        )
        d = cs.to_dict()
        assert d["param_type"] is None
        assert d["direction"] is None
        assert d["lower"] is None
        assert d["upper"] is None
        assert d["categories"] is None

    def test_default_values(self):
        cs = ColumnSuggestion(
            column_name="col",
            suggested_role=ColumnRole.UNKNOWN,
            confidence=0.0,
        )
        assert cs.param_type is None
        assert cs.direction is None
        assert cs.lower is None
        assert cs.upper is None
        assert cs.categories is None
        assert cs.reason == ""


# -- TestProblemSuggestions ------------------------------------------------


class TestProblemSuggestions:
    """Tests for the ProblemSuggestions dataclass."""

    def test_create_with_suggestions(self):
        cs = ColumnSuggestion(
            column_name="x",
            suggested_role=ColumnRole.PARAMETER,
            confidence=0.9,
        )
        ps = ProblemSuggestions(
            campaign_id="camp",
            n_rows=10,
            n_columns=3,
            all_suggestions=[cs],
            input_suggestions=[cs],
        )
        assert ps.campaign_id == "camp"
        assert ps.n_rows == 10
        assert len(ps.all_suggestions) == 1
        assert len(ps.input_suggestions) == 1

    def test_to_dict_serialization(self):
        cs = ColumnSuggestion(
            column_name="y",
            suggested_role=ColumnRole.KPI,
            confidence=0.8,
        )
        ps = ProblemSuggestions(
            campaign_id="camp",
            n_rows=5,
            n_columns=2,
            all_suggestions=[cs],
            objective_suggestions=[cs],
        )
        d = ps.to_dict()
        assert d["campaign_id"] == "camp"
        assert d["n_rows"] == 5
        assert d["n_columns"] == 2
        assert len(d["all_suggestions"]) == 1
        assert len(d["objective_suggestions"]) == 1
        assert d["all_suggestions"][0]["column_name"] == "y"

    def test_empty_suggestions(self):
        ps = ProblemSuggestions(
            campaign_id="empty",
            n_rows=0,
            n_columns=0,
            all_suggestions=[],
        )
        assert ps.input_suggestions == []
        assert ps.objective_suggestions == []
        assert ps.metadata_suggestions == []
        d = ps.to_dict()
        assert d["all_suggestions"] == []


# -- TestProblemBuilder ----------------------------------------------------


class TestProblemBuilder:
    """Tests for the ProblemBuilder fluent API."""

    def test_build_with_minimal_valid_spec(self):
        """Manual inputs + objective + budget produces a valid spec."""
        spec = (
            ProblemBuilder("camp-1")
            .set_input("x", param_type=ParamType.CONTINUOUS, lower=0.0, upper=1.0)
            .set_objective("yield", direction=Direction.MAXIMIZE)
            .set_budget(max_iterations=50)
            .build()
        )
        assert spec.campaign_id == "camp-1"
        assert len(spec.parameters) == 1
        assert len(spec.objectives) == 1
        assert spec.budget.max_iterations == 50

    def test_fluent_api_chaining_returns_self(self):
        builder = ProblemBuilder("camp")
        result = builder.set_input("x", param_type=ParamType.CONTINUOUS, lower=0.0, upper=1.0)
        assert result is builder
        result2 = builder.set_objective("y", direction=Direction.MINIMIZE)
        assert result2 is builder
        result3 = builder.set_budget(max_iterations=10)
        assert result3 is builder
        result4 = builder.set_risk_preference("aggressive")
        assert result4 is builder
        result5 = builder.set_parallel(batch_size=4)
        assert result5 is builder

    def test_set_inputs_auto_infers_types_from_store(self):
        """x -> CONTINUOUS, temperature -> DISCRETE, catalyst -> CATEGORICAL."""
        store = _make_store()
        builder = ProblemBuilder("test", store=store)
        builder.set_inputs(["x", "temperature", "catalyst"])
        builder.set_objective("yield", direction=Direction.MAXIMIZE)
        spec = builder.build()

        param_map = {p.name: p for p in spec.parameters}
        assert param_map["x"].type == ParamType.CONTINUOUS
        assert param_map["temperature"].type == ParamType.DISCRETE
        assert param_map["catalyst"].type == ParamType.CATEGORICAL

    def test_set_input_explicit_param_type_overrides_inference(self):
        store = _make_store()
        builder = ProblemBuilder("test", store=store)
        builder.set_input("temperature", param_type=ParamType.CONTINUOUS)
        builder.set_objective("yield", direction=Direction.MAXIMIZE)
        spec = builder.build()
        assert spec.parameters[0].type == ParamType.CONTINUOUS

    def test_set_input_explicit_bounds_overrides_inference(self):
        store = _make_store()
        builder = ProblemBuilder("test", store=store)
        builder.set_input("x", lower=-5.0, upper=5.0)
        builder.set_objective("yield", direction=Direction.MAXIMIZE)
        spec = builder.build()
        assert spec.parameters[0].lower == -5.0
        assert spec.parameters[0].upper == 5.0

    def test_set_objective_auto_infers_direction_yield_maximize(self):
        spec = (
            ProblemBuilder("camp")
            .set_input("x", param_type=ParamType.CONTINUOUS, lower=0.0, upper=1.0)
            .set_objective("yield")
            .set_budget(max_iterations=10)
            .build()
        )
        assert spec.objectives[0].direction == Direction.MAXIMIZE

    def test_set_objective_auto_infers_direction_error_minimize(self):
        spec = (
            ProblemBuilder("camp")
            .set_input("x", param_type=ParamType.CONTINUOUS, lower=0.0, upper=1.0)
            .set_objective("error")
            .set_budget(max_iterations=10)
            .build()
        )
        assert spec.objectives[0].direction == Direction.MINIMIZE

    def test_set_objective_explicit_direction_overrides_inference(self):
        spec = (
            ProblemBuilder("camp")
            .set_input("x", param_type=ParamType.CONTINUOUS, lower=0.0, upper=1.0)
            .set_objective("yield", direction=Direction.MINIMIZE)
            .set_budget(max_iterations=10)
            .build()
        )
        assert spec.objectives[0].direction == Direction.MINIMIZE

    def test_add_constraint_adds_bounds_to_objective(self):
        spec = (
            ProblemBuilder("camp")
            .set_input("x", param_type=ParamType.CONTINUOUS, lower=0.0, upper=1.0)
            .set_objective("yield", direction=Direction.MAXIMIZE)
            .add_constraint("yield", lower=50.0, upper=100.0)
            .set_budget(max_iterations=10)
            .build()
        )
        obj = spec.objectives[0]
        assert obj.constraint_lower == 50.0
        assert obj.constraint_upper == 100.0

    def test_add_constraint_on_nonexistent_objective_raises(self):
        builder = ProblemBuilder("camp")
        with pytest.raises(BuilderError) as exc_info:
            builder.add_constraint("nonexistent", lower=0.0)
        assert "nonexistent" in str(exc_info.value)

    def test_set_budget_configures_correctly(self):
        spec = (
            ProblemBuilder("camp")
            .set_input("x", param_type=ParamType.CONTINUOUS, lower=0.0, upper=1.0)
            .set_objective("y", direction=Direction.MINIMIZE)
            .set_budget(
                max_iterations=100,
                max_samples=500,
                max_time_seconds=3600.0,
                max_cost=1000.0,
            )
            .build()
        )
        assert spec.budget.max_iterations == 100
        assert spec.budget.max_samples == 500
        assert spec.budget.max_time_seconds == 3600.0
        assert spec.budget.max_cost == 1000.0

    def test_set_risk_preference(self):
        spec = (
            ProblemBuilder("camp")
            .set_input("x", param_type=ParamType.CONTINUOUS, lower=0.0, upper=1.0)
            .set_objective("y", direction=Direction.MINIMIZE)
            .set_budget(max_iterations=10)
            .set_risk_preference("aggressive")
            .build()
        )
        assert spec.risk_preference == RiskPreference.AGGRESSIVE

    def test_set_parallel_batch_size_and_diversity(self):
        spec = (
            ProblemBuilder("camp")
            .set_input("x", param_type=ParamType.CONTINUOUS, lower=0.0, upper=1.0)
            .set_objective("y", direction=Direction.MINIMIZE)
            .set_budget(max_iterations=10)
            .set_parallel(batch_size=4, diversity_strategy="coverage")
            .build()
        )
        assert spec.parallel.batch_size == 4
        assert spec.parallel.diversity_strategy == DiversityStrategy.COVERAGE

    def test_build_validates_and_returns_spec(self):
        spec = (
            ProblemBuilder("camp")
            .set_input("x", param_type=ParamType.CONTINUOUS, lower=0.0, upper=1.0)
            .set_objective("y", direction=Direction.MINIMIZE)
            .set_budget(max_iterations=10)
            .build()
        )
        assert spec.campaign_id == "camp"
        assert isinstance(spec.parameters, list)
        assert isinstance(spec.objectives, list)

    def test_build_raises_when_no_parameters(self):
        builder = ProblemBuilder("camp").set_objective("y", direction=Direction.MINIMIZE)
        with pytest.raises(BuilderError) as exc_info:
            builder.build()
        assert any("parameter" in e.lower() for e in exc_info.value.errors)

    def test_build_raises_when_no_objectives(self):
        builder = ProblemBuilder("camp").set_input(
            "x", param_type=ParamType.CONTINUOUS, lower=0.0, upper=1.0
        )
        with pytest.raises(BuilderError) as exc_info:
            builder.build()
        assert any("objective" in e.lower() for e in exc_info.value.errors)

    def test_build_raises_with_empty_campaign_id(self):
        builder = (
            ProblemBuilder("")
            .set_input("x", param_type=ParamType.CONTINUOUS, lower=0.0, upper=1.0)
            .set_objective("y", direction=Direction.MINIMIZE)
        )
        with pytest.raises(BuilderError) as exc_info:
            builder.build()
        assert any("campaign_id" in e for e in exc_info.value.errors)

    def test_infer_bounds_returns_min_max_with_padding(self):
        store = _make_store()
        builder = ProblemBuilder("test", store=store)
        lo, hi = builder._infer_bounds("x")
        # Values: 0.1, 0.3, 0.5, 0.7, 0.9
        # min=0.1, max=0.9, span=0.8, padding=0.08
        assert lo == pytest.approx(0.1 - 0.08, abs=1e-9)
        assert hi == pytest.approx(0.9 + 0.08, abs=1e-9)

    def test_infer_bounds_returns_default_when_no_data(self):
        store = ExperimentStore()
        builder = ProblemBuilder("empty", store=store)
        lo, hi = builder._infer_bounds("nonexistent")
        assert lo == 0.0
        assert hi == 1.0

    def test_infer_direction_defaults_to_minimize_for_unknown(self):
        builder = ProblemBuilder("camp")
        direction = builder._infer_direction("unknown_metric")
        assert direction == Direction.MINIMIZE

    def test_infer_categories_returns_sorted_unique_strings(self):
        store = _make_store()
        builder = ProblemBuilder("test", store=store)
        cats = builder._infer_categories("catalyst")
        assert cats == ["A", "B", "C"]

    def test_builder_without_store_uses_defaults(self):
        """Without store, type defaults to CONTINUOUS and bounds to (0.0, 1.0)."""
        builder = ProblemBuilder("camp")
        builder.set_input("x")
        builder.set_objective("y", direction=Direction.MINIMIZE)
        spec = builder.build()
        param = spec.parameters[0]
        assert param.type == ParamType.CONTINUOUS
        assert param.lower == 0.0
        assert param.upper == 1.0


# -- TestProblemGuide ------------------------------------------------------


class TestProblemGuide:
    """Tests for the ProblemGuide guided mode."""

    def test_suggest_roles_returns_problem_suggestions(self):
        store = _make_store()
        guide = ProblemGuide(store, "test")
        suggestions = guide.suggest_roles()
        assert isinstance(suggestions, ProblemSuggestions)

    def test_suggest_roles_categorizes_columns(self):
        """Input/objective/metadata categories should be populated."""
        store = _make_store()
        guide = ProblemGuide(store, "test")
        suggestions = guide.suggest_roles()
        # Should have some suggestions in total
        assert len(suggestions.all_suggestions) > 0
        # At least some should be input or objective
        total_categorized = (
            len(suggestions.input_suggestions)
            + len(suggestions.objective_suggestions)
            + len(suggestions.metadata_suggestions)
        )
        assert total_categorized > 0

    def test_suggest_roles_on_empty_campaign(self):
        store = ExperimentStore()
        guide = ProblemGuide(store, "nonexistent")
        suggestions = guide.suggest_roles()
        assert suggestions.n_rows == 0
        assert suggestions.n_columns == 0
        assert suggestions.all_suggestions == []

    def test_accept_inputs_and_objective_and_build(self):
        """End-to-end: accept_inputs + accept_objective + build works."""
        store = _make_store()
        spec = (
            ProblemGuide(store, "test")
            .accept_inputs(["x", "temperature", "catalyst"])
            .accept_objective("yield")
            .set_budget(max_iterations=50)
            .build()
        )
        assert spec.campaign_id == "test"
        assert len(spec.parameters) == 3
        assert len(spec.objectives) == 1
        assert spec.objectives[0].direction == Direction.MAXIMIZE

    def test_accept_objective_with_explicit_direction(self):
        store = _make_store()
        spec = (
            ProblemGuide(store, "test")
            .accept_inputs(["x"])
            .accept_objective("yield", direction=Direction.MINIMIZE)
            .set_budget(max_iterations=10)
            .build()
        )
        assert spec.objectives[0].direction == Direction.MINIMIZE

    def test_set_budget_delegates_to_builder(self):
        store = _make_store()
        spec = (
            ProblemGuide(store, "test")
            .accept_inputs(["x"])
            .accept_objective("yield")
            .set_budget(max_iterations=42, max_samples=200)
            .build()
        )
        assert spec.budget.max_iterations == 42
        assert spec.budget.max_samples == 200

    def test_set_risk_preference_delegates_to_builder(self):
        store = _make_store()
        spec = (
            ProblemGuide(store, "test")
            .accept_inputs(["x"])
            .accept_objective("yield")
            .set_budget(max_iterations=10)
            .set_risk_preference("conservative")
            .build()
        )
        assert spec.risk_preference == RiskPreference.CONSERVATIVE

    def test_fluent_api_chaining(self):
        store = _make_store()
        guide = ProblemGuide(store, "test")
        result = guide.accept_inputs(["x"])
        assert result is guide
        result2 = guide.accept_objective("yield")
        assert result2 is guide
        result3 = guide.set_budget(max_iterations=10)
        assert result3 is guide
        result4 = guide.set_risk_preference("moderate")
        assert result4 is guide

    def test_guide_builds_valid_optimization_spec(self):
        store = _make_store()
        spec = (
            ProblemGuide(store, "test")
            .accept_inputs(["x", "temperature"])
            .accept_objective("error")
            .set_budget(max_iterations=100)
            .build()
        )
        from optimization_copilot.dsl.spec import OptimizationSpec

        assert isinstance(spec, OptimizationSpec)
        assert spec.campaign_id == "test"
        param_names = {p.name for p in spec.parameters}
        assert "x" in param_names
        assert "temperature" in param_names

    def test_infer_direction_yield_maximize(self):
        store = _make_store()
        guide = ProblemGuide(store, "test")
        direction = guide._infer_direction("yield")
        assert direction == Direction.MAXIMIZE

    def test_infer_direction_error_minimize(self):
        store = _make_store()
        guide = ProblemGuide(store, "test")
        direction = guide._infer_direction("error")
        assert direction == Direction.MINIMIZE

    def test_infer_param_type_from_profile_string_categorical(self):
        """String DataType -> CATEGORICAL."""
        from optimization_copilot.ingestion.models import ColumnProfile, DataType

        store = _make_store()
        guide = ProblemGuide(store, "test")
        profile = ColumnProfile(
            name="catalyst",
            data_type=DataType.STRING,
            n_unique=3,
            n_values=5,
        )
        result = guide._infer_param_type_from_profile(profile)
        assert result == ParamType.CATEGORICAL

    def test_infer_param_type_from_profile_integer_discrete(self):
        """Integer DataType with <20 unique -> DISCRETE."""
        from optimization_copilot.ingestion.models import ColumnProfile, DataType

        store = _make_store()
        guide = ProblemGuide(store, "test")
        profile = ColumnProfile(
            name="temperature",
            data_type=DataType.INTEGER,
            n_unique=5,
            n_values=5,
        )
        result = guide._infer_param_type_from_profile(profile)
        assert result == ParamType.DISCRETE

    def test_infer_param_type_from_profile_float_continuous(self):
        """Float DataType -> CONTINUOUS."""
        from optimization_copilot.ingestion.models import ColumnProfile, DataType

        store = _make_store()
        guide = ProblemGuide(store, "test")
        profile = ColumnProfile(
            name="x",
            data_type=DataType.FLOAT,
            n_unique=5,
            n_values=5,
        )
        result = guide._infer_param_type_from_profile(profile)
        assert result == ParamType.CONTINUOUS

    def test_n_rows_and_n_columns_match_data(self):
        store = _make_store()
        guide = ProblemGuide(store, "test")
        suggestions = guide.suggest_roles()
        assert suggestions.n_rows == 5
        # Columns: x, temperature, catalyst, yield, error = 5
        assert suggestions.n_columns == 5
