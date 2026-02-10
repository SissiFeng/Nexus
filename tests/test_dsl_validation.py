"""Tests for DSL spec validation."""

import pytest

from optimization_copilot.dsl.spec import (
    BudgetDef,
    ConditionDef,
    Direction,
    ObjectiveDef,
    OptimizationSpec,
    ParallelDef,
    ParameterDef,
    ParamType,
    RiskPreference,
)
from optimization_copilot.dsl.validation import (
    DSLValidationError,
    SpecValidator,
    ValidationResult,
)


# ── Helpers ──────────────────────────────────────────────


def _make_continuous_param(
    name: str = "x1",
    lower: float = 0.0,
    upper: float = 1.0,
    **kwargs,
) -> ParameterDef:
    return ParameterDef(
        name=name,
        type=ParamType.CONTINUOUS,
        lower=lower,
        upper=upper,
        **kwargs,
    )


def _make_discrete_param(
    name: str = "n",
    lower: float = 1.0,
    upper: float = 10.0,
    **kwargs,
) -> ParameterDef:
    return ParameterDef(
        name=name,
        type=ParamType.DISCRETE,
        lower=lower,
        upper=upper,
        **kwargs,
    )


def _make_categorical_param(
    name: str = "method",
    categories: list[str] | None = None,
    **kwargs,
) -> ParameterDef:
    if categories is None:
        categories = ["adam", "sgd", "rmsprop"]
    return ParameterDef(
        name=name,
        type=ParamType.CATEGORICAL,
        categories=categories,
        **kwargs,
    )


def _make_objective(
    name: str = "loss",
    direction: Direction = Direction.MINIMIZE,
    **kwargs,
) -> ObjectiveDef:
    return ObjectiveDef(name=name, direction=direction, **kwargs)


def _make_valid_spec(**kwargs) -> OptimizationSpec:
    defaults = {
        "campaign_id": "valid-campaign",
        "parameters": [
            _make_continuous_param(name="x1"),
            _make_continuous_param(name="x2", lower=-1.0, upper=1.0),
        ],
        "objectives": [_make_objective()],
        "budget": BudgetDef(max_samples=100),
    }
    defaults.update(kwargs)
    return OptimizationSpec(**defaults)


# ── TestValidSpec ────────────────────────────────────────


class TestValidSpec:
    def test_valid_spec_passes(self):
        validator = SpecValidator()
        result = validator.validate(_make_valid_spec())
        assert result.valid is True
        assert result.errors == []

    def test_valid_spec_no_warnings_with_budget(self):
        validator = SpecValidator()
        result = validator.validate(_make_valid_spec())
        assert result.valid is True
        # With 2 active params and a budget set, no warnings expected.
        assert result.warnings == []

    def test_valid_spec_with_all_types(self):
        spec = _make_valid_spec(
            parameters=[
                _make_continuous_param(name="cont"),
                _make_discrete_param(name="disc"),
                _make_categorical_param(name="cat"),
            ]
        )
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is True

    def test_valid_spec_multi_objective(self):
        spec = _make_valid_spec(
            objectives=[
                _make_objective(name="loss", direction=Direction.MINIMIZE),
                _make_objective(name="throughput", direction=Direction.MAXIMIZE),
            ]
        )
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is True


# ── TestMissingFields ────────────────────────────────────


class TestMissingFields:
    def test_no_parameters(self):
        spec = _make_valid_spec(parameters=[])
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("At least one parameter" in e for e in result.errors)

    def test_no_objectives(self):
        spec = _make_valid_spec(objectives=[])
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("At least one objective" in e for e in result.errors)

    def test_empty_campaign_id(self):
        spec = _make_valid_spec(campaign_id="")
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("campaign_id" in e for e in result.errors)

    def test_whitespace_campaign_id(self):
        spec = _make_valid_spec(campaign_id="   ")
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("campaign_id" in e for e in result.errors)

    def test_multiple_missing_fields(self):
        spec = _make_valid_spec(campaign_id="", parameters=[], objectives=[])
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert len(result.errors) == 3


# ── TestParameterValidation ──────────────────────────────


class TestParameterValidation:
    def test_bad_bounds_lower_greater_than_upper(self):
        spec = _make_valid_spec(
            parameters=[_make_continuous_param(name="bad", lower=10.0, upper=1.0)]
        )
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("lower bound" in e and "upper bound" in e for e in result.errors)

    def test_bad_bounds_equal(self):
        spec = _make_valid_spec(
            parameters=[_make_continuous_param(name="eq", lower=5.0, upper=5.0)]
        )
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("strictly less than" in e for e in result.errors)

    def test_missing_bounds_for_continuous(self):
        p = ParameterDef(name="no_bounds", type=ParamType.CONTINUOUS)
        spec = _make_valid_spec(parameters=[p])
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("lower and upper bounds are required" in e for e in result.errors)

    def test_missing_lower_bound_for_discrete(self):
        p = ParameterDef(name="half_bounds", type=ParamType.DISCRETE, upper=10.0)
        spec = _make_valid_spec(parameters=[p])
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("bounds are required" in e for e in result.errors)

    def test_missing_categories_for_categorical(self):
        p = ParameterDef(name="no_cats", type=ParamType.CATEGORICAL, categories=None)
        spec = _make_valid_spec(parameters=[p])
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("categories" in e for e in result.errors)

    def test_empty_categories_for_categorical(self):
        p = ParameterDef(name="empty_cats", type=ParamType.CATEGORICAL, categories=[])
        spec = _make_valid_spec(parameters=[p])
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("categories" in e for e in result.errors)

    def test_duplicate_parameter_names(self):
        spec = _make_valid_spec(
            parameters=[
                _make_continuous_param(name="dup"),
                _make_continuous_param(name="dup", lower=-1.0, upper=1.0),
            ]
        )
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("Duplicate parameter name" in e and "'dup'" in e for e in result.errors)


# ── TestObjectiveValidation ──────────────────────────────


class TestObjectiveValidation:
    def test_duplicate_objective_names(self):
        spec = _make_valid_spec(
            objectives=[
                _make_objective(name="loss"),
                _make_objective(name="loss", direction=Direction.MAXIMIZE),
            ]
        )
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("Duplicate objective name" in e and "'loss'" in e for e in result.errors)


# ── TestConditionalValidation ────────────────────────────


class TestConditionalValidation:
    def test_parent_not_found(self):
        child = _make_continuous_param(
            name="child",
            condition=ConditionDef(parent_name="nonexistent", parent_value="x"),
        )
        spec = _make_valid_spec(
            parameters=[_make_continuous_param(name="x1"), child]
        )
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("does not reference an existing parameter" in e for e in result.errors)

    def test_invalid_parent_value_categorical(self):
        parent = _make_categorical_param(name="algo", categories=["cma", "tpe"])
        child = _make_continuous_param(
            name="sigma",
            condition=ConditionDef(parent_name="algo", parent_value="invalid_algo"),
        )
        spec = _make_valid_spec(parameters=[parent, child])
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("not in parent" in e and "categories" in e for e in result.errors)

    def test_valid_conditional_categorical(self):
        parent = _make_categorical_param(name="algo", categories=["cma", "tpe"])
        child = _make_continuous_param(
            name="sigma",
            condition=ConditionDef(parent_name="algo", parent_value="cma"),
        )
        spec = _make_valid_spec(parameters=[parent, child])
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is True

    def test_invalid_parent_value_numeric_out_of_bounds(self):
        parent = _make_continuous_param(name="temp", lower=100.0, upper=500.0)
        child = _make_continuous_param(
            name="pressure",
            condition=ConditionDef(parent_name="temp", parent_value=600.0),
        )
        spec = _make_valid_spec(parameters=[parent, child])
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("outside parent" in e and "bounds" in e for e in result.errors)

    def test_invalid_parent_value_non_numeric_for_continuous(self):
        parent = _make_continuous_param(name="temp", lower=100.0, upper=500.0)
        child = _make_continuous_param(
            name="pressure",
            condition=ConditionDef(parent_name="temp", parent_value="not_a_number"),
        )
        spec = _make_valid_spec(parameters=[parent, child])
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("not numeric" in e for e in result.errors)

    def test_valid_conditional_numeric(self):
        parent = _make_continuous_param(name="temp", lower=100.0, upper=500.0)
        child = _make_continuous_param(
            name="pressure",
            condition=ConditionDef(parent_name="temp", parent_value=250.0),
        )
        spec = _make_valid_spec(parameters=[parent, child])
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is True


# ── TestFrozenValidation ─────────────────────────────────


class TestFrozenValidation:
    def test_frozen_value_out_of_bounds_continuous(self):
        p = _make_continuous_param(
            name="x", lower=0.0, upper=1.0, frozen=True, frozen_value=2.0
        )
        spec = _make_valid_spec(parameters=[p, _make_continuous_param(name="y")])
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("frozen_value" in e and "outside bounds" in e for e in result.errors)

    def test_frozen_value_below_lower_bound(self):
        p = _make_continuous_param(
            name="x", lower=0.0, upper=1.0, frozen=True, frozen_value=-0.5
        )
        spec = _make_valid_spec(parameters=[p, _make_continuous_param(name="y")])
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("frozen_value" in e for e in result.errors)

    def test_frozen_value_not_in_categories(self):
        p = _make_categorical_param(
            name="method",
            categories=["adam", "sgd"],
            frozen=True,
            frozen_value="rmsprop",
        )
        spec = _make_valid_spec(parameters=[p, _make_continuous_param(name="x")])
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("frozen_value" in e and "not in categories" in e for e in result.errors)

    def test_frozen_value_valid_continuous(self):
        p = _make_continuous_param(
            name="x", lower=0.0, upper=1.0, frozen=True, frozen_value=0.5
        )
        spec = _make_valid_spec(parameters=[p, _make_continuous_param(name="y")])
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is True

    def test_frozen_value_valid_categorical(self):
        p = _make_categorical_param(
            name="method",
            categories=["adam", "sgd"],
            frozen=True,
            frozen_value="adam",
        )
        spec = _make_valid_spec(parameters=[p, _make_continuous_param(name="x")])
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is True

    def test_frozen_without_value_skipped(self):
        """Frozen param with frozen_value=None should not trigger value validation."""
        p = _make_continuous_param(
            name="x", lower=0.0, upper=1.0, frozen=True, frozen_value=None
        )
        spec = _make_valid_spec(parameters=[p, _make_continuous_param(name="y")])
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is True


# ── TestBudgetValidation ─────────────────────────────────


class TestBudgetValidation:
    def test_negative_max_samples(self):
        spec = _make_valid_spec(budget=BudgetDef(max_samples=-10))
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("max_samples" in e and "positive" in e for e in result.errors)

    def test_zero_max_samples(self):
        spec = _make_valid_spec(budget=BudgetDef(max_samples=0))
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("max_samples" in e for e in result.errors)

    def test_negative_max_time_seconds(self):
        spec = _make_valid_spec(budget=BudgetDef(max_time_seconds=-1.0))
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("max_time_seconds" in e for e in result.errors)

    def test_negative_max_cost(self):
        spec = _make_valid_spec(budget=BudgetDef(max_cost=-100.0))
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("max_cost" in e for e in result.errors)

    def test_negative_max_iterations(self):
        spec = _make_valid_spec(budget=BudgetDef(max_iterations=-5))
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is False
        assert any("max_iterations" in e for e in result.errors)

    def test_valid_budget(self):
        spec = _make_valid_spec(
            budget=BudgetDef(max_samples=100, max_time_seconds=3600.0)
        )
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is True


# ── TestWarnings ─────────────────────────────────────────


class TestWarnings:
    def test_no_budget_warning(self):
        spec = _make_valid_spec(budget=BudgetDef())
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is True
        assert any("run indefinitely" in w for w in result.warnings)

    def test_all_frozen_warning(self):
        params = [
            _make_continuous_param(name="x1", frozen=True, frozen_value=0.5),
            _make_continuous_param(name="x2", frozen=True, frozen_value=0.3),
        ]
        spec = _make_valid_spec(parameters=params)
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is True
        assert any("All parameters are frozen" in w for w in result.warnings)

    def test_single_param_warning(self):
        spec = _make_valid_spec(parameters=[_make_continuous_param(name="solo")])
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is True
        assert any("Only one active parameter" in w for w in result.warnings)

    def test_single_active_param_among_frozen(self):
        params = [
            _make_continuous_param(name="active"),
            _make_continuous_param(name="frozen1", frozen=True, frozen_value=0.5),
        ]
        spec = _make_valid_spec(parameters=params)
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.valid is True
        assert any("Only one active parameter" in w and "'active'" in w for w in result.warnings)

    def test_no_warnings_for_normal_spec(self):
        spec = _make_valid_spec()
        validator = SpecValidator()
        result = validator.validate(spec)
        assert result.warnings == []


# ── TestValidateOrRaise ──────────────────────────────────


class TestValidateOrRaise:
    def test_raises_on_invalid(self):
        spec = _make_valid_spec(campaign_id="", parameters=[])
        validator = SpecValidator()
        with pytest.raises(DSLValidationError) as exc_info:
            validator.validate_or_raise(spec)
        assert len(exc_info.value.errors) >= 2
        assert "campaign_id" in exc_info.value.errors[0]

    def test_raises_with_correct_error_count(self):
        spec = _make_valid_spec(campaign_id="", parameters=[], objectives=[])
        validator = SpecValidator()
        with pytest.raises(DSLValidationError) as exc_info:
            validator.validate_or_raise(spec)
        assert len(exc_info.value.errors) == 3

    def test_does_not_raise_on_valid(self):
        spec = _make_valid_spec()
        validator = SpecValidator()
        # Should not raise.
        validator.validate_or_raise(spec)

    def test_error_message_formatting(self):
        spec = _make_valid_spec(campaign_id="")
        validator = SpecValidator()
        with pytest.raises(DSLValidationError) as exc_info:
            validator.validate_or_raise(spec)
        error_str = str(exc_info.value)
        assert "validation failed" in error_str
        assert "error(s)" in error_str


# ── TestValidationResult ─────────────────────────────────


class TestValidationResult:
    def test_creation(self):
        r = ValidationResult(valid=True)
        assert r.valid is True
        assert r.errors == []
        assert r.warnings == []

    def test_with_errors(self):
        r = ValidationResult(valid=False, errors=["err1", "err2"])
        assert r.valid is False
        assert len(r.errors) == 2

    def test_with_warnings(self):
        r = ValidationResult(valid=True, warnings=["warn1"])
        assert r.valid is True
        assert len(r.warnings) == 1
