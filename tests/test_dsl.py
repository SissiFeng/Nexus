"""Tests for the core DSL spec dataclasses and enums."""

import pytest

from optimization_copilot.dsl.spec import (
    ConditionDef,
    Direction,
    DiversityStrategy,
    BudgetDef,
    ObjectiveDef,
    OptimizationSpec,
    ParallelDef,
    ParameterDef,
    ParamType,
    RiskPreference,
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


def _make_spec(**kwargs) -> OptimizationSpec:
    defaults = {
        "campaign_id": "test-campaign",
        "parameters": [_make_continuous_param(), _make_categorical_param()],
        "objectives": [_make_objective()],
        "budget": BudgetDef(max_samples=100),
    }
    defaults.update(kwargs)
    return OptimizationSpec(**defaults)


# ── TestEnums ────────────────────────────────────────────


class TestEnums:
    def test_param_type_values(self):
        assert ParamType.CONTINUOUS.value == "continuous"
        assert ParamType.DISCRETE.value == "discrete"
        assert ParamType.CATEGORICAL.value == "categorical"

    def test_param_type_str_behavior(self):
        assert ParamType.CONTINUOUS == "continuous"
        assert str(ParamType.CONTINUOUS) == "ParamType.CONTINUOUS" or ParamType.CONTINUOUS.value == "continuous"

    def test_direction_values(self):
        assert Direction.MINIMIZE.value == "minimize"
        assert Direction.MAXIMIZE.value == "maximize"

    def test_risk_preference_values(self):
        assert RiskPreference.CONSERVATIVE.value == "conservative"
        assert RiskPreference.MODERATE.value == "moderate"
        assert RiskPreference.AGGRESSIVE.value == "aggressive"

    def test_diversity_strategy_values(self):
        assert DiversityStrategy.MAXIMIN.value == "maximin"
        assert DiversityStrategy.COVERAGE.value == "coverage"
        assert DiversityStrategy.HYBRID.value == "hybrid"

    def test_enum_from_value(self):
        assert ParamType("continuous") is ParamType.CONTINUOUS
        assert Direction("maximize") is Direction.MAXIMIZE
        assert RiskPreference("moderate") is RiskPreference.MODERATE
        assert DiversityStrategy("hybrid") is DiversityStrategy.HYBRID

    def test_enum_invalid_value(self):
        with pytest.raises(ValueError):
            ParamType("invalid")
        with pytest.raises(ValueError):
            Direction("invalid")


# ── TestConditionDef ─────────────────────────────────────


class TestConditionDef:
    def test_creation(self):
        cond = ConditionDef(parent_name="optimizer", parent_value="adam")
        assert cond.parent_name == "optimizer"
        assert cond.parent_value == "adam"

    def test_to_dict(self):
        cond = ConditionDef(parent_name="optimizer", parent_value="adam")
        d = cond.to_dict()
        assert d == {"parent_name": "optimizer", "parent_value": "adam"}

    def test_from_dict(self):
        d = {"parent_name": "optimizer", "parent_value": "adam"}
        cond = ConditionDef.from_dict(d)
        assert cond.parent_name == "optimizer"
        assert cond.parent_value == "adam"

    def test_round_trip(self):
        original = ConditionDef(parent_name="method", parent_value="sgd")
        restored = ConditionDef.from_dict(original.to_dict())
        assert restored.parent_name == original.parent_name
        assert restored.parent_value == original.parent_value

    def test_numeric_parent_value(self):
        cond = ConditionDef(parent_name="num_layers", parent_value=3)
        d = cond.to_dict()
        assert d["parent_value"] == 3
        restored = ConditionDef.from_dict(d)
        assert restored.parent_value == 3


# ── TestParameterDef ─────────────────────────────────────


class TestParameterDef:
    def test_continuous_creation(self):
        p = _make_continuous_param(name="learning_rate", lower=1e-5, upper=1e-1)
        assert p.name == "learning_rate"
        assert p.type == ParamType.CONTINUOUS
        assert p.lower == 1e-5
        assert p.upper == 1e-1
        assert p.categories is None
        assert p.step_size is None
        assert p.condition is None
        assert p.frozen is False
        assert p.frozen_value is None
        assert p.description == ""
        assert p.metadata == {}

    def test_discrete_creation(self):
        p = ParameterDef(
            name="num_layers",
            type=ParamType.DISCRETE,
            lower=1,
            upper=10,
            step_size=1,
        )
        assert p.type == ParamType.DISCRETE
        assert p.step_size == 1

    def test_categorical_creation(self):
        p = _make_categorical_param(name="optimizer", categories=["adam", "sgd"])
        assert p.type == ParamType.CATEGORICAL
        assert p.categories == ["adam", "sgd"]
        assert p.lower is None
        assert p.upper is None

    def test_all_fields(self):
        cond = ConditionDef(parent_name="method", parent_value="adam")
        p = ParameterDef(
            name="lr",
            type=ParamType.CONTINUOUS,
            lower=0.001,
            upper=0.1,
            categories=None,
            step_size=0.001,
            condition=cond,
            frozen=False,
            frozen_value=None,
            description="Learning rate for Adam optimizer",
            metadata={"importance": "high"},
        )
        assert p.condition.parent_name == "method"
        assert p.description == "Learning rate for Adam optimizer"
        assert p.metadata["importance"] == "high"

    def test_to_dict(self):
        p = _make_continuous_param(name="x")
        d = p.to_dict()
        assert d["name"] == "x"
        assert d["type"] == "continuous"
        assert d["lower"] == 0.0
        assert d["upper"] == 1.0
        assert d["condition"] is None

    def test_to_dict_with_condition(self):
        cond = ConditionDef(parent_name="algo", parent_value="cma")
        p = _make_continuous_param(condition=cond)
        d = p.to_dict()
        assert d["condition"] == {"parent_name": "algo", "parent_value": "cma"}

    def test_from_dict(self):
        d = {
            "name": "x",
            "type": "continuous",
            "lower": -5.0,
            "upper": 5.0,
            "categories": None,
            "step_size": None,
            "condition": None,
            "frozen": False,
            "frozen_value": None,
            "description": "",
            "metadata": {},
        }
        p = ParameterDef.from_dict(d)
        assert p.name == "x"
        assert p.type == ParamType.CONTINUOUS
        assert p.lower == -5.0

    def test_round_trip(self):
        original = _make_continuous_param(
            name="temp",
            lower=100.0,
            upper=500.0,
            description="Reaction temperature",
            metadata={"unit": "kelvin"},
        )
        restored = ParameterDef.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.type == original.type
        assert restored.lower == original.lower
        assert restored.upper == original.upper
        assert restored.description == original.description
        assert restored.metadata == original.metadata

    def test_round_trip_with_condition(self):
        cond = ConditionDef(parent_name="method", parent_value="sgd")
        original = _make_continuous_param(condition=cond)
        restored = ParameterDef.from_dict(original.to_dict())
        assert restored.condition is not None
        assert restored.condition.parent_name == "method"
        assert restored.condition.parent_value == "sgd"

    def test_frozen_param(self):
        p = _make_continuous_param(frozen=True, frozen_value=0.5)
        assert p.frozen is True
        assert p.frozen_value == 0.5

    def test_frozen_categorical_param(self):
        p = _make_categorical_param(frozen=True, frozen_value="adam")
        assert p.frozen is True
        assert p.frozen_value == "adam"


# ── TestObjectiveDef ─────────────────────────────────────


class TestObjectiveDef:
    def test_creation_minimize(self):
        obj = _make_objective(name="mse", direction=Direction.MINIMIZE)
        assert obj.name == "mse"
        assert obj.direction == Direction.MINIMIZE
        assert obj.constraint_lower is None
        assert obj.constraint_upper is None
        assert obj.is_primary is True
        assert obj.weight == 1.0

    def test_creation_maximize(self):
        obj = _make_objective(name="accuracy", direction=Direction.MAXIMIZE)
        assert obj.direction == Direction.MAXIMIZE

    def test_constraint_fields(self):
        obj = ObjectiveDef(
            name="cost",
            direction=Direction.MINIMIZE,
            constraint_lower=0.0,
            constraint_upper=100.0,
            is_primary=False,
            weight=0.5,
        )
        assert obj.constraint_lower == 0.0
        assert obj.constraint_upper == 100.0
        assert obj.is_primary is False
        assert obj.weight == 0.5

    def test_to_dict(self):
        obj = _make_objective()
        d = obj.to_dict()
        assert d["name"] == "loss"
        assert d["direction"] == "minimize"
        assert d["is_primary"] is True

    def test_from_dict(self):
        d = {
            "name": "throughput",
            "direction": "maximize",
            "constraint_lower": 10.0,
            "constraint_upper": None,
            "is_primary": True,
            "weight": 1.0,
        }
        obj = ObjectiveDef.from_dict(d)
        assert obj.name == "throughput"
        assert obj.direction == Direction.MAXIMIZE
        assert obj.constraint_lower == 10.0

    def test_round_trip(self):
        original = ObjectiveDef(
            name="quality",
            direction=Direction.MAXIMIZE,
            constraint_lower=0.5,
            constraint_upper=1.0,
            is_primary=False,
            weight=0.8,
        )
        restored = ObjectiveDef.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.direction == original.direction
        assert restored.constraint_lower == original.constraint_lower
        assert restored.constraint_upper == original.constraint_upper
        assert restored.is_primary == original.is_primary
        assert restored.weight == original.weight


# ── TestBudgetDef ────────────────────────────────────────


class TestBudgetDef:
    def test_creation_all_fields(self):
        b = BudgetDef(
            max_samples=200,
            max_time_seconds=3600.0,
            max_cost=500.0,
            max_iterations=50,
        )
        assert b.max_samples == 200
        assert b.max_time_seconds == 3600.0
        assert b.max_cost == 500.0
        assert b.max_iterations == 50

    def test_creation_defaults(self):
        b = BudgetDef()
        assert b.max_samples is None
        assert b.max_time_seconds is None
        assert b.max_cost is None
        assert b.max_iterations is None

    def test_optional_fields(self):
        b = BudgetDef(max_samples=100)
        assert b.max_samples == 100
        assert b.max_time_seconds is None

    def test_to_dict(self):
        b = BudgetDef(max_samples=50)
        d = b.to_dict()
        assert d["max_samples"] == 50
        assert d["max_time_seconds"] is None

    def test_from_dict(self):
        d = {
            "max_samples": 100,
            "max_time_seconds": None,
            "max_cost": 200.0,
            "max_iterations": None,
        }
        b = BudgetDef.from_dict(d)
        assert b.max_samples == 100
        assert b.max_cost == 200.0

    def test_round_trip(self):
        original = BudgetDef(max_samples=75, max_time_seconds=1800.0)
        restored = BudgetDef.from_dict(original.to_dict())
        assert restored.max_samples == original.max_samples
        assert restored.max_time_seconds == original.max_time_seconds
        assert restored.max_cost == original.max_cost
        assert restored.max_iterations == original.max_iterations


# ── TestParallelDef ──────────────────────────────────────


class TestParallelDef:
    def test_creation(self):
        p = ParallelDef(batch_size=4, diversity_strategy=DiversityStrategy.MAXIMIN)
        assert p.batch_size == 4
        assert p.diversity_strategy == DiversityStrategy.MAXIMIN

    def test_defaults(self):
        p = ParallelDef()
        assert p.batch_size == 1
        assert p.diversity_strategy == DiversityStrategy.HYBRID

    def test_to_dict(self):
        p = ParallelDef(batch_size=8, diversity_strategy=DiversityStrategy.COVERAGE)
        d = p.to_dict()
        assert d["batch_size"] == 8
        assert d["diversity_strategy"] == "coverage"

    def test_from_dict(self):
        d = {"batch_size": 3, "diversity_strategy": "maximin"}
        p = ParallelDef.from_dict(d)
        assert p.batch_size == 3
        assert p.diversity_strategy == DiversityStrategy.MAXIMIN

    def test_round_trip(self):
        original = ParallelDef(batch_size=5, diversity_strategy=DiversityStrategy.HYBRID)
        restored = ParallelDef.from_dict(original.to_dict())
        assert restored.batch_size == original.batch_size
        assert restored.diversity_strategy == original.diversity_strategy


# ── TestOptimizationSpec ─────────────────────────────────


class TestOptimizationSpec:
    def test_full_creation(self):
        spec = _make_spec(
            campaign_id="my-campaign",
            name="Test Campaign",
            description="A test optimization campaign",
            risk_preference=RiskPreference.AGGRESSIVE,
            parallel=ParallelDef(batch_size=4),
            metadata={"team": "research"},
            seed=123,
        )
        assert spec.campaign_id == "my-campaign"
        assert spec.name == "Test Campaign"
        assert spec.description == "A test optimization campaign"
        assert spec.risk_preference == RiskPreference.AGGRESSIVE
        assert spec.parallel.batch_size == 4
        assert spec.metadata == {"team": "research"}
        assert spec.seed == 123
        assert len(spec.parameters) == 2
        assert len(spec.objectives) == 1

    def test_defaults(self):
        spec = _make_spec()
        assert spec.risk_preference == RiskPreference.MODERATE
        assert spec.parallel.batch_size == 1
        assert spec.name == ""
        assert spec.description == ""
        assert spec.metadata == {}
        assert spec.seed == 42

    def test_seed_field(self):
        spec = _make_spec(seed=999)
        assert spec.seed == 999

    def test_to_dict(self):
        spec = _make_spec(campaign_id="dict-test", seed=7)
        d = spec.to_dict()
        assert d["campaign_id"] == "dict-test"
        assert d["risk_preference"] == "moderate"
        assert d["seed"] == 7
        assert isinstance(d["parameters"], list)
        assert isinstance(d["objectives"], list)
        assert isinstance(d["budget"], dict)
        assert isinstance(d["parallel"], dict)
        assert d["parameters"][0]["type"] == "continuous"
        assert d["objectives"][0]["direction"] == "minimize"

    def test_from_dict(self):
        d = _make_spec().to_dict()
        spec = OptimizationSpec.from_dict(d)
        assert spec.campaign_id == "test-campaign"
        assert isinstance(spec.parameters[0], ParameterDef)
        assert isinstance(spec.objectives[0], ObjectiveDef)
        assert isinstance(spec.budget, BudgetDef)
        assert isinstance(spec.parallel, ParallelDef)
        assert spec.risk_preference == RiskPreference.MODERATE

    def test_round_trip(self):
        original = _make_spec(
            campaign_id="roundtrip-test",
            name="Roundtrip",
            description="Full round-trip test",
            risk_preference=RiskPreference.CONSERVATIVE,
            parallel=ParallelDef(batch_size=3, diversity_strategy=DiversityStrategy.COVERAGE),
            metadata={"version": 2},
            seed=77,
        )
        d = original.to_dict()
        restored = OptimizationSpec.from_dict(d)

        assert restored.campaign_id == original.campaign_id
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.risk_preference == original.risk_preference
        assert restored.seed == original.seed
        assert restored.parallel.batch_size == original.parallel.batch_size
        assert restored.parallel.diversity_strategy == original.parallel.diversity_strategy
        assert restored.metadata == original.metadata
        assert len(restored.parameters) == len(original.parameters)
        assert len(restored.objectives) == len(original.objectives)

        for orig_p, rest_p in zip(original.parameters, restored.parameters):
            assert orig_p.name == rest_p.name
            assert orig_p.type == rest_p.type
            assert orig_p.lower == rest_p.lower
            assert orig_p.upper == rest_p.upper

    def test_round_trip_with_conditional_and_frozen(self):
        params = [
            _make_categorical_param(name="algo", categories=["cma", "tpe"]),
            _make_continuous_param(
                name="sigma",
                condition=ConditionDef(parent_name="algo", parent_value="cma"),
            ),
            _make_continuous_param(name="fixed_lr", frozen=True, frozen_value=0.01),
        ]
        original = _make_spec(parameters=params)
        restored = OptimizationSpec.from_dict(original.to_dict())

        assert restored.parameters[1].condition is not None
        assert restored.parameters[1].condition.parent_name == "algo"
        assert restored.parameters[1].condition.parent_value == "cma"
        assert restored.parameters[2].frozen is True
        assert restored.parameters[2].frozen_value == 0.01
