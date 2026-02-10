"""Tests for the DSL-to-core bridge."""

import pytest

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ObjectiveForm,
    ParameterSpec,
    ProblemFingerprint,
    RiskPosture,
    VariableType,
)
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
from optimization_copilot.dsl.bridge import SpecBridge


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


def _make_spec(**kwargs) -> OptimizationSpec:
    defaults = {
        "campaign_id": "bridge-test",
        "parameters": [
            _make_continuous_param(name="x1"),
            _make_continuous_param(name="x2", lower=-1.0, upper=1.0),
        ],
        "objectives": [_make_objective()],
        "budget": BudgetDef(max_samples=100),
    }
    defaults.update(kwargs)
    return OptimizationSpec(**defaults)


def _make_observation(iteration: int = 0, **kwargs) -> Observation:
    defaults = {
        "iteration": iteration,
        "parameters": {"x1": 0.5, "x2": 0.3},
        "kpi_values": {"loss": 1.0},
    }
    defaults.update(kwargs)
    return Observation(**defaults)


# ── TestToParameterSpecs ─────────────────────────────────


class TestToParameterSpecs:
    def test_correct_mapping_continuous(self):
        spec = _make_spec(parameters=[_make_continuous_param(name="temp", lower=100.0, upper=500.0)])
        result = SpecBridge.to_parameter_specs(spec)

        assert len(result) == 1
        assert isinstance(result[0], ParameterSpec)
        assert result[0].name == "temp"
        assert result[0].type == VariableType.CONTINUOUS
        assert result[0].lower == 100.0
        assert result[0].upper == 500.0
        assert result[0].categories is None

    def test_correct_mapping_discrete(self):
        spec = _make_spec(parameters=[_make_discrete_param(name="layers", lower=1.0, upper=10.0)])
        result = SpecBridge.to_parameter_specs(spec)

        assert len(result) == 1
        assert result[0].type == VariableType.DISCRETE
        assert result[0].lower == 1.0
        assert result[0].upper == 10.0

    def test_correct_mapping_categorical(self):
        spec = _make_spec(
            parameters=[_make_categorical_param(name="opt", categories=["a", "b", "c"])]
        )
        result = SpecBridge.to_parameter_specs(spec)

        assert len(result) == 1
        assert result[0].type == VariableType.CATEGORICAL
        assert result[0].categories == ["a", "b", "c"]

    def test_frozen_params_excluded(self):
        params = [
            _make_continuous_param(name="active1"),
            _make_continuous_param(name="frozen1", frozen=True, frozen_value=0.5),
            _make_continuous_param(name="active2", lower=-1.0, upper=1.0),
        ]
        spec = _make_spec(parameters=params)
        result = SpecBridge.to_parameter_specs(spec)

        assert len(result) == 2
        names = [p.name for p in result]
        assert "active1" in names
        assert "active2" in names
        assert "frozen1" not in names

    def test_all_frozen_returns_empty(self):
        params = [
            _make_continuous_param(name="f1", frozen=True, frozen_value=0.1),
            _make_continuous_param(name="f2", frozen=True, frozen_value=0.2),
        ]
        spec = _make_spec(parameters=params)
        result = SpecBridge.to_parameter_specs(spec)
        assert result == []

    def test_type_mapping_comprehensive(self):
        params = [
            _make_continuous_param(name="cont"),
            _make_discrete_param(name="disc"),
            _make_categorical_param(name="cat"),
        ]
        spec = _make_spec(parameters=params)
        result = SpecBridge.to_parameter_specs(spec)

        type_map = {p.name: p.type for p in result}
        assert type_map["cont"] == VariableType.CONTINUOUS
        assert type_map["disc"] == VariableType.DISCRETE
        assert type_map["cat"] == VariableType.CATEGORICAL

    def test_multiple_params_preserved_order(self):
        params = [
            _make_continuous_param(name="z"),
            _make_continuous_param(name="a", lower=-1.0, upper=1.0),
            _make_continuous_param(name="m", lower=0.0, upper=10.0),
        ]
        spec = _make_spec(parameters=params)
        result = SpecBridge.to_parameter_specs(spec)

        assert [p.name for p in result] == ["z", "a", "m"]


# ── TestToCampaignSnapshot ───────────────────────────────


class TestToCampaignSnapshot:
    def test_basic_snapshot(self):
        spec = _make_spec()
        snapshot = SpecBridge.to_campaign_snapshot(spec)

        assert isinstance(snapshot, CampaignSnapshot)
        assert snapshot.campaign_id == "bridge-test"
        assert len(snapshot.parameter_specs) == 2
        assert snapshot.observations == []
        assert snapshot.current_iteration == 0

    def test_objectives_mapped(self):
        spec = _make_spec(
            objectives=[
                _make_objective(name="loss", direction=Direction.MINIMIZE),
                _make_objective(name="throughput", direction=Direction.MAXIMIZE),
            ]
        )
        snapshot = SpecBridge.to_campaign_snapshot(spec)

        assert snapshot.objective_names == ["loss", "throughput"]
        assert snapshot.objective_directions == ["minimize", "maximize"]

    def test_constraints_from_objectives(self):
        spec = _make_spec(
            objectives=[
                ObjectiveDef(
                    name="cost",
                    direction=Direction.MINIMIZE,
                    constraint_lower=0.0,
                    constraint_upper=100.0,
                ),
            ]
        )
        snapshot = SpecBridge.to_campaign_snapshot(spec)

        assert len(snapshot.constraints) == 2
        lower_constraint = next(c for c in snapshot.constraints if c["type"] == "lower_bound")
        upper_constraint = next(c for c in snapshot.constraints if c["type"] == "upper_bound")
        assert lower_constraint["target"] == "cost"
        assert lower_constraint["value"] == 0.0
        assert upper_constraint["target"] == "cost"
        assert upper_constraint["value"] == 100.0

    def test_constraints_only_lower(self):
        spec = _make_spec(
            objectives=[
                ObjectiveDef(
                    name="quality",
                    direction=Direction.MAXIMIZE,
                    constraint_lower=0.5,
                ),
            ]
        )
        snapshot = SpecBridge.to_campaign_snapshot(spec)
        assert len(snapshot.constraints) == 1
        assert snapshot.constraints[0]["type"] == "lower_bound"

    def test_no_constraints_when_none(self):
        spec = _make_spec(
            objectives=[_make_objective()]
        )
        snapshot = SpecBridge.to_campaign_snapshot(spec)
        assert snapshot.constraints == []

    def test_empty_observations_default(self):
        spec = _make_spec()
        snapshot = SpecBridge.to_campaign_snapshot(spec)
        assert snapshot.observations == []
        assert snapshot.current_iteration == 0

    def test_with_observations(self):
        spec = _make_spec()
        obs = [
            _make_observation(iteration=0),
            _make_observation(iteration=1),
            _make_observation(iteration=2),
        ]
        snapshot = SpecBridge.to_campaign_snapshot(spec, observations=obs)

        assert len(snapshot.observations) == 3
        assert snapshot.current_iteration == 2

    def test_metadata_includes_dsl_fields(self):
        spec = _make_spec(
            name="My Campaign",
            description="Test desc",
            seed=77,
            risk_preference=RiskPreference.AGGRESSIVE,
            parallel=ParallelDef(batch_size=4),
            metadata={"custom_key": "custom_value"},
        )
        snapshot = SpecBridge.to_campaign_snapshot(spec)

        assert snapshot.metadata["dsl_campaign_id"] == "bridge-test"
        assert snapshot.metadata["seed"] == 77
        assert snapshot.metadata["risk_preference"] == "aggressive"
        assert snapshot.metadata["batch_size"] == 4
        assert snapshot.metadata["name"] == "My Campaign"
        assert snapshot.metadata["description"] == "Test desc"
        assert snapshot.metadata["custom_key"] == "custom_value"

    def test_frozen_params_excluded_from_specs(self):
        params = [
            _make_continuous_param(name="active"),
            _make_continuous_param(name="frozen", frozen=True, frozen_value=0.5),
        ]
        spec = _make_spec(parameters=params)
        snapshot = SpecBridge.to_campaign_snapshot(spec)

        param_names = [p.name for p in snapshot.parameter_specs]
        assert "active" in param_names
        assert "frozen" not in param_names


# ── TestToInitialFingerprint ─────────────────────────────


class TestToInitialFingerprint:
    def test_single_continuous_type(self):
        spec = _make_spec(
            parameters=[_make_continuous_param(name="x1")]
        )
        fp = SpecBridge.to_initial_fingerprint(spec)

        assert isinstance(fp, ProblemFingerprint)
        assert fp.variable_types == VariableType.CONTINUOUS

    def test_single_discrete_type(self):
        spec = _make_spec(
            parameters=[_make_discrete_param(name="n")]
        )
        fp = SpecBridge.to_initial_fingerprint(spec)
        assert fp.variable_types == VariableType.DISCRETE

    def test_single_categorical_type(self):
        spec = _make_spec(
            parameters=[_make_categorical_param(name="algo")]
        )
        fp = SpecBridge.to_initial_fingerprint(spec)
        assert fp.variable_types == VariableType.CATEGORICAL

    def test_mixed_types(self):
        spec = _make_spec(
            parameters=[
                _make_continuous_param(name="x"),
                _make_categorical_param(name="algo"),
            ]
        )
        fp = SpecBridge.to_initial_fingerprint(spec)
        assert fp.variable_types == VariableType.MIXED

    def test_mixed_three_types(self):
        spec = _make_spec(
            parameters=[
                _make_continuous_param(name="cont"),
                _make_discrete_param(name="disc"),
                _make_categorical_param(name="cat"),
            ]
        )
        fp = SpecBridge.to_initial_fingerprint(spec)
        assert fp.variable_types == VariableType.MIXED

    def test_frozen_params_excluded_from_type_inference(self):
        spec = _make_spec(
            parameters=[
                _make_continuous_param(name="active"),
                _make_categorical_param(name="frozen_cat", frozen=True, frozen_value="adam"),
            ]
        )
        fp = SpecBridge.to_initial_fingerprint(spec)
        # Only the active continuous param contributes.
        assert fp.variable_types == VariableType.CONTINUOUS

    def test_all_frozen_defaults_to_continuous(self):
        spec = _make_spec(
            parameters=[
                _make_continuous_param(name="f1", frozen=True, frozen_value=0.5),
            ]
        )
        fp = SpecBridge.to_initial_fingerprint(spec)
        assert fp.variable_types == VariableType.CONTINUOUS

    def test_single_objective_form(self):
        spec = _make_spec(
            objectives=[_make_objective(name="loss")]
        )
        fp = SpecBridge.to_initial_fingerprint(spec)
        assert fp.objective_form == ObjectiveForm.SINGLE

    def test_multi_objective_form(self):
        spec = _make_spec(
            objectives=[
                _make_objective(name="loss"),
                _make_objective(name="throughput", direction=Direction.MAXIMIZE),
            ]
        )
        fp = SpecBridge.to_initial_fingerprint(spec)
        assert fp.objective_form == ObjectiveForm.MULTI_OBJECTIVE

    def test_constrained_objective_form(self):
        spec = _make_spec(
            objectives=[
                ObjectiveDef(
                    name="cost",
                    direction=Direction.MINIMIZE,
                    constraint_upper=100.0,
                ),
            ]
        )
        fp = SpecBridge.to_initial_fingerprint(spec)
        assert fp.objective_form == ObjectiveForm.CONSTRAINED

    def test_constrained_takes_priority_over_multi(self):
        spec = _make_spec(
            objectives=[
                ObjectiveDef(
                    name="cost",
                    direction=Direction.MINIMIZE,
                    constraint_upper=100.0,
                ),
                _make_objective(name="quality", direction=Direction.MAXIMIZE),
            ]
        )
        fp = SpecBridge.to_initial_fingerprint(spec)
        assert fp.objective_form == ObjectiveForm.CONSTRAINED

    def test_conservative_defaults(self):
        """Observation-dependent dimensions should have conservative defaults."""
        spec = _make_spec()
        fp = SpecBridge.to_initial_fingerprint(spec)

        from optimization_copilot.core.models import (
            CostProfile,
            DataScale,
            Dynamics,
            FailureInformativeness,
            FeasibleRegion,
            NoiseRegime,
        )
        assert fp.noise_regime == NoiseRegime.MEDIUM
        assert fp.cost_profile == CostProfile.UNIFORM
        assert fp.failure_informativeness == FailureInformativeness.WEAK
        assert fp.data_scale == DataScale.TINY
        assert fp.dynamics == Dynamics.STATIC
        assert fp.feasible_region == FeasibleRegion.WIDE


# ── TestToRiskPosture ────────────────────────────────────


class TestToRiskPosture:
    def test_conservative_mapping(self):
        spec = _make_spec(risk_preference=RiskPreference.CONSERVATIVE)
        assert SpecBridge.to_risk_posture(spec) == RiskPosture.CONSERVATIVE

    def test_moderate_mapping(self):
        spec = _make_spec(risk_preference=RiskPreference.MODERATE)
        assert SpecBridge.to_risk_posture(spec) == RiskPosture.MODERATE

    def test_aggressive_mapping(self):
        spec = _make_spec(risk_preference=RiskPreference.AGGRESSIVE)
        assert SpecBridge.to_risk_posture(spec) == RiskPosture.AGGRESSIVE

    def test_all_risk_preferences_mapped(self):
        """Every RiskPreference value must map to a RiskPosture."""
        for pref in RiskPreference:
            spec = _make_spec(risk_preference=pref)
            posture = SpecBridge.to_risk_posture(spec)
            assert isinstance(posture, RiskPosture)


# ── TestActiveParameters ─────────────────────────────────


class TestActiveParameters:
    def test_no_frozen_no_conditions(self):
        spec = _make_spec()
        result = SpecBridge.active_parameters(spec)
        assert len(result) == 2
        assert all(isinstance(p, ParameterDef) for p in result)

    def test_frozen_excluded(self):
        params = [
            _make_continuous_param(name="active"),
            _make_continuous_param(name="frozen", frozen=True, frozen_value=0.5),
        ]
        spec = _make_spec(parameters=params)
        result = SpecBridge.active_parameters(spec)

        assert len(result) == 1
        assert result[0].name == "active"

    def test_conditional_without_values_included(self):
        """Without current_values, conditional params are included."""
        params = [
            _make_categorical_param(name="algo", categories=["cma", "tpe"]),
            _make_continuous_param(
                name="sigma",
                condition=ConditionDef(parent_name="algo", parent_value="cma"),
            ),
        ]
        spec = _make_spec(parameters=params)
        result = SpecBridge.active_parameters(spec)

        assert len(result) == 2

    def test_conditional_with_matching_value(self):
        params = [
            _make_categorical_param(name="algo", categories=["cma", "tpe"]),
            _make_continuous_param(
                name="sigma",
                condition=ConditionDef(parent_name="algo", parent_value="cma"),
            ),
        ]
        spec = _make_spec(parameters=params)
        result = SpecBridge.active_parameters(spec, current_values={"algo": "cma"})

        assert len(result) == 2
        assert any(p.name == "sigma" for p in result)

    def test_conditional_with_non_matching_value(self):
        params = [
            _make_categorical_param(name="algo", categories=["cma", "tpe"]),
            _make_continuous_param(
                name="sigma",
                condition=ConditionDef(parent_name="algo", parent_value="cma"),
            ),
        ]
        spec = _make_spec(parameters=params)
        result = SpecBridge.active_parameters(spec, current_values={"algo": "tpe"})

        assert len(result) == 1
        assert result[0].name == "algo"

    def test_frozen_and_conditional_combined(self):
        params = [
            _make_categorical_param(name="algo", categories=["cma", "tpe"]),
            _make_continuous_param(
                name="sigma",
                condition=ConditionDef(parent_name="algo", parent_value="cma"),
            ),
            _make_continuous_param(name="fixed", frozen=True, frozen_value=0.1),
        ]
        spec = _make_spec(parameters=params)
        result = SpecBridge.active_parameters(spec, current_values={"algo": "tpe"})

        # algo is active, sigma is excluded (condition not met), fixed is excluded (frozen).
        assert len(result) == 1
        assert result[0].name == "algo"

    def test_all_frozen_returns_empty(self):
        params = [
            _make_continuous_param(name="f1", frozen=True, frozen_value=0.1),
            _make_continuous_param(name="f2", frozen=True, frozen_value=0.2),
        ]
        spec = _make_spec(parameters=params)
        result = SpecBridge.active_parameters(spec)
        assert result == []


# ── TestApplyFrozenValues ────────────────────────────────


class TestApplyFrozenValues:
    def test_frozen_values_injected(self):
        params = [
            _make_continuous_param(name="active"),
            _make_continuous_param(name="frozen_x", frozen=True, frozen_value=0.42),
            _make_categorical_param(name="frozen_m", frozen=True, frozen_value="adam"),
        ]
        spec = _make_spec(parameters=params)
        candidates = [{"active": 0.5}, {"active": 0.8}]

        result = SpecBridge.apply_frozen_values(spec, candidates)

        assert len(result) == 2
        for r in result:
            assert r["frozen_x"] == 0.42
            assert r["frozen_m"] == "adam"
        assert result[0]["active"] == 0.5
        assert result[1]["active"] == 0.8

    def test_existing_keys_preserved(self):
        """Existing keys in candidates should NOT be overwritten by frozen values."""
        params = [
            _make_continuous_param(name="overlap", frozen=True, frozen_value=999.0),
        ]
        spec = _make_spec(parameters=params)
        candidates = [{"overlap": 0.5, "other": 1.0}]

        result = SpecBridge.apply_frozen_values(spec, candidates)

        # The candidate's existing "overlap" key should take precedence.
        assert result[0]["overlap"] == 0.5
        assert result[0]["other"] == 1.0

    def test_no_frozen_params_returns_same(self):
        spec = _make_spec()
        candidates = [{"x1": 0.1, "x2": 0.2}]
        result = SpecBridge.apply_frozen_values(spec, candidates)

        assert result is candidates  # Should return the same list object.

    def test_empty_candidates_list(self):
        params = [
            _make_continuous_param(name="frozen_x", frozen=True, frozen_value=0.5),
        ]
        spec = _make_spec(parameters=params)
        result = SpecBridge.apply_frozen_values(spec, [])
        assert result == []

    def test_multiple_candidates(self):
        params = [
            _make_continuous_param(name="active"),
            _make_continuous_param(name="fixed", frozen=True, frozen_value=3.14),
        ]
        spec = _make_spec(parameters=params)
        candidates = [
            {"active": 0.1},
            {"active": 0.5},
            {"active": 0.9},
        ]

        result = SpecBridge.apply_frozen_values(spec, candidates)

        assert len(result) == 3
        for i, r in enumerate(result):
            assert r["fixed"] == 3.14
            assert r["active"] == candidates[i]["active"]

    def test_returns_new_list(self):
        """Result should be a new list, not mutating the input."""
        params = [
            _make_continuous_param(name="frozen_x", frozen=True, frozen_value=0.5),
        ]
        spec = _make_spec(parameters=params)
        original_candidates = [{"y": 1.0}]
        result = SpecBridge.apply_frozen_values(spec, original_candidates)

        assert result is not original_candidates
        # Original should not be modified.
        assert "frozen_x" not in original_candidates[0]
