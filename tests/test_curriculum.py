"""Tests for the Auto Curriculum Optimization (Capability 17).

Covers:
- Module helpers: _rank_parameters, _widen_bounds, _interpolate_bounds
- CurriculumStage dataclass: construction, n_active_parameters, to_dict/from_dict
- CurriculumPolicy dataclass: defaults, custom values, to_dict/from_dict
- CurriculumPlan dataclass: properties (current_stage, is_final_stage, is_complete, n_stages), to_dict/from_dict
- CurriculumEngine: create_plan (single param, multi-param, importance scores, constraints),
  evaluate (promotion, demotion, insufficient data, immutability), get_active_snapshot
"""

from __future__ import annotations

import copy
import math

import pytest

from optimization_copilot.curriculum.engine import (
    CurriculumEngine,
    CurriculumPlan,
    CurriculumPolicy,
    CurriculumStage,
    _interpolate_bounds,
    _rank_parameters,
    _widen_bounds,
)
from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.diagnostics.engine import DiagnosticsVector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(
    name: str,
    lower: float | None,
    upper: float | None,
    var_type: VariableType = VariableType.CONTINUOUS,
) -> ParameterSpec:
    """Shorthand for creating a ParameterSpec."""
    return ParameterSpec(name=name, type=var_type, lower=lower, upper=upper)


def _make_snapshot(
    n_params: int = 5,
    n_constraints: int = 2,
    n_obs: int = 20,
) -> CampaignSnapshot:
    """Create a CampaignSnapshot with *n_params* continuous parameters.

    Parameter ``pi`` has bounds ``(0, 10*(i+1))``.  Observations have
    deterministic parameter values and KPI values for reproducibility.
    """
    specs = [
        ParameterSpec(
            name=f"p{i}",
            type=VariableType.CONTINUOUS,
            lower=0.0,
            upper=10.0 * (i + 1),
        )
        for i in range(n_params)
    ]

    observations: list[Observation] = []
    for j in range(n_obs):
        params = {f"p{i}": float(j * (i + 1)) % (10.0 * (i + 1)) for i in range(n_params)}
        observations.append(
            Observation(
                iteration=j,
                parameters=params,
                kpi_values={"kpi": float(j)},
                qc_passed=True,
                is_failure=False,
                timestamp=float(j),
            )
        )

    constraints = [{"name": f"c{i}"} for i in range(n_constraints)]

    return CampaignSnapshot(
        campaign_id="test-curriculum",
        parameter_specs=specs,
        observations=observations,
        objective_names=["kpi"],
        objective_directions=["minimize"],
        constraints=constraints,
        current_iteration=n_obs,
    )


# ===========================================================================
# Module helpers
# ===========================================================================


class TestRankParameters:
    """Tests for ``_rank_parameters``."""

    def test_rank_parameters_no_scores(self) -> None:
        """No importance_scores -> original order preserved."""
        specs = [_make_spec("b", 0, 1), _make_spec("a", 0, 1), _make_spec("c", 0, 1)]
        result = _rank_parameters(specs, None)
        assert result == ["b", "a", "c"]

    def test_rank_parameters_with_scores(self) -> None:
        """Scored params sorted descending; unscored params appended."""
        specs = [
            _make_spec("x", 0, 1),
            _make_spec("y", 0, 1),
            _make_spec("z", 0, 1),
            _make_spec("w", 0, 1),
        ]
        scores = {"x": 0.1, "z": 0.9, "y": 0.5}
        result = _rank_parameters(specs, scores)
        # z(0.9) > y(0.5) > x(0.1), then unscored w
        assert result == ["z", "y", "x", "w"]

    def test_rank_parameters_tie_breaking(self) -> None:
        """Same score -> alphabetical by name."""
        specs = [
            _make_spec("beta", 0, 1),
            _make_spec("alpha", 0, 1),
            _make_spec("gamma", 0, 1),
        ]
        scores = {"beta": 0.5, "alpha": 0.5, "gamma": 0.5}
        result = _rank_parameters(specs, scores)
        assert result == ["alpha", "beta", "gamma"]


class TestWidenBounds:
    """Tests for ``_widen_bounds``."""

    def test_widen_bounds_basic(self) -> None:
        """Widening by factor=1.5 should expand around the midpoint."""
        spec = _make_spec("p", 10.0, 20.0)
        lower, upper = _widen_bounds(spec, 1.5)
        midpoint = 15.0
        original_half_range = 5.0
        new_half = original_half_range * 1.5  # 7.5
        # Clamp: max_half = original_range = 10.0, so 7.5 < 10.0 => not clamped.
        assert lower == pytest.approx(midpoint - new_half, abs=1e-9)
        assert upper == pytest.approx(midpoint + new_half, abs=1e-9)
        # Midpoint preserved:
        assert (lower + upper) / 2.0 == pytest.approx(midpoint, abs=1e-9)

    def test_widen_bounds_none_bounds(self) -> None:
        """Spec with None bounds -> (0.0, 1.0)."""
        spec_lower_none = _make_spec("p", None, 10.0)
        assert _widen_bounds(spec_lower_none, 1.5) == (0.0, 1.0)

        spec_upper_none = _make_spec("p", 0.0, None)
        assert _widen_bounds(spec_upper_none, 1.5) == (0.0, 1.0)

        spec_both_none = _make_spec("p", None, None)
        assert _widen_bounds(spec_both_none, 1.5) == (0.0, 1.0)


class TestInterpolateBounds:
    """Tests for ``_interpolate_bounds``."""

    def test_interpolate_bounds(self) -> None:
        """t=0 -> widened, t=1 -> original, t=0.5 -> midpoint."""
        spec = _make_spec("p", 10.0, 20.0)
        widened = (5.0, 25.0)  # wider bounds

        # t=0 -> fully widened
        lower_0, upper_0 = _interpolate_bounds(spec, widened, 0.0)
        assert lower_0 == pytest.approx(5.0, abs=1e-9)
        assert upper_0 == pytest.approx(25.0, abs=1e-9)

        # t=1 -> original
        lower_1, upper_1 = _interpolate_bounds(spec, widened, 1.0)
        assert lower_1 == pytest.approx(10.0, abs=1e-9)
        assert upper_1 == pytest.approx(20.0, abs=1e-9)

        # t=0.5 -> midpoint between widened and original
        lower_half, upper_half = _interpolate_bounds(spec, widened, 0.5)
        assert lower_half == pytest.approx(7.5, abs=1e-9)
        assert upper_half == pytest.approx(22.5, abs=1e-9)

    def test_interpolate_bounds_none_returns_widened(self) -> None:
        """If spec has None bounds, returns *widened* unchanged."""
        spec = _make_spec("p", None, 10.0)
        widened = (0.0, 1.0)
        result = _interpolate_bounds(spec, widened, 0.5)
        assert result == widened


# ===========================================================================
# CurriculumStage
# ===========================================================================


class TestCurriculumStage:
    """Tests for the CurriculumStage dataclass."""

    def test_stage_construction_and_properties(self) -> None:
        stage = CurriculumStage(
            stage_id=0,
            active_parameters=["a", "b", "c"],
            modified_bounds={"a": (0.0, 10.0), "b": (1.0, 5.0), "c": (2.0, 8.0)},
            constraints_enabled=["c1"],
            difficulty_level=0.5,
            metadata={"note": "first"},
        )
        assert stage.stage_id == 0
        assert stage.n_active_parameters == 3
        assert stage.difficulty_level == 0.5
        assert stage.constraints_enabled == ["c1"]
        assert stage.metadata == {"note": "first"}

    def test_stage_to_dict(self) -> None:
        stage = CurriculumStage(
            stage_id=1,
            active_parameters=["x"],
            modified_bounds={"x": (0.0, 1.0)},
            constraints_enabled=[],
            difficulty_level=0.25,
        )
        d = stage.to_dict()
        assert d["stage_id"] == 1
        assert d["active_parameters"] == ["x"]
        # modified_bounds values should be lists in dict form
        assert d["modified_bounds"]["x"] == [0.0, 1.0]
        assert d["constraints_enabled"] == []
        assert d["difficulty_level"] == 0.25
        assert d["metadata"] == {}

    def test_stage_from_dict_roundtrip(self) -> None:
        original = CurriculumStage(
            stage_id=2,
            active_parameters=["a", "b"],
            modified_bounds={"a": (0.0, 5.0), "b": (1.0, 9.0)},
            constraints_enabled=["c0"],
            difficulty_level=0.75,
            metadata={"round": 1},
        )
        d = original.to_dict()
        restored = CurriculumStage.from_dict(d)
        assert restored.stage_id == original.stage_id
        assert restored.active_parameters == original.active_parameters
        assert restored.modified_bounds == original.modified_bounds
        assert restored.constraints_enabled == original.constraints_enabled
        assert restored.difficulty_level == original.difficulty_level
        assert restored.metadata == original.metadata


# ===========================================================================
# CurriculumPolicy
# ===========================================================================


class TestCurriculumPolicy:
    """Tests for the CurriculumPolicy dataclass."""

    def test_policy_defaults(self) -> None:
        policy = CurriculumPolicy()
        assert policy.min_observations_per_stage == 10
        assert policy.promotion_plateau_threshold == 3
        assert policy.promotion_convergence_threshold == pytest.approx(0.2)
        assert policy.promotion_velocity_threshold == pytest.approx(0.0)
        assert policy.demotion_plateau_threshold == 8
        assert policy.demotion_failure_rate_threshold == pytest.approx(0.4)
        assert policy.initial_parameter_fraction == pytest.approx(0.3)
        assert policy.parameter_increment_fraction == pytest.approx(0.2)
        assert policy.bounds_widening_factor == pytest.approx(1.5)
        assert policy.bounds_tightening_per_stage == pytest.approx(0.8)
        assert policy.constraint_introduction_stage == 2

    def test_policy_custom_values(self) -> None:
        policy = CurriculumPolicy(
            min_observations_per_stage=5,
            promotion_plateau_threshold=2,
            demotion_failure_rate_threshold=0.3,
            initial_parameter_fraction=0.5,
            constraint_introduction_stage=1,
        )
        assert policy.min_observations_per_stage == 5
        assert policy.promotion_plateau_threshold == 2
        assert policy.demotion_failure_rate_threshold == pytest.approx(0.3)
        assert policy.initial_parameter_fraction == pytest.approx(0.5)
        assert policy.constraint_introduction_stage == 1

    def test_policy_to_from_dict_roundtrip(self) -> None:
        original = CurriculumPolicy(
            min_observations_per_stage=7,
            bounds_widening_factor=2.0,
            constraint_introduction_stage=3,
        )
        d = original.to_dict()
        restored = CurriculumPolicy.from_dict(d)
        assert restored.min_observations_per_stage == original.min_observations_per_stage
        assert restored.bounds_widening_factor == pytest.approx(original.bounds_widening_factor)
        assert restored.constraint_introduction_stage == original.constraint_introduction_stage
        # All fields should match
        assert restored.to_dict() == original.to_dict()


# ===========================================================================
# CurriculumPlan
# ===========================================================================


class TestCurriculumPlan:
    """Tests for the CurriculumPlan dataclass."""

    @staticmethod
    def _make_plan(n_stages: int = 3, current_idx: int = 0) -> CurriculumPlan:
        """Build a simple plan with *n_stages* stages."""
        stages = []
        all_params = ["p0", "p1", "p2", "p3", "p4"]
        all_constraints = ["c0", "c1"]
        for i in range(n_stages):
            # Progressive parameter inclusion
            n_active = min(len(all_params), 1 + i * 2)
            active = all_params[:n_active]
            stages.append(
                CurriculumStage(
                    stage_id=i,
                    active_parameters=active,
                    modified_bounds={p: (0.0, 10.0) for p in active},
                    constraints_enabled=all_constraints[:i],
                    difficulty_level=round(i / max(1, n_stages - 1), 4),
                )
            )
        return CurriculumPlan(
            stages=stages,
            current_stage_index=current_idx,
            total_parameters=len(all_params),
            total_constraints=len(all_constraints),
        )

    def test_plan_current_stage(self) -> None:
        plan = self._make_plan(n_stages=3, current_idx=1)
        assert plan.current_stage.stage_id == 1

    def test_plan_is_final_stage(self) -> None:
        plan_not_final = self._make_plan(n_stages=3, current_idx=1)
        assert plan_not_final.is_final_stage is False

        plan_final = self._make_plan(n_stages=3, current_idx=2)
        assert plan_final.is_final_stage is True

    def test_plan_is_complete(self) -> None:
        # Build a plan where the final stage has all params and all constraints
        stages = [
            CurriculumStage(
                stage_id=0,
                active_parameters=["p0", "p1", "p2"],
                modified_bounds={"p0": (0, 10), "p1": (0, 20), "p2": (0, 30)},
                constraints_enabled=["c0", "c1"],
                difficulty_level=1.0,
            ),
        ]
        plan_complete = CurriculumPlan(
            stages=stages,
            current_stage_index=0,
            total_parameters=3,
            total_constraints=2,
        )
        assert plan_complete.is_complete is True

        # Incomplete: fewer params than total
        plan_incomplete = CurriculumPlan(
            stages=stages,
            current_stage_index=0,
            total_parameters=5,  # more than stage has
            total_constraints=2,
        )
        assert plan_incomplete.is_complete is False

    def test_plan_to_from_dict_roundtrip(self) -> None:
        plan = self._make_plan(n_stages=4, current_idx=2)
        plan.promotion_history.append({"from_stage": 0, "to_stage": 1, "reason": "good"})
        plan.metadata["test"] = True

        d = plan.to_dict()
        restored = CurriculumPlan.from_dict(d)

        assert restored.current_stage_index == plan.current_stage_index
        assert restored.total_parameters == plan.total_parameters
        assert restored.total_constraints == plan.total_constraints
        assert restored.n_stages == plan.n_stages
        assert len(restored.promotion_history) == 1
        assert restored.metadata == {"test": True}
        # Verify stages survived roundtrip
        for orig_stage, rest_stage in zip(plan.stages, restored.stages):
            assert rest_stage.stage_id == orig_stage.stage_id
            assert rest_stage.active_parameters == orig_stage.active_parameters
            assert rest_stage.modified_bounds == orig_stage.modified_bounds


# ===========================================================================
# CurriculumEngine.create_plan
# ===========================================================================


class TestCurriculumEngineCreatePlan:
    """Tests for ``CurriculumEngine.create_plan``."""

    def test_create_plan_single_param(self) -> None:
        """1 parameter -> single stage plan with difficulty=1.0."""
        snapshot = _make_snapshot(n_params=1, n_constraints=0, n_obs=5)
        engine = CurriculumEngine()
        plan = engine.create_plan(snapshot, seed=42)

        assert plan.n_stages == 1
        assert plan.current_stage_index == 0
        assert plan.total_parameters == 1
        assert plan.current_stage.active_parameters == ["p0"]
        assert plan.current_stage.difficulty_level == 1.0

    def test_create_plan_multi_param_stages(self) -> None:
        """5+ params -> multiple stages with progressive parameter inclusion."""
        snapshot = _make_snapshot(n_params=10, n_constraints=2, n_obs=20)
        engine = CurriculumEngine()
        plan = engine.create_plan(snapshot, seed=42)

        assert plan.n_stages > 1
        assert plan.total_parameters == 10

        # Verify progressive parameter counts
        prev_count = 0
        for stage in plan.stages:
            assert stage.n_active_parameters >= prev_count
            prev_count = stage.n_active_parameters

        # Final stage should have all parameters
        assert plan.stages[-1].n_active_parameters == 10

    def test_create_plan_importance_scores(self) -> None:
        """Important params appear in earlier stages."""
        snapshot = _make_snapshot(n_params=5, n_constraints=0, n_obs=10)
        # Mark p4 as most important -- it should appear in the first stage
        scores = {"p4": 1.0, "p3": 0.8, "p2": 0.5}
        engine = CurriculumEngine()
        plan = engine.create_plan(snapshot, importance_scores=scores, seed=42)

        first_stage = plan.stages[0]
        # p4 (highest score) should be in the first stage
        assert "p4" in first_stage.active_parameters
        # p3 should be before p0 (unscored) in the ordering
        ranked = plan.stages[-1].active_parameters
        assert ranked.index("p4") < ranked.index("p0")
        assert ranked.index("p3") < ranked.index("p0")

    def test_create_plan_bounds_widened_first_stage(self) -> None:
        """First stage has wider bounds than last stage."""
        snapshot = _make_snapshot(n_params=5, n_constraints=0, n_obs=10)
        engine = CurriculumEngine()
        plan = engine.create_plan(snapshot, seed=42)

        assert plan.n_stages > 1

        first_stage = plan.stages[0]
        last_stage = plan.stages[-1]

        # Check a parameter that appears in both stages.
        # First stage active params are a subset of last stage params.
        common_param = first_stage.active_parameters[0]
        first_lower, first_upper = first_stage.modified_bounds[common_param]
        last_lower, last_upper = last_stage.modified_bounds[common_param]

        first_range = first_upper - first_lower
        last_range = last_upper - last_lower

        # First stage should have wider (or equal) range compared to last stage
        assert first_range >= last_range - 1e-9

    def test_create_plan_constraints_introduction(self) -> None:
        """Constraints only appear from constraint_introduction_stage onward."""
        snapshot = _make_snapshot(n_params=10, n_constraints=3, n_obs=20)
        policy = CurriculumPolicy(constraint_introduction_stage=2)
        engine = CurriculumEngine(policy=policy)
        plan = engine.create_plan(snapshot, seed=42)

        assert plan.n_stages > 2  # Enough stages for constraint introduction

        # Stages before introduction should have no constraints
        for i in range(min(2, plan.n_stages)):
            assert plan.stages[i].constraints_enabled == [], (
                f"Stage {i} should have no constraints"
            )

        # At least the final stage should have constraints (if there are enough stages)
        if plan.n_stages > 2:
            assert len(plan.stages[-1].constraints_enabled) > 0

    def test_create_plan_deterministic(self) -> None:
        """Same inputs produce identical plans."""
        snapshot = _make_snapshot(n_params=5, n_constraints=2, n_obs=20)
        engine = CurriculumEngine()
        plan_a = engine.create_plan(snapshot, seed=42)
        plan_b = engine.create_plan(snapshot, seed=42)

        assert plan_a.to_dict() == plan_b.to_dict()


# ===========================================================================
# CurriculumEngine.evaluate
# ===========================================================================


class TestCurriculumEngineEvaluate:
    """Tests for ``CurriculumEngine.evaluate``."""

    @staticmethod
    def _build_plan_and_snapshot(
        n_params: int = 5,
        n_constraints: int = 2,
        n_obs: int = 20,
    ) -> tuple[CurriculumPlan, CampaignSnapshot]:
        """Create a plan from a snapshot and return both."""
        snapshot = _make_snapshot(n_params=n_params, n_constraints=n_constraints, n_obs=n_obs)
        engine = CurriculumEngine()
        plan = engine.create_plan(snapshot, seed=42)
        return plan, snapshot

    def test_evaluate_insufficient_data(self) -> None:
        """n_obs < min_observations -> no change in stage index."""
        plan, _ = self._build_plan_and_snapshot(n_obs=20)
        # Create snapshot with fewer observations than the threshold
        small_snapshot = _make_snapshot(n_params=5, n_constraints=2, n_obs=5)
        policy = CurriculumPolicy(min_observations_per_stage=10)
        engine = CurriculumEngine(policy=policy)

        diagnostics = DiagnosticsVector(
            kpi_plateau_length=1,
            convergence_trend=0.5,
            improvement_velocity=0.1,
        )
        updated = engine.evaluate(plan, small_snapshot, diagnostics)
        assert updated.current_stage_index == plan.current_stage_index

    def test_evaluate_promotion(self) -> None:
        """Good signals -> stage advances by 1."""
        plan, snapshot = self._build_plan_and_snapshot(n_obs=20)
        assert plan.n_stages > 1, "Need multi-stage plan for promotion test"
        assert plan.current_stage_index == 0

        engine = CurriculumEngine()
        diagnostics = DiagnosticsVector(
            kpi_plateau_length=1,  # < promotion_plateau_threshold (3)
            convergence_trend=0.5,  # > promotion_convergence_threshold (0.2)
            improvement_velocity=0.1,  # > promotion_velocity_threshold (0.0)
            failure_rate=0.0,
        )
        updated = engine.evaluate(plan, snapshot, diagnostics)
        assert updated.current_stage_index == 1
        assert len(updated.promotion_history) == 1
        assert updated.promotion_history[0]["from_stage"] == 0
        assert updated.promotion_history[0]["to_stage"] == 1

    def test_evaluate_demotion(self) -> None:
        """High failure rate -> stage drops by 1."""
        plan, snapshot = self._build_plan_and_snapshot(n_obs=20)
        assert plan.n_stages > 1

        # Move plan to stage 1 so demotion is possible
        plan_at_1 = CurriculumPlan(
            stages=plan.stages,
            current_stage_index=1,
            total_parameters=plan.total_parameters,
            total_constraints=plan.total_constraints,
        )

        engine = CurriculumEngine()
        diagnostics = DiagnosticsVector(
            failure_rate=0.5,  # > demotion_failure_rate_threshold (0.4)
            kpi_plateau_length=0,
            convergence_trend=0.5,
            improvement_velocity=0.1,
        )
        updated = engine.evaluate(plan_at_1, snapshot, diagnostics)
        assert updated.current_stage_index == 0
        assert len(updated.demotion_history) == 1
        assert updated.demotion_history[0]["from_stage"] == 1
        assert updated.demotion_history[0]["to_stage"] == 0

    def test_evaluate_demotion_priority(self) -> None:
        """Demotion checked before promotion (even if promotion criteria also met)."""
        plan, snapshot = self._build_plan_and_snapshot(n_params=10, n_obs=20)
        assert plan.n_stages > 2, "Need 3+ stage plan for this test"

        # Put plan at stage 1 (not first, not last)
        plan_at_1 = CurriculumPlan(
            stages=plan.stages,
            current_stage_index=1,
            total_parameters=plan.total_parameters,
            total_constraints=plan.total_constraints,
        )

        engine = CurriculumEngine()
        # Set both demotion conditions (failure_rate > 0.4) AND
        # promotion conditions (plateau < 3, convergence > 0.2, velocity > 0.0)
        diagnostics = DiagnosticsVector(
            failure_rate=0.5,  # triggers demotion
            kpi_plateau_length=1,  # would trigger promotion
            convergence_trend=0.5,  # would trigger promotion
            improvement_velocity=0.1,  # would trigger promotion
        )
        updated = engine.evaluate(plan_at_1, snapshot, diagnostics)
        # Demotion should win
        assert updated.current_stage_index == 0
        assert len(updated.demotion_history) == 1
        assert len(updated.promotion_history) == 0

    def test_evaluate_immutability(self) -> None:
        """Original plan unchanged after evaluate()."""
        plan, snapshot = self._build_plan_and_snapshot(n_obs=20)
        original_index = plan.current_stage_index
        original_dict = plan.to_dict()

        engine = CurriculumEngine()
        diagnostics = DiagnosticsVector(
            kpi_plateau_length=1,
            convergence_trend=0.5,
            improvement_velocity=0.1,
        )
        updated = engine.evaluate(plan, snapshot, diagnostics)

        # The updated plan should be different (promoted)
        assert updated.current_stage_index != original_index or updated is not plan
        # The original plan must be unchanged
        assert plan.current_stage_index == original_index
        assert plan.to_dict() == original_dict

    def test_evaluate_at_final_stage(self) -> None:
        """Already at final stage -> no promotion."""
        plan, snapshot = self._build_plan_and_snapshot(n_obs=20)
        # Move to final stage
        final_idx = plan.n_stages - 1
        plan_at_final = CurriculumPlan(
            stages=plan.stages,
            current_stage_index=final_idx,
            total_parameters=plan.total_parameters,
            total_constraints=plan.total_constraints,
        )

        engine = CurriculumEngine()
        # Good diagnostics that would normally promote
        diagnostics = DiagnosticsVector(
            kpi_plateau_length=1,
            convergence_trend=0.5,
            improvement_velocity=0.1,
            failure_rate=0.0,
        )
        updated = engine.evaluate(plan_at_final, snapshot, diagnostics)
        assert updated.current_stage_index == final_idx
        assert len(updated.promotion_history) == 0


# ===========================================================================
# CurriculumEngine.get_active_snapshot
# ===========================================================================


class TestCurriculumEngineGetActiveSnapshot:
    """Tests for ``CurriculumEngine.get_active_snapshot``."""

    @staticmethod
    def _build_plan_at_stage(
        stage_idx: int = 0,
        n_params: int = 5,
        n_constraints: int = 2,
        n_obs: int = 20,
    ) -> tuple[CurriculumPlan, CampaignSnapshot]:
        """Create a plan and position it at *stage_idx*."""
        snapshot = _make_snapshot(n_params=n_params, n_constraints=n_constraints, n_obs=n_obs)
        engine = CurriculumEngine()
        plan = engine.create_plan(snapshot, seed=42)
        # Move to requested stage
        plan_at = CurriculumPlan(
            stages=plan.stages,
            current_stage_index=min(stage_idx, plan.n_stages - 1),
            total_parameters=plan.total_parameters,
            total_constraints=plan.total_constraints,
        )
        return plan_at, snapshot

    def test_active_snapshot_filters_parameters(self) -> None:
        """Only active params in output."""
        plan, snapshot = self._build_plan_at_stage(stage_idx=0, n_params=5)
        active_snapshot = CurriculumEngine.get_active_snapshot(plan, snapshot)

        stage = plan.current_stage
        active_set = set(stage.active_parameters)
        output_param_names = {s.name for s in active_snapshot.parameter_specs}

        assert output_param_names == active_set
        # First stage should have fewer params than original
        if plan.n_stages > 1:
            assert len(active_snapshot.parameter_specs) < len(snapshot.parameter_specs)

    def test_active_snapshot_modifies_bounds(self) -> None:
        """Bounds match stage's modified_bounds."""
        plan, snapshot = self._build_plan_at_stage(stage_idx=0, n_params=5)
        active_snapshot = CurriculumEngine.get_active_snapshot(plan, snapshot)

        stage = plan.current_stage
        for spec in active_snapshot.parameter_specs:
            if spec.name in stage.modified_bounds:
                expected_lower, expected_upper = stage.modified_bounds[spec.name]
                assert spec.lower == pytest.approx(expected_lower, abs=1e-9)
                assert spec.upper == pytest.approx(expected_upper, abs=1e-9)

    def test_active_snapshot_filters_constraints(self) -> None:
        """Only enabled constraints in output."""
        # Use a later stage that has constraints enabled
        plan, snapshot = self._build_plan_at_stage(
            stage_idx=0, n_params=10, n_constraints=3
        )
        active_snapshot = CurriculumEngine.get_active_snapshot(plan, snapshot)

        stage = plan.current_stage
        enabled_set = set(stage.constraints_enabled)

        for c in active_snapshot.constraints:
            assert c["name"] in enabled_set

        assert len(active_snapshot.constraints) == len(stage.constraints_enabled)

    def test_active_snapshot_preserves_observations(self) -> None:
        """All observations kept regardless of stage."""
        plan, snapshot = self._build_plan_at_stage(stage_idx=0, n_params=5, n_obs=25)
        active_snapshot = CurriculumEngine.get_active_snapshot(plan, snapshot)

        assert len(active_snapshot.observations) == len(snapshot.observations)
        # Verify observation content is unchanged
        for orig, filtered in zip(snapshot.observations, active_snapshot.observations):
            assert orig.iteration == filtered.iteration
            assert orig.kpi_values == filtered.kpi_values
            assert orig.parameters == filtered.parameters
