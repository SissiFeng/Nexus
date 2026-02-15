"""Tests for search-space surgery: diagnosis and application of dimension-reduction actions.

Verifies:
- SurgeryAction construction and serialization round-trip
- SurgeryReport properties, filtering, and serialization
- Range tightening on concentrated vs spread-out data
- Freeze candidate detection via importance scores
- Conditional freezing with synthetic conditional patterns
- Redundancy detection for correlated / anti-correlated / uncorrelated pairs
- Derived parameter suggestions (LOG, RATIO)
- Apply: tightening, freezing, merging modify the spec correctly
- Determinism: same inputs produce same report
- Edge cases: empty snapshot, single parameter, all categorical, insufficient obs
"""

from __future__ import annotations

import copy
import math

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.dsl.spec import (
    BudgetDef,
    Direction,
    ObjectiveDef,
    OptimizationSpec,
    ParameterDef,
    ParamType,
)
from optimization_copilot.screening.screener import ScreeningResult
from optimization_copilot.surgery.models import (
    ActionType,
    DerivedType,
    SurgeryAction,
    SurgeryReport,
)
from optimization_copilot.surgery.surgeon import SearchSpaceSurgeon


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_specs() -> list[ParameterSpec]:
    """Three continuous parameters on [0, 10]."""
    return [
        ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ParameterSpec(name="x3", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
    ]


def _make_snapshot_concentrated(n_obs: int = 30) -> CampaignSnapshot:
    """All successful obs with x1 concentrated in [4, 6] out of [0, 10].

    x2 and x3 spread across the full range.
    Should trigger tightening for x1 but not x2/x3.
    """
    specs = _make_specs()
    obs: list[Observation] = []
    for i in range(n_obs):
        t = i / max(n_obs - 1, 1)
        obs.append(Observation(
            iteration=i,
            parameters={
                "x1": 4.0 + 2.0 * t,        # [4, 6] — concentrated
                "x2": 10.0 * t,               # [0, 10] — full range
                "x3": 10.0 * t,               # [0, 10] — full range
            },
            kpi_values={"y": 1.0 + t},
            is_failure=False,
            timestamp=float(i),
        ))
    return CampaignSnapshot(
        campaign_id="concentrated",
        parameter_specs=specs,
        observations=obs,
        objective_names=["y"],
        objective_directions=["minimize"],
        current_iteration=n_obs,
    )


def _make_snapshot_spread(n_obs: int = 30) -> CampaignSnapshot:
    """All parameters spread across the full range — no tightening expected."""
    specs = _make_specs()
    obs: list[Observation] = []
    for i in range(n_obs):
        t = i / max(n_obs - 1, 1)
        obs.append(Observation(
            iteration=i,
            parameters={
                "x1": 10.0 * t,
                "x2": 10.0 * t,
                "x3": 10.0 * t,
            },
            kpi_values={"y": 1.0 + t},
            is_failure=False,
            timestamp=float(i),
        ))
    return CampaignSnapshot(
        campaign_id="spread",
        parameter_specs=specs,
        observations=obs,
        objective_names=["y"],
        objective_directions=["minimize"],
        current_iteration=n_obs,
    )


def _make_snapshot_with_redundancy(n_obs: int = 30) -> CampaignSnapshot:
    """x2 = x1 + small noise -> should detect redundancy.

    x3 is independent (reverse order).
    """
    specs = _make_specs()
    obs: list[Observation] = []
    for i in range(n_obs):
        t = i / max(n_obs - 1, 1)
        x1_val = 10.0 * t
        x2_val = x1_val + (0.01 * (i % 3 - 1))  # tiny noise
        x3_val = 10.0 * (1.0 - t)  # anti-correlated with x1
        obs.append(Observation(
            iteration=i,
            parameters={"x1": x1_val, "x2": x2_val, "x3": x3_val},
            kpi_values={"y": 1.0 + t},
            is_failure=False,
            timestamp=float(i),
        ))
    return CampaignSnapshot(
        campaign_id="redundancy",
        parameter_specs=specs,
        observations=obs,
        objective_names=["y"],
        objective_directions=["minimize"],
        current_iteration=n_obs,
    )


def _make_screening_result(importance_scores: dict[str, float]) -> ScreeningResult:
    """Create a ScreeningResult with specified importance scores."""
    ranked = sorted(importance_scores, key=importance_scores.get, reverse=True)
    return ScreeningResult(
        ranked_parameters=ranked,
        importance_scores=importance_scores,
        suspected_interactions=[],
        directionality={k: 1.0 for k in importance_scores},
        recommended_step_sizes={k: 0.1 for k in importance_scores},
    )


def _make_opt_spec() -> OptimizationSpec:
    """Create OptimizationSpec with 3 continuous params for apply() tests."""
    return OptimizationSpec(
        campaign_id="test-spec",
        parameters=[
            ParameterDef(name="x1", type=ParamType.CONTINUOUS, lower=0.0, upper=10.0),
            ParameterDef(name="x2", type=ParamType.CONTINUOUS, lower=0.0, upper=10.0),
            ParameterDef(name="x3", type=ParamType.CONTINUOUS, lower=0.0, upper=10.0),
        ],
        objectives=[ObjectiveDef(name="y", direction=Direction.MINIMIZE)],
        budget=BudgetDef(max_samples=100),
    )


def _make_conditional_snapshot(n_obs: int = 30) -> CampaignSnapshot:
    """Synthetic data with a clear conditional pattern.

    When x1 <= median, y depends strongly on x2.
    When x1 > median, y is nearly constant regardless of x2 (low correlation).
    The overall correlation of x2-y is moderate (driven by the below-median half),
    but the above-median subset has much lower correlation. This triggers a
    conditional freeze of x2 when x1 is above median.
    """
    specs = [
        ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
    ]
    obs: list[Observation] = []
    for i in range(n_obs):
        t = i / max(n_obs - 1, 1)
        x1_val = 10.0 * t
        # x2 covers range in a varied pattern
        x2_val = 10.0 * ((i * 7) % n_obs) / n_obs

        if x1_val <= 5.0:
            # Below median: y depends strongly on x2
            y_val = 3.0 * x2_val + 0.01 * i
        else:
            # Above median: y is nearly constant (tiny noise, no x2 dependency)
            y_val = 15.0 + 0.001 * (i % 7)

        obs.append(Observation(
            iteration=i,
            parameters={"x1": x1_val, "x2": x2_val},
            kpi_values={"y": y_val},
            is_failure=False,
            timestamp=float(i),
        ))
    return CampaignSnapshot(
        campaign_id="conditional",
        parameter_specs=specs,
        observations=obs,
        objective_names=["y"],
        objective_directions=["minimize"],
        current_iteration=n_obs,
    )


# ---------------------------------------------------------------------------
# Tests: SurgeryAction
# ---------------------------------------------------------------------------


class TestSurgeryAction:
    """Construction and serialization of SurgeryAction."""

    def test_construction_basic(self):
        action = SurgeryAction(
            action_type=ActionType.TIGHTEN_RANGE,
            target_params=["x1"],
            new_lower=2.0,
            new_upper=8.0,
            reason="test",
            confidence=0.9,
        )
        assert action.action_type == ActionType.TIGHTEN_RANGE
        assert action.target_params == ["x1"]
        assert action.new_lower == 2.0
        assert action.new_upper == 8.0

    def test_each_action_type(self):
        """Every ActionType enum can be used to construct an action."""
        for at in ActionType:
            action = SurgeryAction(action_type=at, target_params=["p1"])
            assert action.action_type == at

    def test_to_dict_from_dict_round_trip(self):
        action = SurgeryAction(
            action_type=ActionType.DERIVE_PARAMETER,
            target_params=["x1", "x2"],
            derived_type=DerivedType.RATIO,
            derived_name="x1_over_x2",
            derived_source_params=["x1", "x2"],
            reason="moderate correlation",
            confidence=0.7,
            evidence={"correlation": 0.65},
        )
        d = action.to_dict()
        restored = SurgeryAction.from_dict(d)

        assert restored.action_type == action.action_type
        assert restored.target_params == action.target_params
        assert restored.derived_type == action.derived_type
        assert restored.derived_name == action.derived_name
        assert restored.derived_source_params == action.derived_source_params
        assert restored.reason == action.reason
        assert restored.confidence == action.confidence
        assert restored.evidence == action.evidence

    def test_to_dict_from_dict_no_derived(self):
        """Round-trip when derived_type is None."""
        action = SurgeryAction(
            action_type=ActionType.FREEZE_PARAMETER,
            target_params=["x3"],
            freeze_value=5.0,
            reason="low importance",
            confidence=0.8,
        )
        d = action.to_dict()
        assert d["derived_type"] is None
        restored = SurgeryAction.from_dict(d)
        assert restored.derived_type is None
        assert restored.freeze_value == 5.0

    def test_to_dict_contains_all_fields(self):
        action = SurgeryAction(
            action_type=ActionType.TIGHTEN_RANGE,
            target_params=["x1"],
        )
        d = action.to_dict()
        expected_keys = {
            "action_type", "target_params", "new_lower", "new_upper",
            "freeze_value", "condition_param", "condition_threshold",
            "condition_direction", "merge_into", "derived_type",
            "derived_name", "derived_source_params", "reason",
            "confidence", "evidence",
        }
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Tests: SurgeryReport
# ---------------------------------------------------------------------------


class TestSurgeryReport:
    """Construction, properties, and serialization of SurgeryReport."""

    def test_construction_empty(self):
        report = SurgeryReport()
        assert report.n_actions == 0
        assert report.has_actions is False
        assert report.original_dim == 0
        assert report.effective_dim == 0

    def test_n_actions_and_has_actions(self):
        actions = [
            SurgeryAction(action_type=ActionType.TIGHTEN_RANGE, target_params=["x1"]),
            SurgeryAction(action_type=ActionType.FREEZE_PARAMETER, target_params=["x2"]),
        ]
        report = SurgeryReport(actions=actions, original_dim=3, effective_dim=2)
        assert report.n_actions == 2
        assert report.has_actions is True

    def test_actions_by_type(self):
        actions = [
            SurgeryAction(action_type=ActionType.TIGHTEN_RANGE, target_params=["x1"]),
            SurgeryAction(action_type=ActionType.FREEZE_PARAMETER, target_params=["x2"]),
            SurgeryAction(action_type=ActionType.TIGHTEN_RANGE, target_params=["x3"]),
        ]
        report = SurgeryReport(actions=actions)
        tighten_actions = report.actions_by_type(ActionType.TIGHTEN_RANGE)
        assert len(tighten_actions) == 2
        freeze_actions = report.actions_by_type(ActionType.FREEZE_PARAMETER)
        assert len(freeze_actions) == 1
        merge_actions = report.actions_by_type(ActionType.MERGE_PARAMETERS)
        assert len(merge_actions) == 0

    def test_to_dict_from_dict_round_trip(self):
        actions = [
            SurgeryAction(
                action_type=ActionType.TIGHTEN_RANGE,
                target_params=["x1"],
                new_lower=2.0, new_upper=8.0,
                reason="concentrated",
                confidence=0.9,
                evidence={"n_values": 30},
            ),
            SurgeryAction(
                action_type=ActionType.FREEZE_PARAMETER,
                target_params=["x2"],
                freeze_value=5.0,
                reason="low importance",
                confidence=0.8,
            ),
        ]
        report = SurgeryReport(
            actions=actions,
            original_dim=3,
            effective_dim=2,
            space_reduction_ratio=1 / 3,
            reason_codes=["range_tightening", "freeze_unimportant"],
            metadata={"n_successful": 30},
        )
        d = report.to_dict()
        restored = SurgeryReport.from_dict(d)

        assert restored.n_actions == 2
        assert restored.original_dim == 3
        assert restored.effective_dim == 2
        assert abs(restored.space_reduction_ratio - 1 / 3) < 1e-9
        assert restored.reason_codes == ["range_tightening", "freeze_unimportant"]
        assert restored.metadata == {"n_successful": 30}
        assert restored.actions[0].action_type == ActionType.TIGHTEN_RANGE
        assert restored.actions[1].freeze_value == 5.0


# ---------------------------------------------------------------------------
# Tests: Range Tightening
# ---------------------------------------------------------------------------


class TestRangeTightening:
    """Range tightening detection on concentrated vs spread data."""

    def test_concentrated_data_produces_tightening(self):
        """When x1 values cluster in [4, 6] out of [0, 10], tightening is triggered."""
        snap = _make_snapshot_concentrated(30)
        surgeon = SearchSpaceSurgeon(min_observations=10)
        report = surgeon.diagnose(snap)

        tighten = report.actions_by_type(ActionType.TIGHTEN_RANGE)
        target_names = [a.target_params[0] for a in tighten]
        assert "x1" in target_names, "x1 should be tightened (concentrated data)"

        # The tightened range for x1 should be within [0, 10]
        x1_action = [a for a in tighten if a.target_params[0] == "x1"][0]
        assert x1_action.new_lower >= 0.0
        assert x1_action.new_upper <= 10.0
        # Tightened range should be significantly smaller than original
        assert (x1_action.new_upper - x1_action.new_lower) < 8.0

    def test_spread_data_no_tightening(self):
        """When data spans the full range, reduction < 20%, no tightening."""
        snap = _make_snapshot_spread(30)
        surgeon = SearchSpaceSurgeon(min_observations=10)
        report = surgeon.diagnose(snap)

        tighten = report.actions_by_type(ActionType.TIGHTEN_RANGE)
        # x1, x2, x3 all span [0, 10] fully so no tightening expected
        assert len(tighten) == 0

    def test_categorical_params_skipped(self):
        """Categorical parameters should not produce tightening actions."""
        specs = [
            ParameterSpec(name="cat", type=VariableType.CATEGORICAL, categories=["a", "b"]),
            ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ]
        obs = [
            Observation(
                iteration=i,
                parameters={"cat": "a", "x1": 4.0 + 2.0 * i / 29},
                kpi_values={"y": float(i)},
                is_failure=False,
                timestamp=float(i),
            )
            for i in range(30)
        ]
        snap = CampaignSnapshot(
            campaign_id="cat-test",
            parameter_specs=specs,
            observations=obs,
            objective_names=["y"],
            objective_directions=["minimize"],
            current_iteration=30,
        )
        surgeon = SearchSpaceSurgeon(min_observations=10)
        report = surgeon.diagnose(snap)

        tighten = report.actions_by_type(ActionType.TIGHTEN_RANGE)
        for a in tighten:
            assert "cat" not in a.target_params

    def test_insufficient_data_returns_empty(self):
        """Fewer than 5 values for a parameter should not trigger tightening."""
        specs = _make_specs()
        obs = [
            Observation(
                iteration=i,
                parameters={"x1": 5.0, "x2": 5.0, "x3": 5.0},
                kpi_values={"y": 1.0},
                is_failure=False,
                timestamp=float(i),
            )
            for i in range(4)
        ]
        snap = CampaignSnapshot(
            campaign_id="tiny",
            parameter_specs=specs,
            observations=obs,
            objective_names=["y"],
            objective_directions=["minimize"],
            current_iteration=4,
        )
        # min_observations=10, but we only have 4 obs -> insufficient_data guard
        surgeon = SearchSpaceSurgeon(min_observations=10)
        report = surgeon.diagnose(snap)
        assert report.n_actions == 0
        assert "insufficient_data" in report.reason_codes

    def test_new_bounds_within_original(self):
        """Tightened bounds must remain within original bounds."""
        snap = _make_snapshot_concentrated(30)
        surgeon = SearchSpaceSurgeon(min_observations=10)
        report = surgeon.diagnose(snap)

        for a in report.actions_by_type(ActionType.TIGHTEN_RANGE):
            param_name = a.target_params[0]
            orig_spec = [s for s in snap.parameter_specs if s.name == param_name][0]
            assert a.new_lower >= orig_spec.lower
            assert a.new_upper <= orig_spec.upper


# ---------------------------------------------------------------------------
# Tests: Freeze Candidates
# ---------------------------------------------------------------------------


class TestFreezeCandidates:
    """Freeze detection via importance scores from screening."""

    def test_low_importance_param_detected(self):
        """Parameter with importance < 0.05 should be flagged for freezing."""
        snap = _make_snapshot_spread(30)
        screening = _make_screening_result({"x1": 0.9, "x2": 0.01, "x3": 0.5})
        surgeon = SearchSpaceSurgeon(min_observations=10, freeze_importance_threshold=0.05)
        report = surgeon.diagnose(snap, screening_result=screening)

        freeze = report.actions_by_type(ActionType.FREEZE_PARAMETER)
        frozen_names = [a.target_params[0] for a in freeze]
        assert "x2" in frozen_names, "x2 (importance=0.01) should be frozen"

    def test_param_at_threshold_not_frozen(self):
        """Parameter with importance exactly at threshold should NOT be frozen."""
        snap = _make_snapshot_spread(30)
        screening = _make_screening_result({"x1": 0.9, "x2": 0.05, "x3": 0.5})
        surgeon = SearchSpaceSurgeon(min_observations=10, freeze_importance_threshold=0.05)
        report = surgeon.diagnose(snap, screening_result=screening)

        freeze = report.actions_by_type(ActionType.FREEZE_PARAMETER)
        frozen_names = [a.target_params[0] for a in freeze]
        assert "x2" not in frozen_names, "x2 at threshold should not be frozen"

    def test_freeze_value_from_best_observation(self):
        """Frozen value should come from the observation with the best (lowest) KPI."""
        specs = _make_specs()
        obs = []
        # Create observations: the best y (lowest) occurs when x2=7.0
        for i in range(20):
            t = i / 19
            y_val = 10.0 - 5.0 * t if i < 10 else 10.0 + t
            obs.append(Observation(
                iteration=i,
                parameters={"x1": 5.0, "x2": 3.0 + 4.0 * t, "x3": 5.0},
                kpi_values={"y": y_val},
                is_failure=False,
                timestamp=float(i),
            ))
        # Best observation is i=9 (y=10 - 5*9/19 ~= 7.63), x2 = 3.0 + 4.0*9/19 ~= 4.89
        best_idx = min(range(len(obs)), key=lambda j: obs[j].kpi_values["y"])
        expected_x2 = obs[best_idx].parameters["x2"]

        snap = CampaignSnapshot(
            campaign_id="freeze-val",
            parameter_specs=specs,
            observations=obs,
            objective_names=["y"],
            objective_directions=["minimize"],
            current_iteration=20,
        )
        screening = _make_screening_result({"x1": 0.9, "x2": 0.01, "x3": 0.5})
        surgeon = SearchSpaceSurgeon(min_observations=10, freeze_importance_threshold=0.05)
        report = surgeon.diagnose(snap, screening_result=screening)

        freeze = report.actions_by_type(ActionType.FREEZE_PARAMETER)
        x2_freeze = [a for a in freeze if a.target_params[0] == "x2"]
        assert len(x2_freeze) == 1
        assert x2_freeze[0].freeze_value == expected_x2


# ---------------------------------------------------------------------------
# Tests: Conditional Freezing
# ---------------------------------------------------------------------------


class TestConditionalFreezing:
    """Conditional freeze detection with synthetic conditional patterns."""

    def test_conditional_pattern_detected(self):
        """Synthetic data with a clear conditional pattern should produce a conditional freeze."""
        snap = _make_conditional_snapshot(n_obs=30)
        surgeon = SearchSpaceSurgeon(min_observations=10)
        report = surgeon.diagnose(snap)

        cond = report.actions_by_type(ActionType.CONDITIONAL_FREEZE)
        # Should have at least one conditional freeze
        assert len(cond) >= 1
        # Check direction is valid
        for a in cond:
            assert a.condition_direction in ("above", "below")

    def test_need_enough_observations(self):
        """With < 15 observations, conditional freezing should not trigger."""
        snap = _make_conditional_snapshot(n_obs=14)
        surgeon = SearchSpaceSurgeon(min_observations=10)
        report = surgeon.diagnose(snap)

        cond = report.actions_by_type(ActionType.CONDITIONAL_FREEZE)
        assert len(cond) == 0

    def test_direction_is_above_or_below(self):
        """The condition_direction field must be 'above' or 'below'."""
        snap = _make_conditional_snapshot(n_obs=30)
        surgeon = SearchSpaceSurgeon(min_observations=10)
        report = surgeon.diagnose(snap)

        for a in report.actions_by_type(ActionType.CONDITIONAL_FREEZE):
            assert a.condition_direction in ("above", "below")
            assert a.condition_param is not None
            assert a.condition_threshold is not None


# ---------------------------------------------------------------------------
# Tests: Redundancy Detection
# ---------------------------------------------------------------------------


class TestRedundancy:
    """Redundancy detection for correlated parameter pairs."""

    def test_perfectly_correlated_pair_merged(self):
        """x2 ~ x1 + noise (r close to 1.0) should trigger merge."""
        snap = _make_snapshot_with_redundancy(30)
        surgeon = SearchSpaceSurgeon(min_observations=10, correlation_threshold=0.9)
        report = surgeon.diagnose(snap)

        merge = report.actions_by_type(ActionType.MERGE_PARAMETERS)
        merged_pairs = [(a.target_params[0], a.target_params[1]) for a in merge]
        # x1 and x2 are highly correlated
        found_x1_x2 = any(
            ("x1" in pair and "x2" in pair) for pair in merged_pairs
        )
        assert found_x1_x2, "x1 and x2 should be merged (highly correlated)"

    def test_anti_correlated_pair_merged(self):
        """x3 = 10 - x1 (r close to -1.0) should also trigger merge (abs correlation)."""
        snap = _make_snapshot_with_redundancy(30)
        surgeon = SearchSpaceSurgeon(min_observations=10, correlation_threshold=0.9)
        report = surgeon.diagnose(snap)

        merge = report.actions_by_type(ActionType.MERGE_PARAMETERS)
        merged_pairs = [(a.target_params[0], a.target_params[1]) for a in merge]
        # x1 and x3 are anti-correlated (|r| near 1.0)
        found_x1_x3 = any(
            ("x1" in pair and "x3" in pair) for pair in merged_pairs
        )
        assert found_x1_x3, "x1 and x3 should be merged (anti-correlated)"

    def test_uncorrelated_pair_not_merged(self):
        """Independent parameters should not be merged."""
        specs = _make_specs()
        obs = []
        # x1 and x2 are independent (different patterns, no correlation)
        for i in range(30):
            t = i / 29
            obs.append(Observation(
                iteration=i,
                parameters={
                    "x1": 10.0 * t,
                    "x2": 5.0 + 2.0 * math.sin(2.0 * math.pi * t * 3),
                    "x3": 5.0 + 2.0 * math.cos(2.0 * math.pi * t * 7),
                },
                kpi_values={"y": 1.0 + t},
                is_failure=False,
                timestamp=float(i),
            ))
        snap = CampaignSnapshot(
            campaign_id="uncorrelated",
            parameter_specs=specs,
            observations=obs,
            objective_names=["y"],
            objective_directions=["minimize"],
            current_iteration=30,
        )
        surgeon = SearchSpaceSurgeon(min_observations=10, correlation_threshold=0.9)
        report = surgeon.diagnose(snap)

        merge = report.actions_by_type(ActionType.MERGE_PARAMETERS)
        assert len(merge) == 0, "Uncorrelated pairs should not be merged"

    def test_no_double_merging(self):
        """The already_merged set should prevent a parameter from being merged twice."""
        # x2 = x1 + noise and x3 = x1 + noise => both correlated with x1.
        # But x2 is merged first, and then x3 should still be considered
        # since x3 correlates with x1 separately. However, if x3 were already
        # in already_merged (e.g. from being secondary in a prior merge), it
        # wouldn't be merged again. We test that x2 is only merged once.
        specs = _make_specs()
        obs = []
        for i in range(30):
            t = i / 29
            base = 10.0 * t
            obs.append(Observation(
                iteration=i,
                parameters={
                    "x1": base,
                    "x2": base + 0.001 * (i % 3),
                    "x3": base + 0.002 * (i % 5),
                },
                kpi_values={"y": 1.0 + t},
                is_failure=False,
                timestamp=float(i),
            ))
        snap = CampaignSnapshot(
            campaign_id="double-merge",
            parameter_specs=specs,
            observations=obs,
            objective_names=["y"],
            objective_directions=["minimize"],
            current_iteration=30,
        )
        surgeon = SearchSpaceSurgeon(min_observations=10, correlation_threshold=0.9)
        report = surgeon.diagnose(snap)

        merge = report.actions_by_type(ActionType.MERGE_PARAMETERS)
        # Collect all secondary params (the ones that get frozen)
        secondaries = []
        for a in merge:
            for p in a.target_params:
                if p != a.merge_into:
                    secondaries.append(p)
        # No duplicate secondaries
        assert len(secondaries) == len(set(secondaries)), (
            "A parameter should not be merged as secondary more than once"
        )


# ---------------------------------------------------------------------------
# Tests: Derived Parameters
# ---------------------------------------------------------------------------


class TestDerivedParams:
    """Derived parameter suggestions (LOG, RATIO)."""

    def test_wide_range_log_suggestion(self):
        """Parameter with upper/lower >= 100 should get LOG suggestion."""
        specs = [
            ParameterSpec(name="x_wide", type=VariableType.CONTINUOUS, lower=0.01, upper=10.0),
            # ratio = 10.0 / 0.01 = 1000 >= 100, should trigger LOG
        ]
        obs = [
            Observation(
                iteration=i,
                parameters={"x_wide": 0.01 + 9.99 * i / 29},
                kpi_values={"y": float(i)},
                is_failure=False,
                timestamp=float(i),
            )
            for i in range(30)
        ]
        snap = CampaignSnapshot(
            campaign_id="wide-range",
            parameter_specs=specs,
            observations=obs,
            objective_names=["y"],
            objective_directions=["minimize"],
            current_iteration=30,
        )
        surgeon = SearchSpaceSurgeon(min_observations=10)
        report = surgeon.diagnose(snap)

        derived = report.actions_by_type(ActionType.DERIVE_PARAMETER)
        log_actions = [a for a in derived if a.derived_type == DerivedType.LOG]
        assert len(log_actions) >= 1
        assert log_actions[0].derived_name == "log_x_wide"

    def test_moderately_correlated_ratio_suggestion(self):
        """Two params with 0.5 <= |corr| < 0.9 should trigger RATIO suggestion."""
        specs = [
            ParameterSpec(name="a", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            ParameterSpec(name="b", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ]
        obs = []
        for i in range(30):
            t = i / 29
            a_val = 10.0 * t
            # b = a + significant noise => moderate correlation
            noise = 3.0 * math.sin(2.0 * math.pi * t * 5)
            b_val = max(0.0, min(10.0, a_val + noise))
            obs.append(Observation(
                iteration=i,
                parameters={"a": a_val, "b": b_val},
                kpi_values={"y": 1.0 + t},
                is_failure=False,
                timestamp=float(i),
            ))
        snap = CampaignSnapshot(
            campaign_id="moderate-corr",
            parameter_specs=specs,
            observations=obs,
            objective_names=["y"],
            objective_directions=["minimize"],
            current_iteration=30,
        )
        surgeon = SearchSpaceSurgeon(min_observations=10, correlation_threshold=0.9)
        report = surgeon.diagnose(snap)

        derived = report.actions_by_type(ActionType.DERIVE_PARAMETER)
        ratio_actions = [a for a in derived if a.derived_type == DerivedType.RATIO]
        assert len(ratio_actions) >= 1
        assert ratio_actions[0].derived_name == "a_over_b"

    def test_narrow_range_no_log(self):
        """Parameter with upper/lower < 100 should NOT get LOG suggestion."""
        specs = [
            ParameterSpec(name="x_narrow", type=VariableType.CONTINUOUS, lower=1.0, upper=10.0),
            # ratio = 10.0 / 1.0 = 10 < 100, no LOG
        ]
        obs = [
            Observation(
                iteration=i,
                parameters={"x_narrow": 1.0 + 9.0 * i / 29},
                kpi_values={"y": float(i)},
                is_failure=False,
                timestamp=float(i),
            )
            for i in range(30)
        ]
        snap = CampaignSnapshot(
            campaign_id="narrow-range",
            parameter_specs=specs,
            observations=obs,
            objective_names=["y"],
            objective_directions=["minimize"],
            current_iteration=30,
        )
        surgeon = SearchSpaceSurgeon(min_observations=10)
        report = surgeon.diagnose(snap)

        derived = report.actions_by_type(ActionType.DERIVE_PARAMETER)
        log_actions = [a for a in derived if a.derived_type == DerivedType.LOG]
        assert len(log_actions) == 0, "Narrow range should not suggest LOG"


# ---------------------------------------------------------------------------
# Tests: Apply
# ---------------------------------------------------------------------------


class TestApply:
    """Applying surgery actions to an OptimizationSpec."""

    def test_tightening_modifies_bounds(self):
        spec = _make_opt_spec()
        report = SurgeryReport(
            actions=[
                SurgeryAction(
                    action_type=ActionType.TIGHTEN_RANGE,
                    target_params=["x1"],
                    new_lower=3.0,
                    new_upper=7.0,
                ),
            ],
            original_dim=3,
            effective_dim=3,
        )
        surgeon = SearchSpaceSurgeon()
        new_spec = surgeon.apply(spec, report)

        x1 = [p for p in new_spec.parameters if p.name == "x1"][0]
        assert x1.lower == 3.0
        assert x1.upper == 7.0

    def test_freezing_sets_frozen(self):
        spec = _make_opt_spec()
        report = SurgeryReport(
            actions=[
                SurgeryAction(
                    action_type=ActionType.FREEZE_PARAMETER,
                    target_params=["x2"],
                    freeze_value=4.5,
                ),
            ],
            original_dim=3,
            effective_dim=2,
        )
        surgeon = SearchSpaceSurgeon()
        new_spec = surgeon.apply(spec, report)

        x2 = [p for p in new_spec.parameters if p.name == "x2"][0]
        assert x2.frozen is True
        assert x2.frozen_value == 4.5

    def test_merge_freezes_secondary(self):
        spec = _make_opt_spec()
        report = SurgeryReport(
            actions=[
                SurgeryAction(
                    action_type=ActionType.MERGE_PARAMETERS,
                    target_params=["x1", "x2"],
                    merge_into="x1",
                    freeze_value=5.0,
                ),
            ],
            original_dim=3,
            effective_dim=2,
        )
        surgeon = SearchSpaceSurgeon()
        new_spec = surgeon.apply(spec, report)

        x1 = [p for p in new_spec.parameters if p.name == "x1"][0]
        x2 = [p for p in new_spec.parameters if p.name == "x2"][0]
        # x1 (primary) should NOT be frozen
        assert x1.frozen is False
        # x2 (secondary) should be frozen
        assert x2.frozen is True
        assert x2.frozen_value == 5.0

    def test_original_spec_not_mutated(self):
        spec = _make_opt_spec()
        original_lower = spec.parameters[0].lower
        original_upper = spec.parameters[0].upper
        original_frozen = spec.parameters[1].frozen

        report = SurgeryReport(
            actions=[
                SurgeryAction(
                    action_type=ActionType.TIGHTEN_RANGE,
                    target_params=["x1"],
                    new_lower=3.0, new_upper=7.0,
                ),
                SurgeryAction(
                    action_type=ActionType.FREEZE_PARAMETER,
                    target_params=["x2"],
                    freeze_value=4.5,
                ),
            ],
        )
        surgeon = SearchSpaceSurgeon()
        surgeon.apply(spec, report)

        # Original spec should be unchanged
        assert spec.parameters[0].lower == original_lower
        assert spec.parameters[0].upper == original_upper
        assert spec.parameters[1].frozen == original_frozen


# ---------------------------------------------------------------------------
# Tests: Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same inputs produce same report."""

    def test_same_inputs_same_report(self):
        snap = _make_snapshot_concentrated(30)
        screening = _make_screening_result({"x1": 0.8, "x2": 0.01, "x3": 0.5})
        surgeon = SearchSpaceSurgeon(min_observations=10)

        report1 = surgeon.diagnose(snap, screening_result=screening, seed=42)
        report2 = surgeon.diagnose(snap, screening_result=screening, seed=42)

        assert report1.n_actions == report2.n_actions
        assert report1.original_dim == report2.original_dim
        assert report1.effective_dim == report2.effective_dim
        assert report1.space_reduction_ratio == report2.space_reduction_ratio
        assert report1.reason_codes == report2.reason_codes

        for a1, a2 in zip(report1.actions, report2.actions):
            assert a1.action_type == a2.action_type
            assert a1.target_params == a2.target_params
            assert a1.new_lower == a2.new_lower
            assert a1.new_upper == a2.new_upper
            assert a1.freeze_value == a2.freeze_value
            assert a1.confidence == a2.confidence


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty snapshot, single parameter, all categorical, insufficient obs."""

    def test_empty_snapshot(self):
        """Empty snapshot should not crash and return no actions."""
        snap = CampaignSnapshot(
            campaign_id="empty",
            parameter_specs=_make_specs(),
            observations=[],
            objective_names=["y"],
            objective_directions=["minimize"],
        )
        surgeon = SearchSpaceSurgeon(min_observations=10)
        report = surgeon.diagnose(snap)

        assert report.n_actions == 0
        assert report.has_actions is False
        assert "insufficient_data" in report.reason_codes
        assert report.original_dim == 3
        assert report.effective_dim == 3

    def test_single_parameter(self):
        """Snapshot with a single parameter should work without error."""
        specs = [ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0)]
        obs = [
            Observation(
                iteration=i,
                parameters={"x1": 4.0 + 2.0 * i / 14},
                kpi_values={"y": float(i)},
                is_failure=False,
                timestamp=float(i),
            )
            for i in range(15)
        ]
        snap = CampaignSnapshot(
            campaign_id="single",
            parameter_specs=specs,
            observations=obs,
            objective_names=["y"],
            objective_directions=["minimize"],
            current_iteration=15,
        )
        surgeon = SearchSpaceSurgeon(min_observations=10)
        report = surgeon.diagnose(snap)

        # Should not crash; may or may not produce tightening for x1
        assert report.original_dim == 1
        assert report.effective_dim >= 0

    def test_all_categorical(self):
        """All-categorical snapshot should produce no continuous-dependent actions."""
        specs = [
            ParameterSpec(name="c1", type=VariableType.CATEGORICAL, categories=["a", "b", "c"]),
            ParameterSpec(name="c2", type=VariableType.CATEGORICAL, categories=["x", "y"]),
        ]
        obs = [
            Observation(
                iteration=i,
                parameters={"c1": "a", "c2": "x"},
                kpi_values={"y": float(i)},
                is_failure=False,
                timestamp=float(i),
            )
            for i in range(20)
        ]
        snap = CampaignSnapshot(
            campaign_id="all-cat",
            parameter_specs=specs,
            observations=obs,
            objective_names=["y"],
            objective_directions=["minimize"],
            current_iteration=20,
        )
        surgeon = SearchSpaceSurgeon(min_observations=10)
        report = surgeon.diagnose(snap)

        # No tightening, no redundancy, no derived params for categoricals
        assert len(report.actions_by_type(ActionType.TIGHTEN_RANGE)) == 0
        assert len(report.actions_by_type(ActionType.MERGE_PARAMETERS)) == 0
        assert len(report.actions_by_type(ActionType.DERIVE_PARAMETER)) == 0

    def test_insufficient_obs(self):
        """Fewer than min_observations should return empty report with reason code."""
        specs = _make_specs()
        obs = [
            Observation(
                iteration=i,
                parameters={"x1": 5.0, "x2": 5.0, "x3": 5.0},
                kpi_values={"y": 1.0},
                is_failure=False,
                timestamp=float(i),
            )
            for i in range(5)
        ]
        snap = CampaignSnapshot(
            campaign_id="few-obs",
            parameter_specs=specs,
            observations=obs,
            objective_names=["y"],
            objective_directions=["minimize"],
            current_iteration=5,
        )
        surgeon = SearchSpaceSurgeon(min_observations=10)
        report = surgeon.diagnose(snap)

        assert report.n_actions == 0
        assert "insufficient_data" in report.reason_codes
        assert report.metadata["n_successful"] == 5
        assert report.metadata["min_required"] == 10
