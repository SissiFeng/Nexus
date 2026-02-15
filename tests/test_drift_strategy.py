"""Tests for drift-triggered strategy adaptation (Track 2).

Verifies:
- Mild drift → recency reweighting + exploration boost only
- Moderate drift → re-screening triggered
- Severe drift → phase reset + backend switch
- No drift → no actions
- End-to-end: synthetic data with step change → strategy changes
- Smooth ramp does NOT trigger sudden-type actions
"""

from __future__ import annotations

from dataclasses import dataclass

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    Phase,
    RiskPosture,
    StabilizeSpec,
    StrategyDecision,
    VariableType,
)
from optimization_copilot.drift.detector import DriftDetector, DriftReport
from optimization_copilot.drift.strategy import (
    DriftAction,
    DriftAdaptation,
    DriftStrategyAdapter,
)


# ---------------------------------------------------------------------------
# Mock drift report
# ---------------------------------------------------------------------------


def _mock_report(
    severity: float = 0.0,
    detected: bool = False,
    drift_type: str = "none",
    affected: list[str] | None = None,
) -> DriftReport:
    return DriftReport(
        drift_detected=detected,
        drift_score=severity,
        drift_type=drift_type,
        affected_parameters=affected or [],
        recommended_action="continue",
    )


def _base_decision(phase: Phase = Phase.LEARNING) -> StrategyDecision:
    return StrategyDecision(
        backend_name="tpe",
        stabilize_spec=StabilizeSpec(),
        exploration_strength=0.5,
        batch_size=3,
        risk_posture=RiskPosture.MODERATE,
        phase=phase,
        reason_codes=["test"],
    )


# ---------------------------------------------------------------------------
# Helpers for end-to-end tests
# ---------------------------------------------------------------------------


def _make_specs() -> list[ParameterSpec]:
    return [
        ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
    ]


def _make_snapshot(kpi_values: list[float]) -> CampaignSnapshot:
    n = len(kpi_values)
    obs = [
        Observation(
            iteration=i,
            parameters={"x1": float(i), "x2": float(i) * 0.5},
            kpi_values={"y": kpi_values[i]},
            timestamp=float(i),
        )
        for i in range(n)
    ]
    return CampaignSnapshot(
        campaign_id="drift-strategy-test",
        parameter_specs=_make_specs(),
        observations=obs,
        objective_names=["y"],
        objective_directions=["maximize"],
        current_iteration=n,
    )


# ---------------------------------------------------------------------------
# Tests: No drift
# ---------------------------------------------------------------------------


class TestNoDrift:
    def test_no_drift_no_actions(self):
        """Below threshold → no adaptation."""
        adapter = DriftStrategyAdapter()
        report = _mock_report(severity=0.1, detected=False)
        adaptation = adapter.adapt(report, Phase.LEARNING)

        assert len(adaptation.actions) == 0
        assert adaptation.adapted_phase is None
        assert adaptation.exploration_delta == 0.0

    def test_below_threshold_no_actions(self):
        """Drift detected but below exploration boost threshold."""
        adapter = DriftStrategyAdapter(exploration_boost_threshold=0.5)
        report = _mock_report(severity=0.4, detected=True)
        adaptation = adapter.adapt(report, Phase.LEARNING)

        assert len(adaptation.actions) == 0


# ---------------------------------------------------------------------------
# Tests: Mild drift
# ---------------------------------------------------------------------------


class TestMildDrift:
    def test_recency_reweight_triggered(self):
        """Mild drift should trigger recency reweighting."""
        adapter = DriftStrategyAdapter()
        report = _mock_report(severity=0.35, detected=True)
        adaptation = adapter.adapt(report, Phase.LEARNING)

        assert adaptation.reweighting == "recency"
        action_types = [a.action_type for a in adaptation.actions]
        assert "recency_reweight" in action_types

    def test_exploration_boost_applied(self):
        """Mild drift should boost exploration."""
        adapter = DriftStrategyAdapter()
        report = _mock_report(severity=0.4, detected=True)
        adaptation = adapter.adapt(report, Phase.LEARNING, current_exploration=0.5)

        assert adaptation.exploration_delta > 0.0

    def test_no_phase_reset(self):
        """Mild drift should not reset phase."""
        adapter = DriftStrategyAdapter()
        report = _mock_report(severity=0.4, detected=True)
        adaptation = adapter.adapt(report, Phase.EXPLOITATION)

        assert adaptation.adapted_phase is None


# ---------------------------------------------------------------------------
# Tests: Moderate drift
# ---------------------------------------------------------------------------


class TestModerateDrift:
    def test_re_screen_triggered(self):
        """Moderate drift with affected params triggers re-screening."""
        adapter = DriftStrategyAdapter()
        report = _mock_report(
            severity=0.55, detected=True, affected=["x1", "x2"]
        )
        adaptation = adapter.adapt(report, Phase.LEARNING)

        action_types = [a.action_type for a in adaptation.actions]
        assert "re_screen" in action_types

    def test_no_re_screen_without_affected_params(self):
        """Re-screening requires affected parameters."""
        adapter = DriftStrategyAdapter()
        report = _mock_report(severity=0.55, detected=True, affected=[])
        adaptation = adapter.adapt(report, Phase.LEARNING)

        action_types = [a.action_type for a in adaptation.actions]
        assert "re_screen" not in action_types


# ---------------------------------------------------------------------------
# Tests: Severe drift
# ---------------------------------------------------------------------------


class TestSevereDrift:
    def test_phase_reset(self):
        """Severe drift resets exploitation → learning."""
        adapter = DriftStrategyAdapter()
        report = _mock_report(severity=0.8, detected=True)
        adaptation = adapter.adapt(report, Phase.EXPLOITATION)

        assert adaptation.adapted_phase == Phase.LEARNING

    def test_backend_switch_recommended(self):
        """Severe drift recommends drift-robust backends."""
        adapter = DriftStrategyAdapter()
        report = _mock_report(severity=0.7, detected=True)
        adaptation = adapter.adapt(report, Phase.LEARNING)

        action_types = [a.action_type for a in adaptation.actions]
        assert "backend_switch" in action_types

        switch_action = next(
            a for a in adaptation.actions if a.action_type == "backend_switch"
        )
        assert "random" in switch_action.parameters["recommended_backends"]

    def test_no_phase_reset_from_learning(self):
        """Phase reset only from exploitation/stagnation, not learning."""
        adapter = DriftStrategyAdapter()
        report = _mock_report(severity=0.8, detected=True)
        adaptation = adapter.adapt(report, Phase.LEARNING)

        assert adaptation.adapted_phase is None

    def test_phase_reset_from_stagnation(self):
        """Severe drift in stagnation → reset to learning."""
        adapter = DriftStrategyAdapter()
        report = _mock_report(severity=0.8, detected=True)
        adaptation = adapter.adapt(report, Phase.STAGNATION)

        assert adaptation.adapted_phase == Phase.LEARNING


# ---------------------------------------------------------------------------
# Tests: Apply adaptation to decision
# ---------------------------------------------------------------------------


class TestApplyAdaptation:
    def test_apply_exploration_boost(self):
        """Applying adaptation should increase exploration."""
        adapter = DriftStrategyAdapter()
        decision = _base_decision(Phase.LEARNING)
        report = _mock_report(severity=0.5, detected=True)

        adaptation = adapter.adapt(report, Phase.LEARNING, 0.5)
        result = adapter.apply(decision, adaptation)

        assert result.exploration_strength > decision.exploration_strength

    def test_apply_phase_reset(self):
        """Applying phase reset should change phase."""
        adapter = DriftStrategyAdapter()
        decision = _base_decision(Phase.EXPLOITATION)
        report = _mock_report(severity=0.8, detected=True)

        adaptation = adapter.adapt(report, Phase.EXPLOITATION, 0.2)
        result = adapter.apply(decision, adaptation)

        assert result.phase == Phase.LEARNING
        assert result.risk_posture == RiskPosture.CONSERVATIVE

    def test_apply_recency_reweighting(self):
        """Applying adaptation should set recency reweighting."""
        adapter = DriftStrategyAdapter()
        decision = _base_decision()
        report = _mock_report(severity=0.4, detected=True)

        adaptation = adapter.adapt(report, Phase.LEARNING)
        result = adapter.apply(decision, adaptation)

        assert result.stabilize_spec.reweighting_strategy == "recency"

    def test_apply_backend_switch(self):
        """Backend switch should change decision backend."""
        adapter = DriftStrategyAdapter()
        decision = _base_decision()
        report = _mock_report(severity=0.7, detected=True)

        adaptation = adapter.adapt(report, Phase.LEARNING)
        result = adapter.apply(decision, adaptation)

        assert result.backend_name in ("random", "latin_hypercube")

    def test_reason_codes_appended(self):
        """Drift adaptation adds reason codes to decision."""
        adapter = DriftStrategyAdapter()
        decision = _base_decision()
        report = _mock_report(severity=0.5, detected=True)

        adaptation = adapter.adapt(report, Phase.LEARNING)
        result = adapter.apply(decision, adaptation)

        drift_reasons = [r for r in result.reason_codes if r.startswith("drift_")]
        assert len(drift_reasons) > 0

    def test_metadata_includes_drift_info(self):
        """Decision metadata should include drift adaptation info."""
        adapter = DriftStrategyAdapter()
        decision = _base_decision()
        report = _mock_report(severity=0.5, detected=True)

        adaptation = adapter.adapt(report, Phase.LEARNING)
        result = adapter.apply(decision, adaptation)

        assert "drift_adaptation" in result.decision_metadata
        info = result.decision_metadata["drift_adaptation"]
        assert info["n_actions"] > 0

    def test_exploration_clamped_to_1(self):
        """Exploration should never exceed 1.0 after boost."""
        adapter = DriftStrategyAdapter()
        decision = _base_decision()
        decision = StrategyDecision(
            backend_name="tpe",
            stabilize_spec=StabilizeSpec(),
            exploration_strength=0.95,
            batch_size=3,
            risk_posture=RiskPosture.MODERATE,
            phase=Phase.LEARNING,
        )
        report = _mock_report(severity=0.9, detected=True)

        adaptation = adapter.adapt(report, Phase.LEARNING, 0.95)
        result = adapter.apply(decision, adaptation)

        assert result.exploration_strength <= 1.0


# ---------------------------------------------------------------------------
# Tests: End-to-end with synthetic data
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_step_change_triggers_adaptation(self):
        """Synthetic data: stable 20 rounds then KPI jumps → drift detected
        and strategy should adapt."""
        kpi = [5.0] * 20 + [50.0] * 10
        snap = _make_snapshot(kpi)

        detector = DriftDetector(reference_window=10, test_window=10)
        report = detector.detect(snap)

        assert report.drift_detected is True
        assert report.drift_score > 0.3

        adapter = DriftStrategyAdapter()
        adaptation = adapter.adapt(report, Phase.EXPLOITATION, 0.3)

        assert len(adaptation.actions) > 0
        action_types = [a.action_type for a in adaptation.actions]
        assert "exploration_boost" in action_types

    def test_stable_data_no_adaptation(self):
        """Stable KPI data should not trigger any adaptation."""
        kpi = [5.0 + 0.1 * (i % 3) for i in range(30)]
        snap = _make_snapshot(kpi)

        detector = DriftDetector(reference_window=10, test_window=10)
        report = detector.detect(snap)

        adapter = DriftStrategyAdapter()
        adaptation = adapter.adapt(report, Phase.LEARNING)

        assert len(adaptation.actions) == 0

    def test_gradual_ramp_not_sudden_reset(self):
        """Gradual drift should not trigger sudden-type phase reset."""
        # Slowly increasing KPI
        kpi = [float(i) * 0.5 for i in range(40)]
        snap = _make_snapshot(kpi)

        detector = DriftDetector(reference_window=10, test_window=10)
        report = detector.detect(snap)

        # Even if drift is detected, should not be classified as sudden.
        if report.drift_detected:
            adapter = DriftStrategyAdapter()
            adaptation = adapter.adapt(report, Phase.EXPLOITATION, 0.3)
            # Should not reset phase for mild gradual drift.
            if report.drift_score < 0.7:
                assert adaptation.adapted_phase is None
