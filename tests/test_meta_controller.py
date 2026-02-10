"""Tests for the Meta-Controller."""

from optimization_copilot.core import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    Phase,
    ProblemFingerprint,
    RiskPosture,
    VariableType,
    DataScale,
    NoiseRegime,
)
from optimization_copilot.meta_controller.controller import MetaController


def _make_snapshot(n_obs: int = 5, n_failures: int = 0) -> CampaignSnapshot:
    specs = [
        ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
        ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=-1.0, upper=1.0),
    ]
    obs = []
    for i in range(n_obs):
        is_fail = i < n_failures
        obs.append(Observation(
            iteration=i,
            parameters={"x1": i * 0.1, "x2": i * -0.1},
            kpi_values={"y": float(i + 1) if not is_fail else 0.0},
            is_failure=is_fail,
            timestamp=float(i),
        ))
    return CampaignSnapshot(
        campaign_id="test",
        parameter_specs=specs,
        observations=obs,
        objective_names=["y"],
        objective_directions=["maximize"],
        current_iteration=n_obs,
    )


def _base_diagnostics(**overrides) -> dict:
    d = {
        "convergence_trend": 0.1,
        "improvement_velocity": 0.1,
        "variance_contraction": 0.8,
        "noise_estimate": 0.1,
        "failure_rate": 0.0,
        "failure_clustering": 0.0,
        "feasibility_shrinkage": 0.0,
        "parameter_drift": 0.05,
        "model_uncertainty": 0.5,
        "exploration_coverage": 0.3,
        "kpi_plateau_length": 0,
        "best_kpi_value": 5.0,
        "data_efficiency": 0.5,
        "constraint_violation_rate": 0.0,
    }
    d.update(overrides)
    return d


class TestPhaseDetection:
    def test_cold_start_few_observations(self):
        mc = MetaController()
        snap = _make_snapshot(n_obs=3)
        diag = _base_diagnostics()
        fp = ProblemFingerprint()
        decision = mc.decide(snap, diag, fp)
        assert decision.phase == Phase.COLD_START

    def test_learning_with_enough_data(self):
        mc = MetaController()
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()
        fp = ProblemFingerprint()
        decision = mc.decide(snap, diag, fp)
        assert decision.phase == Phase.LEARNING

    def test_stagnation_on_plateau(self):
        mc = MetaController()
        snap = _make_snapshot(n_obs=20)
        diag = _base_diagnostics(kpi_plateau_length=15)
        fp = ProblemFingerprint()
        decision = mc.decide(snap, diag, fp)
        assert decision.phase == Phase.STAGNATION

    def test_stagnation_on_failure_spike(self):
        mc = MetaController()
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics(failure_clustering=0.8)
        fp = ProblemFingerprint()
        decision = mc.decide(snap, diag, fp)
        assert decision.phase == Phase.STAGNATION

    def test_exploitation_on_convergence(self):
        mc = MetaController()
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics(convergence_trend=0.5, model_uncertainty=0.1)
        fp = ProblemFingerprint()
        decision = mc.decide(snap, diag, fp)
        assert decision.phase == Phase.EXPLOITATION


class TestBackendSelection:
    def test_cold_start_selects_lhs(self):
        mc = MetaController()
        snap = _make_snapshot(n_obs=3)
        diag = _base_diagnostics()
        fp = ProblemFingerprint()
        decision = mc.decide(snap, diag, fp)
        assert decision.backend_name in ("latin_hypercube", "random")

    def test_fallback_when_preferred_unavailable(self):
        mc = MetaController(available_backends=["my_custom"])
        snap = _make_snapshot(n_obs=3)
        diag = _base_diagnostics()
        fp = ProblemFingerprint()
        decision = mc.decide(snap, diag, fp)
        assert decision.backend_name == "my_custom"
        assert any("no_preferred" in e for e in decision.fallback_events)


class TestDeterminism:
    def test_same_input_same_output(self):
        mc = MetaController()
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()
        fp = ProblemFingerprint()
        d1 = mc.decide(snap, diag, fp, seed=42)
        d2 = mc.decide(snap, diag, fp, seed=42)
        assert d1.backend_name == d2.backend_name
        assert d1.phase == d2.phase
        assert d1.exploration_strength == d2.exploration_strength
        assert d1.reason_codes == d2.reason_codes


class TestRiskPosture:
    def test_conservative_in_cold_start(self):
        mc = MetaController()
        snap = _make_snapshot(n_obs=3)
        diag = _base_diagnostics()
        fp = ProblemFingerprint()
        decision = mc.decide(snap, diag, fp)
        assert decision.risk_posture == RiskPosture.CONSERVATIVE

    def test_conservative_on_high_failures(self):
        mc = MetaController()
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics(failure_rate=0.5)
        fp = ProblemFingerprint()
        decision = mc.decide(snap, diag, fp)
        assert decision.risk_posture == RiskPosture.CONSERVATIVE


class TestExplorationStrength:
    def test_high_exploration_in_cold_start(self):
        mc = MetaController()
        snap = _make_snapshot(n_obs=3)
        diag = _base_diagnostics()
        fp = ProblemFingerprint()
        decision = mc.decide(snap, diag, fp)
        assert decision.exploration_strength >= 0.8

    def test_low_exploration_in_exploitation(self):
        mc = MetaController()
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics(convergence_trend=0.5, model_uncertainty=0.1)
        fp = ProblemFingerprint()
        decision = mc.decide(snap, diag, fp)
        assert decision.exploration_strength <= 0.4


class TestAuditTrail:
    def test_reason_codes_populated(self):
        mc = MetaController()
        snap = _make_snapshot(n_obs=15)
        diag = _base_diagnostics()
        fp = ProblemFingerprint()
        decision = mc.decide(snap, diag, fp)
        assert len(decision.reason_codes) > 0

    def test_metadata_includes_seed(self):
        mc = MetaController()
        snap = _make_snapshot(n_obs=3)
        diag = _base_diagnostics()
        fp = ProblemFingerprint()
        decision = mc.decide(snap, diag, fp, seed=99)
        assert decision.decision_metadata["seed"] == 99
