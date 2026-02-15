"""Tests for core data models and hashing."""

from optimization_copilot.core import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    ProblemFingerprint,
    StabilizeSpec,
    StrategyDecision,
    VariableType,
    Phase,
    RiskPosture,
    snapshot_hash,
    decision_hash,
)


def _make_snapshot(n_obs: int = 5, n_failures: int = 1) -> CampaignSnapshot:
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
            kpi_values={"y": float(i) if not is_fail else 0.0},
            qc_passed=not is_fail,
            is_failure=is_fail,
            failure_reason="test_fail" if is_fail else None,
            timestamp=float(i),
        ))
    return CampaignSnapshot(
        campaign_id="test-001",
        parameter_specs=specs,
        observations=obs,
        objective_names=["y"],
        objective_directions=["maximize"],
        current_iteration=n_obs,
    )


class TestCampaignSnapshot:
    def test_creation(self):
        snap = _make_snapshot()
        assert snap.n_observations == 5
        assert snap.n_failures == 1
        assert abs(snap.failure_rate - 0.2) < 1e-9
        assert snap.parameter_names == ["x1", "x2"]

    def test_successful_observations(self):
        snap = _make_snapshot(n_obs=5, n_failures=2)
        assert len(snap.successful_observations) == 3

    def test_serialization_roundtrip(self):
        snap = _make_snapshot()
        d = snap.to_dict()
        restored = CampaignSnapshot.from_dict(d)
        assert restored.campaign_id == snap.campaign_id
        assert restored.n_observations == snap.n_observations
        assert restored.parameter_specs[0].name == "x1"

    def test_empty_snapshot(self):
        snap = CampaignSnapshot(
            campaign_id="empty",
            parameter_specs=[],
            observations=[],
            objective_names=["y"],
            objective_directions=["minimize"],
        )
        assert snap.n_observations == 0
        assert snap.failure_rate == 0.0


class TestStrategyDecision:
    def test_creation(self):
        dec = StrategyDecision(
            backend_name="random",
            stabilize_spec=StabilizeSpec(),
            exploration_strength=1.0,
            phase=Phase.COLD_START,
            reason_codes=["cold_start_default"],
        )
        assert dec.backend_name == "random"
        assert dec.risk_posture == RiskPosture.MODERATE

    def test_to_dict(self):
        dec = StrategyDecision(
            backend_name="tpe",
            stabilize_spec=StabilizeSpec(),
            exploration_strength=0.5,
        )
        d = dec.to_dict()
        assert d["backend_name"] == "tpe"
        assert d["risk_posture"] == "moderate"


class TestProblemFingerprint:
    def test_defaults(self):
        fp = ProblemFingerprint()
        assert fp.variable_types == VariableType.CONTINUOUS
        d = fp.to_dict()
        assert d["variable_types"] == "continuous"

    def test_to_tuple(self):
        fp = ProblemFingerprint()
        t = fp.to_tuple()
        assert len(t) == 11


class TestHashing:
    def test_snapshot_hash_deterministic(self):
        snap = _make_snapshot()
        h1 = snapshot_hash(snap)
        h2 = snapshot_hash(snap)
        assert h1 == h2
        assert len(h1) == 16

    def test_different_snapshots_different_hash(self):
        s1 = _make_snapshot(n_obs=3)
        s2 = _make_snapshot(n_obs=5)
        assert snapshot_hash(s1) != snapshot_hash(s2)

    def test_decision_hash_deterministic(self):
        dec = StrategyDecision(
            backend_name="cma_es",
            stabilize_spec=StabilizeSpec(),
            exploration_strength=0.3,
        )
        h1 = decision_hash(dec)
        h2 = decision_hash(dec)
        assert h1 == h2
