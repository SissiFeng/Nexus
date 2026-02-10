"""Integration tests: end-to-end pipeline and golden scenario validation."""

import random

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    Phase,
    ProblemFingerprint,
    RiskPosture,
    VariableType,
)
from optimization_copilot.core.hashing import snapshot_hash, decision_hash, diagnostics_hash
from optimization_copilot.diagnostics.engine import DiagnosticEngine
from optimization_copilot.profiler.profiler import ProblemProfiler
from optimization_copilot.meta_controller.controller import MetaController
from optimization_copilot.stabilization.stabilizer import Stabilizer
from optimization_copilot.screening.screener import VariableScreener
from optimization_copilot.feasibility.feasibility import FeasibilityLearner
from optimization_copilot.multi_objective.pareto import MultiObjectiveAnalyzer
from optimization_copilot.explainability.explainer import DecisionExplainer
from optimization_copilot.validation.scenarios import (
    ValidationRunner,
    GOLDEN_SCENARIOS,
)


# ── Helpers ──────────────────────────────────────────────


def _build_snapshot(
    n_obs: int = 20,
    n_params: int = 3,
    n_objectives: int = 1,
    failure_rate: float = 0.0,
    seed: int = 42,
) -> CampaignSnapshot:
    """Build a deterministic synthetic campaign snapshot."""
    rng = random.Random(seed)
    specs = [
        ParameterSpec(name=f"x{i}", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0)
        for i in range(n_params)
    ]
    obj_names = [f"y{i}" for i in range(n_objectives)]
    obj_dirs = ["maximize"] * n_objectives

    obs = []
    for i in range(n_obs):
        params = {f"x{j}": rng.random() for j in range(n_params)}
        is_fail = rng.random() < failure_rate
        kpis = {}
        for oi, name in enumerate(obj_names):
            kpis[name] = sum(params.values()) + rng.gauss(0, 0.1) + i * 0.1 if not is_fail else 0.0
        obs.append(Observation(
            iteration=i,
            parameters=params,
            kpi_values=kpis,
            is_failure=is_fail,
            timestamp=float(i),
        ))

    return CampaignSnapshot(
        campaign_id="integration_test",
        parameter_specs=specs,
        observations=obs,
        objective_names=obj_names,
        objective_directions=obj_dirs,
        current_iteration=n_obs,
    )


# ── End-to-End Pipeline Tests ────────────────────────────


class TestEndToEndPipeline:
    """Full pipeline: Snapshot → Diagnostics → Profiler → Controller → Explainer."""

    def test_full_pipeline_runs_without_error(self):
        snap = _build_snapshot(n_obs=20)
        engine = DiagnosticEngine()
        profiler = ProblemProfiler()
        controller = MetaController()
        explainer = DecisionExplainer()

        diag = engine.compute(snap)
        fp = profiler.profile(snap)
        decision = controller.decide(snap, diag.to_dict(), fp, seed=42)
        report = explainer.explain(decision, fp, diag.to_dict())

        assert decision.backend_name
        assert decision.phase in Phase
        assert 0.0 <= decision.exploration_strength <= 1.0
        assert decision.batch_size >= 1
        assert report.summary
        assert len(decision.reason_codes) > 0

    def test_full_pipeline_with_failures(self):
        snap = _build_snapshot(n_obs=25, failure_rate=0.4)
        engine = DiagnosticEngine()
        profiler = ProblemProfiler()
        controller = MetaController()
        explainer = DecisionExplainer()

        diag = engine.compute(snap)
        fp = profiler.profile(snap)
        decision = controller.decide(snap, diag.to_dict(), fp, seed=42)
        report = explainer.explain(decision, fp, diag.to_dict())

        # High failure rate should push toward conservative
        assert decision.risk_posture in (RiskPosture.CONSERVATIVE, RiskPosture.MODERATE)
        assert report.summary

    def test_full_pipeline_cold_start(self):
        snap = _build_snapshot(n_obs=3)
        engine = DiagnosticEngine()
        profiler = ProblemProfiler()
        controller = MetaController()
        explainer = DecisionExplainer()

        diag = engine.compute(snap)
        fp = profiler.profile(snap)
        decision = controller.decide(snap, diag.to_dict(), fp, seed=42)
        report = explainer.explain(decision, fp, diag.to_dict())

        assert decision.phase == Phase.COLD_START
        assert decision.exploration_strength >= 0.8
        assert decision.risk_posture == RiskPosture.CONSERVATIVE

    def test_pipeline_with_stabilization(self):
        snap = _build_snapshot(n_obs=20, failure_rate=0.2)
        engine = DiagnosticEngine()
        profiler = ProblemProfiler()
        controller = MetaController()
        stabilizer = Stabilizer()

        diag = engine.compute(snap)
        fp = profiler.profile(snap)
        decision = controller.decide(snap, diag.to_dict(), fp)

        # Stabilize with the decision's spec
        stable = stabilizer.stabilize(snap, decision.stabilize_spec)
        assert len(stable.observations) >= 0
        assert isinstance(stable.applied_policies, list)

    def test_pipeline_with_screening(self):
        snap = _build_snapshot(n_obs=20, n_params=5)
        screener = VariableScreener()
        result = screener.screen(snap, seed=42)

        assert len(result.ranked_parameters) == 5
        assert all(0.0 <= v <= 1.0 for v in result.importance_scores.values())
        assert len(result.recommended_step_sizes) == 5

    def test_pipeline_with_feasibility(self):
        snap = _build_snapshot(n_obs=20, failure_rate=0.3)
        learner = FeasibilityLearner()
        fmap = learner.learn(snap)

        assert 0.0 <= fmap.feasibility_score <= 1.0
        assert len(fmap.safe_bounds) > 0

        # Check that feasibility checker works
        test_params = {"x0": 0.5, "x1": 0.5, "x2": 0.5}
        result = learner.is_feasible(test_params, fmap)
        assert isinstance(result, bool)

    def test_pipeline_multi_objective(self):
        snap = _build_snapshot(n_obs=15, n_objectives=2)
        analyzer = MultiObjectiveAnalyzer()
        result = analyzer.analyze(snap)

        assert len(result.pareto_front) > 0
        assert len(result.dominance_ranks) > 0
        assert all(r >= 1 for r in result.dominance_ranks)


# ── Determinism Tests ────────────────────────────────────


class TestDeterminism:
    """Verify identical input → identical output across runs."""

    def test_pipeline_determinism(self):
        snap = _build_snapshot(n_obs=20)
        engine = DiagnosticEngine()
        profiler = ProblemProfiler()
        controller = MetaController()

        results = []
        for _ in range(3):
            diag = engine.compute(snap)
            fp = profiler.profile(snap)
            decision = controller.decide(snap, diag.to_dict(), fp, seed=42)
            results.append(decision)

        # All runs must produce identical decisions
        for r in results[1:]:
            assert r.backend_name == results[0].backend_name
            assert r.phase == results[0].phase
            assert r.exploration_strength == results[0].exploration_strength
            assert r.risk_posture == results[0].risk_posture
            assert r.batch_size == results[0].batch_size

    def test_hashing_determinism(self):
        snap = _build_snapshot(n_obs=20)
        h1 = snapshot_hash(snap)
        h2 = snapshot_hash(snap)
        assert h1 == h2
        assert len(h1) == 16

    def test_diagnostics_hash_determinism(self):
        snap = _build_snapshot(n_obs=20)
        engine = DiagnosticEngine()
        d1 = engine.compute(snap)
        d2 = engine.compute(snap)
        h1 = diagnostics_hash(d1.to_dict())
        h2 = diagnostics_hash(d2.to_dict())
        assert h1 == h2

    def test_decision_hash_determinism(self):
        snap = _build_snapshot(n_obs=20)
        engine = DiagnosticEngine()
        profiler = ProblemProfiler()
        controller = MetaController()

        diag = engine.compute(snap)
        fp = profiler.profile(snap)
        d1 = controller.decide(snap, diag.to_dict(), fp, seed=42)
        d2 = controller.decide(snap, diag.to_dict(), fp, seed=42)
        assert decision_hash(d1) == decision_hash(d2)


# ── Golden Scenario Tests ────────────────────────────────


class TestGoldenScenarios:
    """Run all golden scenarios and verify expectations."""

    def test_all_golden_scenarios_pass(self):
        runner = ValidationRunner()
        results = runner.run_all()
        for r in results:
            assert r.passed, f"Scenario '{r.scenario_name}' failed: {r.failures}"

    def test_clean_convergence_scenario(self):
        runner = ValidationRunner()
        scenario = GOLDEN_SCENARIOS[0]
        assert scenario.name == "clean_convergence"
        result = runner.run_scenario(scenario)
        assert result.passed, f"Failures: {result.failures}"
        assert result.decision.phase == Phase.EXPLOITATION

    def test_cold_start_scenario(self):
        runner = ValidationRunner()
        scenario = GOLDEN_SCENARIOS[1]
        assert scenario.name == "cold_start"
        result = runner.run_scenario(scenario)
        assert result.passed, f"Failures: {result.failures}"
        assert result.decision.phase == Phase.COLD_START

    def test_failure_heavy_scenario(self):
        runner = ValidationRunner()
        scenario = GOLDEN_SCENARIOS[2]
        assert scenario.name == "failure_heavy"
        result = runner.run_scenario(scenario)
        assert result.passed, f"Failures: {result.failures}"

    def test_scenario_determinism(self):
        runner = ValidationRunner()
        for scenario in GOLDEN_SCENARIOS:
            assert runner.verify_determinism(scenario, n_runs=3), (
                f"Scenario '{scenario.name}' is non-deterministic"
            )

    def test_scenario_hashes_populated(self):
        runner = ValidationRunner()
        for scenario in GOLDEN_SCENARIOS:
            result = runner.run_scenario(scenario)
            assert result.snapshot_hash, f"No snapshot hash for {scenario.name}"
            assert result.decision_hash, f"No decision hash for {scenario.name}"
            assert len(result.snapshot_hash) == 16
            assert len(result.decision_hash) == 16


# ── Audit Trail Tests ────────────────────────────────────


class TestAuditTrail:
    """Verify that the full pipeline produces a complete audit trail."""

    def test_decision_has_reason_codes(self):
        snap = _build_snapshot(n_obs=20)
        engine = DiagnosticEngine()
        profiler = ProblemProfiler()
        controller = MetaController()

        diag = engine.compute(snap)
        fp = profiler.profile(snap)
        decision = controller.decide(snap, diag.to_dict(), fp, seed=42)

        assert len(decision.reason_codes) > 0
        assert decision.decision_metadata.get("seed") == 42
        assert decision.decision_metadata.get("n_observations") == 20

    def test_explainability_report_complete(self):
        snap = _build_snapshot(n_obs=20)
        engine = DiagnosticEngine()
        profiler = ProblemProfiler()
        controller = MetaController()
        explainer = DecisionExplainer()

        diag = engine.compute(snap)
        fp = profiler.profile(snap)
        decision = controller.decide(snap, diag.to_dict(), fp)
        report = explainer.explain(decision, fp, diag.to_dict())

        assert report.summary
        assert report.selected_strategy
        assert report.risk_assessment
        assert report.coverage_status
        assert report.remaining_uncertainty
        assert report.details["backend"] == decision.backend_name
        assert report.details["phase"] == decision.phase.value

    def test_phase_transition_reported(self):
        snap = _build_snapshot(n_obs=20)
        engine = DiagnosticEngine()
        profiler = ProblemProfiler()
        controller = MetaController()
        explainer = DecisionExplainer()

        diag = engine.compute(snap)
        fp = profiler.profile(snap)
        decision = controller.decide(snap, diag.to_dict(), fp)

        # Simulate transition from cold start
        report = explainer.explain(
            decision, fp, diag.to_dict(), previous_phase=Phase.COLD_START
        )
        if decision.phase != Phase.COLD_START:
            assert report.phase_transition is not None


# ── Cross-Module Consistency Tests ───────────────────────


class TestCrossModuleConsistency:
    """Verify modules are consistent with each other."""

    def test_profiler_fingerprint_feeds_controller(self):
        snap = _build_snapshot(n_obs=20)
        profiler = ProblemProfiler()
        fp = profiler.profile(snap)

        engine = DiagnosticEngine()
        diag = engine.compute(snap)

        controller = MetaController()
        decision = controller.decide(snap, diag.to_dict(), fp, seed=42)

        # Fingerprint should be consistent with snapshot
        assert fp.data_scale.value in ("tiny", "small", "moderate")
        assert decision.backend_name in controller.available_backends

    def test_stabilizer_preserves_observation_count(self):
        snap = _build_snapshot(n_obs=20, failure_rate=0.0)
        stabilizer = Stabilizer()
        engine = DiagnosticEngine()
        profiler = ProblemProfiler()
        controller = MetaController()

        diag = engine.compute(snap)
        fp = profiler.profile(snap)
        decision = controller.decide(snap, diag.to_dict(), fp)

        stable = stabilizer.stabilize(snap, decision.stabilize_spec)
        # With penalize policy and no outliers, all obs should be preserved
        if decision.stabilize_spec.failure_handling == "penalize":
            assert len(stable.observations) == snap.n_observations

    def test_feasibility_consistent_with_diagnostics(self):
        snap = _build_snapshot(n_obs=20, failure_rate=0.3)
        engine = DiagnosticEngine()
        learner = FeasibilityLearner()

        diag = engine.compute(snap)
        fmap = learner.learn(snap)

        # Both should agree on approximate failure rate
        assert abs(diag.failure_rate - snap.failure_rate) < 0.01
        assert abs(fmap.feasibility_score - (1.0 - snap.failure_rate)) < 0.01
