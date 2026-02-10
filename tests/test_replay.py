"""Tests for the deterministic replay engine: log, entry, recording, verification.

Covers:
- DecisionLogEntry creation and serialization round-trips
- DecisionLog aggregation, queries, JSON round-trip, file I/O
- ReplayEngine.record_iteration() static helper
- ReplayEngine.verify() for deterministic replay verification
- Determinism (verify same log multiple times)
- Tampering detection (modified hashes cause mismatch)
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any

from optimization_copilot.backends.builtin import (
    LatinHypercubeSampler,
    RandomSampler,
    TPESampler,
)
from optimization_copilot.core.hashing import (
    decision_hash as compute_decision_hash,
    diagnostics_hash as compute_diagnostics_hash,
    snapshot_hash as compute_snapshot_hash,
)
from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    Phase,
    ProblemFingerprint,
    RiskPosture,
    StabilizeSpec,
    StrategyDecision,
    VariableType,
)
from optimization_copilot.diagnostics.engine import DiagnosticEngine
from optimization_copilot.dsl.spec import (
    BudgetDef,
    Direction,
    ObjectiveDef,
    OptimizationSpec,
    ParamType,
    ParameterDef,
)
from optimization_copilot.plugins.registry import PluginRegistry
from optimization_copilot.profiler.profiler import ProblemProfiler
from optimization_copilot.replay.engine import ReplayEngine
from optimization_copilot.replay.log import DecisionLog, DecisionLogEntry
from optimization_copilot.replay.report import ReplayVerification


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec() -> OptimizationSpec:
    """Simple OptimizationSpec: 2 continuous params (x1, x2: 0-10), maximize y, budget 10."""
    return OptimizationSpec(
        campaign_id="test-replay-campaign",
        parameters=[
            ParameterDef(name="x1", type=ParamType.CONTINUOUS, lower=0.0, upper=10.0),
            ParameterDef(name="x2", type=ParamType.CONTINUOUS, lower=0.0, upper=10.0),
        ],
        objectives=[
            ObjectiveDef(name="y", direction=Direction.MAXIMIZE),
        ],
        budget=BudgetDef(max_samples=10),
        seed=42,
    )


def _make_registry() -> PluginRegistry:
    """PluginRegistry with RandomSampler, LatinHypercubeSampler, TPESampler."""
    registry = PluginRegistry()
    registry.register(RandomSampler)
    registry.register(LatinHypercubeSampler)
    registry.register(TPESampler)
    return registry


def _simple_evaluator(params: dict[str, Any]) -> dict[str, float]:
    """Evaluator: y = x1 + x2."""
    return {"y": params["x1"] + params["x2"]}


def _run_and_get_log(
    spec: OptimizationSpec,
    registry: PluginRegistry,
    evaluator: Any,
) -> DecisionLog:
    """Build a DecisionLog by running the optimization pipeline step-by-step.

    Uses the same seed formula as ReplayEngine.verify() (base_seed + iteration)
    so that the resulting log can be verified deterministically.
    """
    from optimization_copilot.meta_controller.controller import MetaController

    diag_engine = DiagnosticEngine()
    profiler = ProblemProfiler()
    controller = MetaController(available_backends=registry.list_plugins())

    log = DecisionLog(
        campaign_id=spec.campaign_id,
        spec=spec.to_dict(),
        base_seed=spec.seed,
        metadata={"source": "test"},
    )

    param_specs = [
        ParameterSpec(
            name=p.name,
            type=VariableType(p.type.value),
            lower=p.lower,
            upper=p.upper,
            categories=p.categories,
        )
        for p in spec.parameters
        if not p.frozen
    ]
    obj_names = [o.name for o in spec.objectives]
    obj_dirs = [o.direction.value for o in spec.objectives]

    accumulated_obs: list[Observation] = []
    previous_phase: Phase | None = None
    max_samples = spec.budget.max_samples or 100
    iteration = 0

    while len(accumulated_obs) < max_samples and iteration < 100:
        # Use the SAME seed formula as ReplayEngine.verify().
        seed = spec.seed + iteration

        snapshot = CampaignSnapshot(
            campaign_id=spec.campaign_id,
            parameter_specs=list(param_specs),
            observations=list(accumulated_obs),
            objective_names=list(obj_names),
            objective_directions=list(obj_dirs),
            current_iteration=iteration,
            metadata={},
        )

        diag_vector = diag_engine.compute(snapshot)
        diag_dict = diag_vector.to_dict()
        fingerprint = profiler.profile(snapshot)

        decision = controller.decide(
            snapshot=snapshot,
            diagnostics=diag_dict,
            fingerprint=fingerprint,
            seed=seed,
            previous_phase=previous_phase,
        )

        plugin = registry.get(decision.backend_name)
        plugin.fit(snapshot.observations, snapshot.parameter_specs)
        candidates = plugin.suggest(n_suggestions=decision.batch_size, seed=seed)

        results: list[dict[str, Any]] = []
        for cand in candidates:
            try:
                kpi_values = evaluator(cand)
                results.append({
                    "iteration": iteration,
                    "parameters": dict(cand),
                    "kpi_values": dict(kpi_values),
                    "qc_passed": True,
                    "is_failure": False,
                    "failure_reason": None,
                    "timestamp": time.time(),
                    "metadata": {},
                })
            except Exception as exc:
                results.append({
                    "iteration": iteration,
                    "parameters": dict(cand),
                    "kpi_values": {},
                    "qc_passed": False,
                    "is_failure": True,
                    "failure_reason": str(exc),
                    "timestamp": time.time(),
                    "metadata": {},
                })

        entry = ReplayEngine.record_iteration(
            iteration=iteration,
            snapshot=snapshot,
            diagnostics_vector=diag_dict,
            fingerprint_dict=fingerprint.to_dict(),
            decision=decision,
            candidates=candidates,
            results=results,
            seed=seed,
        )
        log.append(entry)
        previous_phase = decision.phase

        for r in results:
            accumulated_obs.append(
                Observation(
                    iteration=r["iteration"],
                    parameters=r["parameters"],
                    kpi_values=r["kpi_values"],
                    qc_passed=r["qc_passed"],
                    is_failure=r["is_failure"],
                    failure_reason=r["failure_reason"],
                    timestamp=r.get("timestamp", 0.0),
                    metadata=r.get("metadata", {}),
                )
            )

        iteration += 1

    return log


def _make_sample_entry(iteration: int = 0, seed: int = 42) -> DecisionLogEntry:
    """Create a sample DecisionLogEntry with all fields populated."""
    return DecisionLogEntry(
        iteration=iteration,
        timestamp=1000000.0 + iteration,
        snapshot_hash="abcdef0123456789",
        diagnostics_hash="1234567890abcdef",
        diagnostics={"convergence_trend": 0.5, "best_kpi_value": 3.14},
        fingerprint={"variable_types": "continuous", "objective_form": "single"},
        decision={
            "backend_name": "random_sampler",
            "phase": "cold_start",
            "exploration_strength": 0.9,
        },
        decision_hash="fedcba9876543210",
        suggested_candidates=[{"x1": 1.0, "x2": 2.0}],
        ingested_results=[
            {
                "iteration": iteration,
                "parameters": {"x1": 1.0, "x2": 2.0},
                "kpi_values": {"y": 3.0},
                "qc_passed": True,
                "is_failure": False,
            }
        ],
        phase="cold_start",
        backend_name="random_sampler",
        reason_codes=["cold_start_default"],
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Test: DecisionLogEntry
# ---------------------------------------------------------------------------


class TestDecisionLogEntry:
    """Tests for DecisionLogEntry creation and serialization."""

    def test_creation_all_fields(self):
        entry = _make_sample_entry(iteration=3, seed=99)

        assert entry.iteration == 3
        assert entry.seed == 99
        assert entry.timestamp == 1000003.0
        assert entry.snapshot_hash == "abcdef0123456789"
        assert entry.diagnostics_hash == "1234567890abcdef"
        assert entry.decision_hash == "fedcba9876543210"
        assert entry.phase == "cold_start"
        assert entry.backend_name == "random_sampler"
        assert "convergence_trend" in entry.diagnostics
        assert len(entry.suggested_candidates) == 1
        assert len(entry.ingested_results) == 1
        assert entry.reason_codes == ["cold_start_default"]

    def test_to_dict_round_trip(self):
        original = _make_sample_entry(iteration=5, seed=123)
        d = original.to_dict()

        # Verify dict has all expected keys.
        expected_keys = {
            "iteration", "timestamp", "snapshot_hash", "diagnostics_hash",
            "diagnostics", "fingerprint", "decision", "decision_hash",
            "suggested_candidates", "ingested_results", "phase",
            "backend_name", "reason_codes", "seed",
        }
        assert set(d.keys()) == expected_keys

        # Round-trip.
        restored = DecisionLogEntry.from_dict(d)
        assert restored.iteration == original.iteration
        assert restored.seed == original.seed
        assert restored.snapshot_hash == original.snapshot_hash
        assert restored.diagnostics_hash == original.diagnostics_hash
        assert restored.decision_hash == original.decision_hash
        assert restored.diagnostics == original.diagnostics
        assert restored.fingerprint == original.fingerprint
        assert restored.decision == original.decision
        assert restored.suggested_candidates == original.suggested_candidates
        assert restored.ingested_results == original.ingested_results
        assert restored.phase == original.phase
        assert restored.backend_name == original.backend_name
        assert restored.reason_codes == original.reason_codes

    def test_from_dict_to_dict_idempotent(self):
        original = _make_sample_entry()
        d1 = original.to_dict()
        restored = DecisionLogEntry.from_dict(d1)
        d2 = restored.to_dict()
        assert d1 == d2


# ---------------------------------------------------------------------------
# Test: DecisionLog
# ---------------------------------------------------------------------------


class TestDecisionLog:
    """Tests for DecisionLog aggregation and serialization."""

    def test_append_entries(self):
        log = DecisionLog(
            campaign_id="test-001",
            spec={"parameters": [], "objectives": []},
            base_seed=42,
        )
        for i in range(5):
            log.append(_make_sample_entry(iteration=i))

        assert log.n_entries == 5
        assert len(log.entries) == 5

    def test_get_entry_by_iteration(self):
        log = DecisionLog(
            campaign_id="test-002",
            spec={"parameters": [], "objectives": []},
            base_seed=42,
        )
        for i in range(5):
            log.append(_make_sample_entry(iteration=i))

        entry = log.get_entry(3)
        assert entry is not None
        assert entry.iteration == 3

        # Non-existent iteration.
        missing = log.get_entry(99)
        assert missing is None

    def test_n_entries_property(self):
        log = DecisionLog(
            campaign_id="test-003",
            spec={},
            base_seed=42,
        )
        assert log.n_entries == 0
        log.append(_make_sample_entry(iteration=0))
        assert log.n_entries == 1
        log.append(_make_sample_entry(iteration=1))
        assert log.n_entries == 2

    def test_phase_transitions(self):
        log = DecisionLog(
            campaign_id="test-004",
            spec={},
            base_seed=42,
        )
        # Create entries with phase transitions.
        phases = ["cold_start", "cold_start", "learning", "learning", "exploitation"]
        for i, phase in enumerate(phases):
            entry = _make_sample_entry(iteration=i)
            # Override the phase.
            entry = DecisionLogEntry(
                iteration=i,
                timestamp=entry.timestamp,
                snapshot_hash=entry.snapshot_hash,
                diagnostics_hash=entry.diagnostics_hash,
                diagnostics=entry.diagnostics,
                fingerprint=entry.fingerprint,
                decision=entry.decision,
                decision_hash=entry.decision_hash,
                suggested_candidates=entry.suggested_candidates,
                ingested_results=entry.ingested_results,
                phase=phase,
                backend_name=entry.backend_name,
                reason_codes=entry.reason_codes,
                seed=entry.seed,
            )
            log.append(entry)

        transitions = log.phase_transitions
        assert len(transitions) == 2
        # First transition: cold_start -> learning at iteration 2.
        assert transitions[0] == (2, "cold_start", "learning")
        # Second transition: learning -> exploitation at iteration 4.
        assert transitions[1] == (4, "learning", "exploitation")

    def test_phase_transitions_no_changes(self):
        log = DecisionLog(campaign_id="t", spec={}, base_seed=0)
        for i in range(3):
            entry = _make_sample_entry(iteration=i)
            log.append(entry)
        assert log.phase_transitions == []

    def test_json_round_trip(self):
        log = DecisionLog(
            campaign_id="test-005",
            spec={"parameters": [{"name": "x1"}], "objectives": [{"name": "y"}]},
            base_seed=42,
            metadata={"source": "test"},
        )
        for i in range(3):
            log.append(_make_sample_entry(iteration=i))

        json_str = log.to_json()
        restored = DecisionLog.from_json(json_str)

        assert restored.campaign_id == log.campaign_id
        assert restored.base_seed == log.base_seed
        assert restored.n_entries == log.n_entries
        assert restored.metadata == log.metadata

        # Verify entries match.
        for i in range(log.n_entries):
            orig = log.entries[i]
            rest = restored.entries[i]
            assert orig.iteration == rest.iteration
            assert orig.snapshot_hash == rest.snapshot_hash
            assert orig.decision_hash == rest.decision_hash

    def test_save_load_file(self):
        log = DecisionLog(
            campaign_id="test-006",
            spec={"parameters": [], "objectives": []},
            base_seed=42,
        )
        for i in range(3):
            log.append(_make_sample_entry(iteration=i))

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir="/tmp"
        ) as f:
            tmp_path = f.name

        try:
            log.save(tmp_path)
            assert os.path.exists(tmp_path)

            loaded = DecisionLog.load(tmp_path)
            assert loaded.campaign_id == log.campaign_id
            assert loaded.base_seed == log.base_seed
            assert loaded.n_entries == log.n_entries

            for i in range(log.n_entries):
                assert loaded.entries[i].iteration == log.entries[i].iteration
                assert loaded.entries[i].decision_hash == log.entries[i].decision_hash
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Test: ReplayEngine.record_iteration
# ---------------------------------------------------------------------------


class TestRecordIteration:
    """Tests for ReplayEngine.record_iteration() static method."""

    def test_record_iteration_populates_hashes(self):
        snapshot = CampaignSnapshot(
            campaign_id="test-record",
            parameter_specs=[
                ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
                ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
            ],
            observations=[],
            objective_names=["y"],
            objective_directions=["maximize"],
            current_iteration=0,
        )
        diagnostics_vector = {
            "convergence_trend": 0.0,
            "best_kpi_value": 0.0,
            "improvement_velocity": 0.0,
        }
        fingerprint_dict = {
            "variable_types": "continuous",
            "objective_form": "single",
        }
        decision = StrategyDecision(
            backend_name="random_sampler",
            stabilize_spec=StabilizeSpec(),
            exploration_strength=0.9,
            batch_size=2,
            phase=Phase.COLD_START,
            reason_codes=["cold_start_default"],
        )
        candidates = [{"x1": 1.0, "x2": 2.0}, {"x1": 3.0, "x2": 4.0}]
        results = [
            {
                "iteration": 0,
                "parameters": {"x1": 1.0, "x2": 2.0},
                "kpi_values": {"y": 3.0},
            },
            {
                "iteration": 0,
                "parameters": {"x1": 3.0, "x2": 4.0},
                "kpi_values": {"y": 7.0},
            },
        ]

        entry = ReplayEngine.record_iteration(
            iteration=0,
            snapshot=snapshot,
            diagnostics_vector=diagnostics_vector,
            fingerprint_dict=fingerprint_dict,
            decision=decision,
            candidates=candidates,
            results=results,
            seed=42,
        )

        # All hash fields must be non-empty strings.
        assert isinstance(entry.snapshot_hash, str) and len(entry.snapshot_hash) > 0
        assert isinstance(entry.diagnostics_hash, str) and len(entry.diagnostics_hash) > 0
        assert isinstance(entry.decision_hash, str) and len(entry.decision_hash) > 0

        # Verify hashes are deterministic.
        expected_snap_hash = compute_snapshot_hash(snapshot)
        expected_diag_hash = compute_diagnostics_hash(diagnostics_vector)
        expected_dec_hash = compute_decision_hash(decision)
        assert entry.snapshot_hash == expected_snap_hash
        assert entry.diagnostics_hash == expected_diag_hash
        assert entry.decision_hash == expected_dec_hash

        # Verify other fields.
        assert entry.iteration == 0
        assert entry.seed == 42
        assert entry.phase == "cold_start"
        assert entry.backend_name == "random_sampler"
        assert len(entry.suggested_candidates) == 2
        assert len(entry.ingested_results) == 2
        assert entry.reason_codes == ["cold_start_default"]
        assert entry.timestamp > 0

    def test_record_iteration_deterministic(self):
        """Same inputs produce same hash values."""
        snapshot = CampaignSnapshot(
            campaign_id="det-test",
            parameter_specs=[
                ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=5.0),
            ],
            observations=[],
            objective_names=["y"],
            objective_directions=["maximize"],
        )
        decision = StrategyDecision(
            backend_name="random_sampler",
            stabilize_spec=StabilizeSpec(),
            exploration_strength=0.9,
            phase=Phase.COLD_START,
        )
        diag = {"convergence_trend": 0.0}
        fp = {"variable_types": "continuous"}
        cands = [{"x1": 2.5}]
        results = [{"parameters": {"x1": 2.5}, "kpi_values": {"y": 2.5}}]

        entry1 = ReplayEngine.record_iteration(
            iteration=0, snapshot=snapshot, diagnostics_vector=diag,
            fingerprint_dict=fp, decision=decision,
            candidates=cands, results=results, seed=42,
        )
        entry2 = ReplayEngine.record_iteration(
            iteration=0, snapshot=snapshot, diagnostics_vector=diag,
            fingerprint_dict=fp, decision=decision,
            candidates=cands, results=results, seed=42,
        )

        assert entry1.snapshot_hash == entry2.snapshot_hash
        assert entry1.diagnostics_hash == entry2.diagnostics_hash
        assert entry1.decision_hash == entry2.decision_hash


# ---------------------------------------------------------------------------
# Test: ReplayEngine.verify
# ---------------------------------------------------------------------------


class TestReplayVerify:
    """Tests for ReplayEngine.verify() -- replay and verify determinism."""

    def test_verify_passes_for_self_recorded_log(self):
        spec = _make_spec()
        registry = _make_registry()
        log = _run_and_get_log(spec, registry, _simple_evaluator)

        assert log.n_entries > 0, "Log should have at least one entry"

        replay_engine = ReplayEngine(registry=registry)
        verification = replay_engine.verify(log)

        assert verification.verified is True, (
            f"Verification failed. Summary:\n{verification.summary()}"
        )
        assert verification.n_iterations == log.n_entries
        assert verification.n_mismatches == 0
        assert verification.first_mismatch_iteration is None
        assert verification.mismatched_iterations == []


class TestReplayVerifyDeterminism:
    """Verify the same log multiple times -- all should pass."""

    def test_verify_three_times(self):
        spec = _make_spec()
        registry = _make_registry()
        log = _run_and_get_log(spec, registry, _simple_evaluator)

        replay_engine = ReplayEngine(registry=registry)

        for attempt in range(3):
            verification = replay_engine.verify(log)
            assert verification.verified is True, (
                f"Verification failed on attempt {attempt + 1}. "
                f"Summary:\n{verification.summary()}"
            )
            assert verification.n_mismatches == 0


class TestReplayVerifyDetectsTampering:
    """Modify one entry's decision_hash and verify that verify() detects it."""

    def test_tampered_decision_hash_detected(self):
        spec = _make_spec()
        registry = _make_registry()
        log = _run_and_get_log(spec, registry, _simple_evaluator)

        assert log.n_entries > 0, "Need at least one entry to tamper with"

        # Tamper with the first entry's decision_hash.
        original_hash = log.entries[0].decision_hash
        log.entries[0].decision_hash = "0000000000000000"
        assert log.entries[0].decision_hash != original_hash

        replay_engine = ReplayEngine(registry=registry)
        verification = replay_engine.verify(log)

        # The replay will recompute the correct hash, which will differ from
        # the tampered value. So verification should report a mismatch.
        assert verification.verified is False
        assert verification.n_mismatches > 0
        assert verification.first_mismatch_iteration is not None
        assert len(verification.mismatched_iterations) > 0
        # The first mismatch should be at iteration 0 (the tampered entry).
        assert verification.first_mismatch_iteration == log.entries[0].iteration

    def test_tampered_snapshot_hash_detected(self):
        spec = _make_spec()
        registry = _make_registry()
        log = _run_and_get_log(spec, registry, _simple_evaluator)

        assert log.n_entries > 0

        # Tamper with a later entry's snapshot_hash if available.
        tamper_idx = min(1, log.n_entries - 1)
        log.entries[tamper_idx].snapshot_hash = "ffffffffffffffff"

        replay_engine = ReplayEngine(registry=registry)
        verification = replay_engine.verify(log)

        assert verification.verified is False
        assert verification.n_mismatches > 0

    def test_verification_summary_readable(self):
        spec = _make_spec()
        registry = _make_registry()
        log = _run_and_get_log(spec, registry, _simple_evaluator)

        # Tamper.
        log.entries[0].decision_hash = "0000000000000000"

        replay_engine = ReplayEngine(registry=registry)
        verification = replay_engine.verify(log)

        summary = verification.summary()
        assert isinstance(summary, str)
        assert "MISMATCH" in summary
        assert "diverged" in summary
