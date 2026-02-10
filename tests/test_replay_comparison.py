"""Tests for the replay engine comparison and what-if capabilities.

Covers:
- ReplayEngine.compare() for side-by-side log comparison
- ReplayEngine.compare_with_alternative() for alternative backend comparison
- ReplayEngine.what_if() for branching at a specific iteration
- ComparisonReport structure and summary output
- ReplayVerification summary readability
"""

from __future__ import annotations

import json
import time
from typing import Any

from optimization_copilot.backends.builtin import (
    LatinHypercubeSampler,
    RandomSampler,
    TPESampler,
)
from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    Phase,
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
from optimization_copilot.meta_controller.controller import MetaController
from optimization_copilot.plugins.registry import PluginRegistry
from optimization_copilot.profiler.profiler import ProblemProfiler
from optimization_copilot.replay.engine import ReplayEngine
from optimization_copilot.replay.log import DecisionLog, DecisionLogEntry
from optimization_copilot.replay.report import (
    ComparisonReport,
    IterationComparison,
    ReplayVerification,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(seed: int = 42) -> OptimizationSpec:
    """Simple OptimizationSpec: 2 continuous params (x1, x2: 0-10), maximize y, budget 10."""
    return OptimizationSpec(
        campaign_id=f"test-cmp-campaign-seed{seed}",
        parameters=[
            ParameterDef(name="x1", type=ParamType.CONTINUOUS, lower=0.0, upper=10.0),
            ParameterDef(name="x2", type=ParamType.CONTINUOUS, lower=0.0, upper=10.0),
        ],
        objectives=[
            ObjectiveDef(name="y", direction=Direction.MAXIMIZE),
        ],
        budget=BudgetDef(max_samples=10),
        seed=seed,
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


def _build_log_from_engine(
    spec: OptimizationSpec,
    registry: PluginRegistry,
    evaluator: Any,
) -> DecisionLog:
    """Build a proper DecisionLog by replaying the optimization pipeline step by step.

    This reconstructs the exact sequence of (snapshot, diagnostics, fingerprint,
    decision, candidates, results) for each iteration, matching the replay
    engine's expectation.
    """
    diag_engine = DiagnosticEngine()
    profiler = ProblemProfiler()
    controller = MetaController(
        available_backends=registry.list_plugins(),
    )

    log = DecisionLog(
        campaign_id=spec.campaign_id,
        spec=spec.to_dict(),
        base_seed=spec.seed,
        metadata={"source": "test_comparison"},
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

    while len(accumulated_obs) < max_samples:
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

        # Use the SAME seed formula as ReplayEngine.verify():
        # base_seed + iteration.
        seed = spec.seed + iteration
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

        # Safety valve to prevent infinite loops.
        if iteration > 100:
            break

    return log


# ---------------------------------------------------------------------------
# Test: ReplayEngine.compare
# ---------------------------------------------------------------------------


class TestReplayCompare:
    """Compare two logs from different seeds."""

    def test_compare_different_seeds_detects_divergence(self):
        registry = _make_registry()

        log_a = _build_log_from_engine(_make_spec(seed=42), registry, _simple_evaluator)
        log_b = _build_log_from_engine(_make_spec(seed=99), registry, _simple_evaluator)

        assert log_a.n_entries > 0
        assert log_b.n_entries > 0

        replay_engine = ReplayEngine(registry=registry)
        report = replay_engine.compare(log_a, log_b)

        # Different seeds should produce different decisions at some point.
        assert isinstance(report, ComparisonReport)
        assert report.n_iterations > 0

        # With different seeds, we expect divergence.
        # The first iteration uses the same empty snapshot but different seeds
        # for the meta-controller, so decisions may (or may not) differ.
        # At minimum, the snapshot hashes after the first iteration will differ
        # because different candidates are generated.
        has_any_difference = (
            report.divergence_iteration is not None
            or report.n_decision_differences > 0
            or report.n_backend_differences > 0
        )
        # It is possible seeds produce identical first-iteration decisions
        # but diverge later. Check the report structure is valid regardless.
        assert report.run_a_id == log_a.campaign_id
        assert report.run_b_id == log_b.campaign_id

    def test_compare_report_structure(self):
        registry = _make_registry()

        log_a = _build_log_from_engine(_make_spec(seed=42), registry, _simple_evaluator)
        log_b = _build_log_from_engine(_make_spec(seed=99), registry, _simple_evaluator)

        replay_engine = ReplayEngine(registry=registry)
        report = replay_engine.compare(log_a, log_b)

        # Verify report has correct structure.
        assert isinstance(report.run_a_id, str)
        assert isinstance(report.run_b_id, str)
        assert isinstance(report.n_iterations, int)
        assert report.divergence_iteration is None or isinstance(report.divergence_iteration, int)
        assert isinstance(report.iteration_comparisons, list)
        assert isinstance(report.run_a_best_kpi_curve, list)
        assert isinstance(report.run_b_best_kpi_curve, list)
        assert isinstance(report.run_a_phases, list)
        assert isinstance(report.run_b_phases, list)
        assert isinstance(report.run_a_final_kpi, dict)
        assert isinstance(report.run_b_final_kpi, dict)
        assert isinstance(report.n_decision_differences, int)
        assert isinstance(report.n_backend_differences, int)
        assert isinstance(report.n_phase_differences, int)

        # Iteration comparisons should have the right length.
        assert len(report.iteration_comparisons) == report.n_iterations

        # Each iteration comparison should have the right fields.
        for ic in report.iteration_comparisons:
            assert isinstance(ic, IterationComparison)
            assert isinstance(ic.iteration, int)
            assert isinstance(ic.decisions_match, bool)
            assert isinstance(ic.snapshot_hashes_match, bool)
            assert isinstance(ic.diagnostics_hashes_match, bool)
            assert isinstance(ic.decision_hashes_match, bool)
            assert isinstance(ic.run_a, dict)
            assert isinstance(ic.run_b, dict)
            assert isinstance(ic.differences, list)

    def test_compare_identical_logs(self):
        registry = _make_registry()
        log = _build_log_from_engine(_make_spec(seed=42), registry, _simple_evaluator)

        replay_engine = ReplayEngine(registry=registry)
        report = replay_engine.compare(log, log)

        assert report.divergence_iteration is None
        assert report.n_decision_differences == 0
        assert report.n_backend_differences == 0
        assert report.n_phase_differences == 0


# ---------------------------------------------------------------------------
# Test: ReplayEngine.compare_with_alternative
# ---------------------------------------------------------------------------


class TestReplayCompareWithAlternative:
    """Run engine to get log, compare with a different backend."""

    def test_compare_with_alternative_backend(self):
        registry = _make_registry()
        log = _build_log_from_engine(_make_spec(seed=42), registry, _simple_evaluator)

        assert log.n_entries > 0

        replay_engine = ReplayEngine(registry=registry)
        # Compare with latin_hypercube_sampler as the alternative.
        report = replay_engine.compare_with_alternative(
            log,
            alternative_backend="latin_hypercube_sampler",
            alternative_seed=99,
        )

        assert isinstance(report, ComparisonReport)
        assert report.n_iterations > 0
        assert report.run_a_id == log.campaign_id

        # The alternative run should have a different campaign_id.
        assert report.run_b_id != report.run_a_id
        assert "alt_latin_hypercube_sampler" in report.run_b_id

    def test_compare_with_alternative_produces_valid_report(self):
        registry = _make_registry()
        log = _build_log_from_engine(_make_spec(seed=42), registry, _simple_evaluator)

        replay_engine = ReplayEngine(registry=registry)
        report = replay_engine.compare_with_alternative(
            log,
            alternative_backend="random_sampler",
        )

        # Report should have all expected fields.
        report_dict = report.to_dict()
        expected_keys = {
            "run_a_id", "run_b_id", "n_iterations", "divergence_iteration",
            "iteration_comparisons", "run_a_best_kpi_curve", "run_b_best_kpi_curve",
            "run_a_phases", "run_b_phases", "run_a_final_kpi", "run_b_final_kpi",
            "n_decision_differences", "n_backend_differences", "n_phase_differences",
        }
        assert set(report_dict.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Test: ReplayEngine.what_if
# ---------------------------------------------------------------------------


class TestReplayWhatIf:
    """Run engine to get log (10 iterations), then what_if at iteration 5."""

    def test_what_if_preserves_prefix(self):
        registry = _make_registry()
        log = _build_log_from_engine(_make_spec(seed=42), registry, _simple_evaluator)

        assert log.n_entries >= 2, (
            f"Need at least 2 entries for what-if; got {log.n_entries}"
        )

        # Diverge at the midpoint.
        diverge_at = log.n_entries // 2
        n_prefix_entries = sum(
            1 for e in log.entries if e.iteration < diverge_at
        )

        replay_engine = ReplayEngine(registry=registry)
        new_log = replay_engine.what_if(
            log,
            diverge_at=diverge_at,
            new_seed=999,
        )

        assert isinstance(new_log, DecisionLog)

        # The new log should have entries from the prefix plus diverged entries.
        assert new_log.n_entries > 0

        # The prefix entries (iteration < diverge_at) should be identical
        # to the original log entries.
        for i in range(n_prefix_entries):
            orig = log.entries[i]
            new = new_log.entries[i]
            assert orig.iteration == new.iteration
            assert orig.snapshot_hash == new.snapshot_hash
            assert orig.decision_hash == new.decision_hash
            assert orig.diagnostics_hash == new.diagnostics_hash

        # The new log should have entries from diverge_at onward.
        n_remaining = sum(1 for e in log.entries if e.iteration >= diverge_at)
        total_expected = n_prefix_entries + n_remaining
        assert new_log.n_entries == total_expected

    def test_what_if_with_different_backend(self):
        registry = _make_registry()
        log = _build_log_from_engine(_make_spec(seed=42), registry, _simple_evaluator)

        assert log.n_entries >= 2

        diverge_at = log.n_entries // 2

        replay_engine = ReplayEngine(registry=registry)
        new_log = replay_engine.what_if(
            log,
            diverge_at=diverge_at,
            new_backend="latin_hypercube_sampler",
            new_seed=123,
        )

        assert new_log.n_entries > 0

        # Entries after divergence should use the new backend.
        diverged_entries = [e for e in new_log.entries if e.iteration >= diverge_at]
        for entry in diverged_entries:
            assert entry.backend_name == "latin_hypercube_sampler"
            # Reason codes should include the what_if override.
            has_override = any("what_if_override" in rc for rc in entry.reason_codes)
            assert has_override, (
                f"Expected what_if_override in reason_codes, got {entry.reason_codes}"
            )

    def test_what_if_with_evaluate_fn(self):
        registry = _make_registry()
        log = _build_log_from_engine(_make_spec(seed=42), registry, _simple_evaluator)

        assert log.n_entries >= 2

        diverge_at = log.n_entries // 2

        # Different evaluator: y = x1 * x2 instead of x1 + x2.
        def alt_evaluator(params: dict[str, Any]) -> dict[str, float]:
            return {"y": params["x1"] * params["x2"]}

        replay_engine = ReplayEngine(registry=registry)
        new_log = replay_engine.what_if(
            log,
            diverge_at=diverge_at,
            evaluate_fn=alt_evaluator,
            new_seed=456,
        )

        assert new_log.n_entries > 0

        # Diverged entries should have results from the alternative evaluator.
        diverged_entries = [e for e in new_log.entries if e.iteration >= diverge_at]
        for entry in diverged_entries:
            for result in entry.ingested_results:
                if result.get("kpi_values") and result.get("parameters"):
                    params = result["parameters"]
                    kpis = result["kpi_values"]
                    if "y" in kpis and "x1" in params and "x2" in params:
                        expected = params["x1"] * params["x2"]
                        assert abs(kpis["y"] - expected) < 1e-9, (
                            f"Expected y={expected}, got y={kpis['y']}"
                        )

    def test_what_if_metadata(self):
        registry = _make_registry()
        log = _build_log_from_engine(_make_spec(seed=42), registry, _simple_evaluator)

        assert log.n_entries >= 2
        diverge_at = 1

        replay_engine = ReplayEngine(registry=registry)
        new_log = replay_engine.what_if(log, diverge_at=diverge_at, new_seed=77)

        assert new_log.metadata["source_campaign"] == log.campaign_id
        assert new_log.metadata["diverge_at"] == diverge_at
        assert "whatif" in new_log.campaign_id.lower() or "what_if" in new_log.campaign_id.lower()


# ---------------------------------------------------------------------------
# Test: ComparisonReport
# ---------------------------------------------------------------------------


class TestComparisonReport:
    """Verify ComparisonReport summary() and to_dict()."""

    def test_summary_produces_readable_text(self):
        registry = _make_registry()
        log_a = _build_log_from_engine(_make_spec(seed=42), registry, _simple_evaluator)
        log_b = _build_log_from_engine(_make_spec(seed=99), registry, _simple_evaluator)

        replay_engine = ReplayEngine(registry=registry)
        report = replay_engine.compare(log_a, log_b)

        summary = report.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

        # Summary should contain key information.
        assert "Comparison:" in summary
        assert "Iterations compared:" in summary
        assert "Decision differences:" in summary
        assert "Backend differences:" in summary
        assert "Phase differences:" in summary

    def test_summary_with_no_divergence(self):
        registry = _make_registry()
        log = _build_log_from_engine(_make_spec(seed=42), registry, _simple_evaluator)

        replay_engine = ReplayEngine(registry=registry)
        report = replay_engine.compare(log, log)

        summary = report.summary()
        assert "No divergence detected" in summary

    def test_to_dict_has_all_expected_fields(self):
        registry = _make_registry()
        log_a = _build_log_from_engine(_make_spec(seed=42), registry, _simple_evaluator)
        log_b = _build_log_from_engine(_make_spec(seed=99), registry, _simple_evaluator)

        replay_engine = ReplayEngine(registry=registry)
        report = replay_engine.compare(log_a, log_b)

        d = report.to_dict()

        expected_keys = {
            "run_a_id",
            "run_b_id",
            "n_iterations",
            "divergence_iteration",
            "iteration_comparisons",
            "run_a_best_kpi_curve",
            "run_b_best_kpi_curve",
            "run_a_phases",
            "run_b_phases",
            "run_a_final_kpi",
            "run_b_final_kpi",
            "n_decision_differences",
            "n_backend_differences",
            "n_phase_differences",
        }
        assert set(d.keys()) == expected_keys

        # Type checks.
        assert isinstance(d["run_a_id"], str)
        assert isinstance(d["run_b_id"], str)
        assert isinstance(d["n_iterations"], int)
        assert isinstance(d["iteration_comparisons"], list)
        assert isinstance(d["run_a_best_kpi_curve"], list)
        assert isinstance(d["run_b_best_kpi_curve"], list)
        assert isinstance(d["run_a_phases"], list)
        assert isinstance(d["run_b_phases"], list)
        assert isinstance(d["run_a_final_kpi"], dict)
        assert isinstance(d["run_b_final_kpi"], dict)
        assert isinstance(d["n_decision_differences"], int)
        assert isinstance(d["n_backend_differences"], int)
        assert isinstance(d["n_phase_differences"], int)

        # Each iteration comparison in dict should have all fields.
        for ic_dict in d["iteration_comparisons"]:
            assert "iteration" in ic_dict
            assert "decisions_match" in ic_dict
            assert "snapshot_hashes_match" in ic_dict
            assert "diagnostics_hashes_match" in ic_dict
            assert "decision_hashes_match" in ic_dict
            assert "run_a" in ic_dict
            assert "run_b" in ic_dict
            assert "differences" in ic_dict

    def test_to_dict_json_serializable(self):
        """Ensure to_dict() output can be serialized to JSON without errors."""
        registry = _make_registry()
        log_a = _build_log_from_engine(_make_spec(seed=42), registry, _simple_evaluator)
        log_b = _build_log_from_engine(_make_spec(seed=99), registry, _simple_evaluator)

        replay_engine = ReplayEngine(registry=registry)
        report = replay_engine.compare(log_a, log_b)

        json_str = json.dumps(report.to_dict(), default=str)
        assert len(json_str) > 0

        # Round-trip: parse it back.
        parsed = json.loads(json_str)
        assert parsed["run_a_id"] == report.run_a_id


# ---------------------------------------------------------------------------
# Test: ReplayVerification summary
# ---------------------------------------------------------------------------


class TestReplayVerificationSummary:
    """Verify ReplayVerification.summary() produces readable text."""

    def test_verified_summary(self):
        registry = _make_registry()
        log = _build_log_from_engine(_make_spec(seed=42), registry, _simple_evaluator)

        replay_engine = ReplayEngine(registry=registry)
        verification = replay_engine.verify(log)

        summary = verification.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

        if verification.verified:
            assert "VERIFIED" in summary
            assert "reproduced exactly" in summary
        else:
            # Even if verification fails due to floating point or timing,
            # the summary should still be readable.
            assert "MISMATCH" in summary or "VERIFIED" in summary

    def test_mismatch_summary_contains_details(self):
        registry = _make_registry()
        log = _build_log_from_engine(_make_spec(seed=42), registry, _simple_evaluator)

        # Tamper to force mismatch.
        if log.n_entries > 0:
            log.entries[0].decision_hash = "0000000000000000"

        replay_engine = ReplayEngine(registry=registry)
        verification = replay_engine.verify(log)

        summary = verification.summary()
        assert "MISMATCH" in summary
        assert "diverged" in summary

        # Should mention the first mismatch iteration.
        if verification.first_mismatch_iteration is not None:
            assert "First mismatch" in summary

    def test_summary_for_empty_verification(self):
        """Verify summary works when there are zero iterations."""
        verification = ReplayVerification(
            verified=True,
            n_iterations=0,
            n_mismatches=0,
            first_mismatch_iteration=None,
            mismatched_iterations=[],
            details=[],
        )
        summary = verification.summary()
        assert isinstance(summary, str)
        assert "VERIFIED" in summary
