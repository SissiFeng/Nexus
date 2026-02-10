"""Deterministic replay engine for optimization campaigns.

Supports three modes of operation:

* **VERIFY** -- Replay a recorded decision log with the same seed and
  verify that every iteration produces identical hashes (snapshot,
  diagnostics, decision).
* **COMPARE** -- Lightweight side-by-side comparison of two recorded
  logs without re-execution.
* **WHAT_IF** -- Replay a prefix of a recorded log, then branch with a
  different seed, backend, or evaluation function.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from enum import Enum
from typing import Any, Callable

from optimization_copilot.core.hashing import (
    decision_hash as compute_decision_hash,
    diagnostics_hash as compute_diagnostics_hash,
    snapshot_hash as compute_snapshot_hash,
)
from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.diagnostics.engine import DiagnosticEngine
from optimization_copilot.meta_controller.controller import MetaController
from optimization_copilot.plugins.registry import PluginRegistry
from optimization_copilot.profiler.profiler import ProblemProfiler
from optimization_copilot.replay.log import DecisionLog, DecisionLogEntry
from optimization_copilot.replay.report import (
    ComparisonReport,
    IterationComparison,
    ReplayVerification,
)


class ReplayMode(str, Enum):
    """Supported replay modes."""

    VERIFY = "verify"
    COMPARE = "compare"
    WHAT_IF = "what_if"


class ReplayEngine:
    """Deterministic replay engine for optimization decision logs.

    The engine reconstructs campaign state from recorded log entries and
    re-runs the diagnostic/profiling/decision pipeline to verify
    reproducibility or explore counterfactual scenarios.

    Parameters
    ----------
    registry : PluginRegistry
        Plugin registry for instantiating algorithm backends.
    diag_engine : DiagnosticEngine or None
        Diagnostic engine instance; a default is created if not provided.
    profiler : ProblemProfiler or None
        Problem profiler instance; a default is created if not provided.
    """

    def __init__(
        self,
        registry: PluginRegistry,
        diag_engine: DiagnosticEngine | None = None,
        profiler: ProblemProfiler | None = None,
    ) -> None:
        self.registry = registry
        self.diag_engine = diag_engine or DiagnosticEngine()
        self.profiler = profiler or ProblemProfiler()

    # -- public API ---------------------------------------------------------

    def verify(
        self,
        log: DecisionLog,
        seed: int | None = None,
    ) -> ReplayVerification:
        """Replay a campaign from scratch and verify hash determinism.

        For each iteration the engine:
        1. Reconstructs the snapshot from log entries' ingested_results.
        2. Computes diagnostics via DiagnosticEngine.
        3. Profiles the snapshot via ProblemProfiler.
        4. Runs MetaController.decide() with the same seed.
        5. Compares snapshot_hash, diagnostics_hash, and decision_hash
           against the recorded values.

        Parameters
        ----------
        log : DecisionLog
            The recorded decision log to verify.
        seed : int or None
            Override seed; uses ``log.base_seed`` if None.

        Returns
        -------
        ReplayVerification
        """
        effective_seed = seed if seed is not None else log.base_seed
        controller = MetaController(
            available_backends=self.registry.list_plugins(),
        )

        comparisons: list[IterationComparison] = []
        mismatched: list[int] = []
        first_mismatch: int | None = None

        for entry in log.entries:
            # 1. Reconstruct snapshot at this iteration
            snapshot = self._reconstruct_snapshot_at(log, entry.iteration)

            # 2. Compute diagnostics
            diag_vector = self.diag_engine.compute(snapshot)
            diag_dict = diag_vector.to_dict()

            # 3. Profile
            fingerprint = self.profiler.profile(snapshot)

            # 4. Decide
            previous_phase = None
            prev_entry = log.get_entry(entry.iteration - 1)
            if prev_entry is not None:
                from optimization_copilot.core.models import Phase
                try:
                    previous_phase = Phase(prev_entry.phase)
                except ValueError:
                    previous_phase = None

            decision = controller.decide(
                snapshot=snapshot,
                diagnostics=diag_dict,
                fingerprint=fingerprint,
                seed=effective_seed + entry.iteration,
                previous_phase=previous_phase,
            )

            # 5. Compute hashes
            replay_snapshot_hash = compute_snapshot_hash(snapshot)
            replay_diagnostics_hash = compute_diagnostics_hash(diag_dict)
            replay_decision_hash = compute_decision_hash(decision)

            # 6. Compare
            snapshot_match = replay_snapshot_hash == entry.snapshot_hash
            diagnostics_match = replay_diagnostics_hash == entry.diagnostics_hash
            decision_match = replay_decision_hash == entry.decision_hash
            decisions_match = snapshot_match and diagnostics_match and decision_match

            differences: list[str] = []
            if not snapshot_match:
                differences.append(
                    f"snapshot_hash: expected={entry.snapshot_hash}, "
                    f"got={replay_snapshot_hash}"
                )
            if not diagnostics_match:
                differences.append(
                    f"diagnostics_hash: expected={entry.diagnostics_hash}, "
                    f"got={replay_diagnostics_hash}"
                )
            if not decision_match:
                differences.append(
                    f"decision_hash: expected={entry.decision_hash}, "
                    f"got={replay_decision_hash}"
                )

            comparison = IterationComparison(
                iteration=entry.iteration,
                decisions_match=decisions_match,
                snapshot_hashes_match=snapshot_match,
                diagnostics_hashes_match=diagnostics_match,
                decision_hashes_match=decision_match,
                run_a=entry.to_dict(),
                run_b={
                    "snapshot_hash": replay_snapshot_hash,
                    "diagnostics_hash": replay_diagnostics_hash,
                    "decision_hash": replay_decision_hash,
                    "diagnostics": diag_dict,
                    "fingerprint": fingerprint.to_dict(),
                    "decision": decision.to_dict(),
                    "phase": decision.phase.value,
                    "backend_name": decision.backend_name,
                },
                differences=differences,
            )
            comparisons.append(comparison)

            if not decisions_match:
                mismatched.append(entry.iteration)
                if first_mismatch is None:
                    first_mismatch = entry.iteration

        return ReplayVerification(
            verified=len(mismatched) == 0,
            n_iterations=len(log.entries),
            n_mismatches=len(mismatched),
            first_mismatch_iteration=first_mismatch,
            mismatched_iterations=mismatched,
            details=comparisons,
        )

    def compare(
        self,
        log_a: DecisionLog,
        log_b: DecisionLog,
    ) -> ComparisonReport:
        """Compare two recorded decision logs side-by-side.

        This is a lightweight comparison that does not re-run the
        pipeline -- it only examines the recorded entries.

        Parameters
        ----------
        log_a, log_b : DecisionLog
            The two logs to compare.

        Returns
        -------
        ComparisonReport
        """
        n_iters = min(log_a.n_entries, log_b.n_entries)
        comparisons: list[IterationComparison] = []
        divergence_iter: int | None = None
        n_decision_diffs = 0
        n_backend_diffs = 0
        n_phase_diffs = 0

        for i in range(n_iters):
            entry_a = log_a.entries[i]
            entry_b = log_b.entries[i]

            snapshot_match = entry_a.snapshot_hash == entry_b.snapshot_hash
            diagnostics_match = entry_a.diagnostics_hash == entry_b.diagnostics_hash
            decision_match = entry_a.decision_hash == entry_b.decision_hash
            all_match = snapshot_match and diagnostics_match and decision_match

            differences: list[str] = []
            if not snapshot_match:
                differences.append(
                    f"snapshot_hash: a={entry_a.snapshot_hash}, "
                    f"b={entry_b.snapshot_hash}"
                )
            if not diagnostics_match:
                differences.append(
                    f"diagnostics_hash: a={entry_a.diagnostics_hash}, "
                    f"b={entry_b.diagnostics_hash}"
                )
            if not decision_match:
                differences.append(
                    f"decision_hash: a={entry_a.decision_hash}, "
                    f"b={entry_b.decision_hash}"
                )
                n_decision_diffs += 1
            if entry_a.backend_name != entry_b.backend_name:
                differences.append(
                    f"backend: a={entry_a.backend_name}, "
                    f"b={entry_b.backend_name}"
                )
                n_backend_diffs += 1
            if entry_a.phase != entry_b.phase:
                differences.append(
                    f"phase: a={entry_a.phase}, b={entry_b.phase}"
                )
                n_phase_diffs += 1

            if not all_match and divergence_iter is None:
                divergence_iter = entry_a.iteration

            comparisons.append(
                IterationComparison(
                    iteration=entry_a.iteration,
                    decisions_match=all_match,
                    snapshot_hashes_match=snapshot_match,
                    diagnostics_hashes_match=diagnostics_match,
                    decision_hashes_match=decision_match,
                    run_a=entry_a.to_dict(),
                    run_b=entry_b.to_dict(),
                    differences=differences,
                )
            )

        # Extract phase timelines
        run_a_phases = self._extract_phase_timeline(log_a)
        run_b_phases = self._extract_phase_timeline(log_b)

        # Extract convergence curves
        run_a_curve = self._extract_best_kpi_curve(log_a)
        run_b_curve = self._extract_best_kpi_curve(log_b)

        # Final KPIs
        run_a_final = self._extract_final_kpi(log_a)
        run_b_final = self._extract_final_kpi(log_b)

        return ComparisonReport(
            run_a_id=log_a.campaign_id,
            run_b_id=log_b.campaign_id,
            n_iterations=n_iters,
            divergence_iteration=divergence_iter,
            iteration_comparisons=comparisons,
            run_a_best_kpi_curve=run_a_curve,
            run_b_best_kpi_curve=run_b_curve,
            run_a_phases=run_a_phases,
            run_b_phases=run_b_phases,
            run_a_final_kpi=run_a_final,
            run_b_final_kpi=run_b_final,
            n_decision_differences=n_decision_diffs,
            n_backend_differences=n_backend_diffs,
            n_phase_differences=n_phase_diffs,
        )

    def compare_with_alternative(
        self,
        log: DecisionLog,
        alternative_backend: str,
        alternative_seed: int | None = None,
    ) -> ComparisonReport:
        """Re-run the pipeline with a different backend and compare.

        Uses the nearest-match pattern from CounterfactualEvaluator:
        at each iteration, the alternative backend suggests candidates
        and the nearest observation from the original campaign is used
        as a proxy for the result.

        Parameters
        ----------
        log : DecisionLog
            The original recorded log.
        alternative_backend : str
            Name of the alternative backend to use.
        alternative_seed : int or None
            Seed for the alternative run; defaults to ``log.base_seed + 1``.

        Returns
        -------
        ComparisonReport
        """
        alt_seed = alternative_seed if alternative_seed is not None else log.base_seed + 1
        alt_log = self._replay_with_backend(
            log, alternative_backend, alt_seed,
        )
        return self.compare(log, alt_log)

    def what_if(
        self,
        log: DecisionLog,
        diverge_at: int,
        new_seed: int | None = None,
        new_backend: str | None = None,
        evaluate_fn: Callable[[dict[str, Any]], dict[str, float]] | None = None,
    ) -> DecisionLog:
        """Replay a log prefix then branch with a new strategy.

        Iterations before ``diverge_at`` are copied verbatim from the
        original log.  From ``diverge_at`` onward the pipeline is
        re-run with (optionally) a new seed, a forced backend, and/or a
        custom evaluation function.

        When ``evaluate_fn`` is not provided, the nearest-match pattern
        is used to estimate KPI values from the original observations.

        Parameters
        ----------
        log : DecisionLog
            The original recorded log.
        diverge_at : int
            Iteration number at which to diverge.
        new_seed : int or None
            Override seed for the diverged portion.
        new_backend : str or None
            Force this backend for all diverged iterations.
        evaluate_fn : callable or None
            ``(parameters_dict) -> kpi_dict``; if None, nearest-match
            is used.

        Returns
        -------
        DecisionLog
            A new log containing the prefix + diverged iterations.
        """
        effective_seed = new_seed if new_seed is not None else log.base_seed
        controller = MetaController(
            available_backends=self.registry.list_plugins(),
        )

        # Collect all original observations for nearest-match
        all_original_obs = self._collect_all_observations(log)
        # Infer specs from the log's spec dict
        specs = self._specs_from_log(log)

        new_log = DecisionLog(
            campaign_id=f"{log.campaign_id}_whatif_{diverge_at}",
            spec=log.spec,
            base_seed=effective_seed,
            metadata={
                "source_campaign": log.campaign_id,
                "diverge_at": diverge_at,
                "new_backend": new_backend,
            },
        )

        # Phase 1: copy prefix verbatim
        for entry in log.entries:
            if entry.iteration < diverge_at:
                new_log.append(entry)

        # Phase 2: replay from diverge_at onward
        remaining_iterations = [
            e for e in log.entries if e.iteration >= diverge_at
        ]

        for entry in remaining_iterations:
            iteration = entry.iteration

            # Reconstruct snapshot from new_log's entries
            snapshot = self._reconstruct_snapshot_from_entries(
                new_log, specs, log,
            )

            # Compute diagnostics and profile
            diag_vector = self.diag_engine.compute(snapshot)
            diag_dict = diag_vector.to_dict()
            fingerprint = self.profiler.profile(snapshot)

            # Determine previous phase from new_log
            previous_phase = None
            if new_log.entries:
                from optimization_copilot.core.models import Phase
                try:
                    previous_phase = Phase(new_log.entries[-1].phase)
                except ValueError:
                    previous_phase = None

            # Decide (optionally with forced backend)
            decision = controller.decide(
                snapshot=snapshot,
                diagnostics=diag_dict,
                fingerprint=fingerprint,
                seed=effective_seed + iteration,
                previous_phase=previous_phase,
            )

            # Override backend if requested
            if new_backend is not None:
                decision.backend_name = new_backend
                decision.reason_codes.append(f"what_if_override:backend={new_backend}")

            # Generate candidates using the decided backend
            candidates = self._generate_candidates(
                decision.backend_name,
                snapshot,
                decision.batch_size,
                effective_seed + iteration,
            )

            # Evaluate candidates
            if evaluate_fn is not None:
                results = self._evaluate_with_fn(candidates, evaluate_fn, iteration)
            else:
                results = self._evaluate_nearest_match(
                    candidates, all_original_obs, specs, iteration,
                )

            # Record entry
            new_entry = self.record_iteration(
                iteration=iteration,
                snapshot=snapshot,
                diagnostics_vector=diag_dict,
                fingerprint_dict=fingerprint.to_dict(),
                decision=decision,
                candidates=candidates,
                results=results,
                seed=effective_seed + iteration,
            )
            new_log.append(new_entry)

        return new_log

    # -- static recording helper --------------------------------------------

    @staticmethod
    def record_iteration(
        iteration: int,
        snapshot: CampaignSnapshot,
        diagnostics_vector: dict[str, float],
        fingerprint_dict: dict[str, str],
        decision: Any,
        candidates: list[dict[str, Any]],
        results: list[dict[str, Any]],
        seed: int,
    ) -> DecisionLogEntry:
        """Create a DecisionLogEntry for a single iteration.

        This is the standard way to record an iteration during a live
        optimization run.

        Parameters
        ----------
        iteration : int
        snapshot : CampaignSnapshot
        diagnostics_vector : dict
        fingerprint_dict : dict
        decision : StrategyDecision
        candidates : list of candidate parameter dicts
        results : list of observation dicts (ingested results)
        seed : int

        Returns
        -------
        DecisionLogEntry
        """
        snap_hash = compute_snapshot_hash(snapshot)
        diag_hash = compute_diagnostics_hash(diagnostics_vector)

        # Handle both StrategyDecision objects and plain dicts
        if hasattr(decision, "to_dict"):
            decision_dict = decision.to_dict()
            dec_hash = compute_decision_hash(decision)
            phase = decision.phase.value if hasattr(decision.phase, "value") else str(decision.phase)
            backend = decision.backend_name
            reason = list(decision.reason_codes)
        else:
            decision_dict = dict(decision)
            dec_hash = compute_diagnostics_hash(decision_dict)  # fallback
            phase = str(decision.get("phase", "unknown"))
            backend = str(decision.get("backend_name", "unknown"))
            reason = list(decision.get("reason_codes", []))

        return DecisionLogEntry(
            iteration=iteration,
            timestamp=time.time(),
            snapshot_hash=snap_hash,
            diagnostics_hash=diag_hash,
            diagnostics=dict(diagnostics_vector),
            fingerprint=dict(fingerprint_dict),
            decision=decision_dict,
            decision_hash=dec_hash,
            suggested_candidates=[dict(c) for c in candidates],
            ingested_results=[dict(r) for r in results],
            phase=phase,
            backend_name=backend,
            reason_codes=reason,
            seed=seed,
        )

    # -- internal helpers ---------------------------------------------------

    def _reconstruct_snapshot_at(
        self,
        log: DecisionLog,
        up_to_iteration: int,
    ) -> CampaignSnapshot:
        """Rebuild a CampaignSnapshot from log entries up to an iteration.

        Collects all ingested_results from entries whose iteration is
        strictly less than ``up_to_iteration`` and builds a snapshot
        from the log's spec.
        """
        observations: list[Observation] = []
        for entry in log.entries:
            if entry.iteration >= up_to_iteration:
                break
            for result in entry.ingested_results:
                obs = self._dict_to_observation(result, entry.iteration)
                observations.append(obs)

        return self._build_snapshot(log, observations, up_to_iteration)

    def _reconstruct_snapshot_from_entries(
        self,
        partial_log: DecisionLog,
        specs: list[ParameterSpec],
        original_log: DecisionLog,
    ) -> CampaignSnapshot:
        """Rebuild snapshot from a partial (in-progress) log."""
        observations: list[Observation] = []
        for entry in partial_log.entries:
            for result in entry.ingested_results:
                obs = self._dict_to_observation(result, entry.iteration)
                observations.append(obs)

        return self._build_snapshot(original_log, observations, len(observations))

    def _build_snapshot(
        self,
        log: DecisionLog,
        observations: list[Observation],
        current_iteration: int,
    ) -> CampaignSnapshot:
        """Build a CampaignSnapshot from a log's spec and observations."""
        spec = log.spec
        specs = self._specs_from_log(log)

        objective_names = [
            obj["name"] for obj in spec.get("objectives", [])
        ]
        objective_directions = [
            obj.get("direction", "minimize") for obj in spec.get("objectives", [])
        ]

        # Build constraints from objective bounds
        constraints: list[dict[str, Any]] = []
        for obj in spec.get("objectives", []):
            if obj.get("constraint_lower") is not None:
                constraints.append({
                    "type": "lower_bound",
                    "target": obj["name"],
                    "value": obj["constraint_lower"],
                })
            if obj.get("constraint_upper") is not None:
                constraints.append({
                    "type": "upper_bound",
                    "target": obj["name"],
                    "value": obj["constraint_upper"],
                })

        return CampaignSnapshot(
            campaign_id=log.campaign_id,
            parameter_specs=specs,
            observations=observations,
            objective_names=objective_names,
            objective_directions=objective_directions,
            constraints=constraints,
            current_iteration=current_iteration,
            metadata=spec.get("metadata", {}),
        )

    @staticmethod
    def _specs_from_log(log: DecisionLog) -> list[ParameterSpec]:
        """Extract ParameterSpec list from a log's serialized spec."""
        from optimization_copilot.core.models import VariableType

        specs: list[ParameterSpec] = []
        for p in log.spec.get("parameters", []):
            # Skip frozen parameters
            if p.get("frozen", False):
                continue
            specs.append(
                ParameterSpec(
                    name=p["name"],
                    type=VariableType(p["type"]),
                    lower=p.get("lower"),
                    upper=p.get("upper"),
                    categories=p.get("categories"),
                )
            )
        return specs

    @staticmethod
    def _dict_to_observation(
        data: dict[str, Any],
        fallback_iteration: int,
    ) -> Observation:
        """Convert a result dict to an Observation."""
        return Observation(
            iteration=data.get("iteration", fallback_iteration),
            parameters=data.get("parameters", {}),
            kpi_values=data.get("kpi_values", {}),
            qc_passed=data.get("qc_passed", True),
            is_failure=data.get("is_failure", False),
            failure_reason=data.get("failure_reason"),
            timestamp=data.get("timestamp", 0.0),
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def _extract_best_kpi_curve(log: DecisionLog) -> list[float]:
        """Extract cumulative best KPI values from a log.

        Uses the first objective's best_kpi_value from the diagnostics
        vector at each iteration.  Falls back to 0.0 if not available.
        """
        curve: list[float] = []
        best: float | None = None

        # Determine direction from spec
        objectives = log.spec.get("objectives", [])
        maximize = False
        if objectives:
            direction = objectives[0].get("direction", "minimize")
            maximize = direction.lower().startswith("max")

        for entry in log.entries:
            kpi = entry.diagnostics.get("best_kpi_value", 0.0)
            if best is None:
                best = kpi
            elif maximize:
                best = max(best, kpi)
            else:
                best = min(best, kpi)
            curve.append(best)

        return curve

    @staticmethod
    def _extract_phase_timeline(log: DecisionLog) -> list[tuple[int, str]]:
        """Extract (iteration, phase) pairs from a log."""
        return [(e.iteration, e.phase) for e in log.entries]

    @staticmethod
    def _extract_final_kpi(log: DecisionLog) -> dict[str, float]:
        """Extract the final diagnostics as a KPI summary."""
        if not log.entries:
            return {}
        final = log.entries[-1]
        return {
            "best_kpi_value": final.diagnostics.get("best_kpi_value", 0.0),
            "convergence_trend": final.diagnostics.get("convergence_trend", 0.0),
            "data_efficiency": final.diagnostics.get("data_efficiency", 0.0),
        }

    def _collect_all_observations(
        self,
        log: DecisionLog,
    ) -> list[Observation]:
        """Collect all observations from a log's ingested_results."""
        observations: list[Observation] = []
        for entry in log.entries:
            for result in entry.ingested_results:
                obs = self._dict_to_observation(result, entry.iteration)
                observations.append(obs)
        return observations

    def _generate_candidates(
        self,
        backend_name: str,
        snapshot: CampaignSnapshot,
        batch_size: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        """Generate candidate suggestions using a backend plugin."""
        try:
            plugin = self.registry.get(backend_name)
            plugin.fit(snapshot.observations, snapshot.parameter_specs)
            return plugin.suggest(n_suggestions=batch_size, seed=seed)
        except (KeyError, PermissionError):
            # Backend not available; return empty candidates
            return []

    def _evaluate_nearest_match(
        self,
        candidates: list[dict[str, Any]],
        all_observations: list[Observation],
        specs: list[ParameterSpec],
        iteration: int,
    ) -> list[dict[str, Any]]:
        """Evaluate candidates via nearest-match from original observations.

        Reuses the normalized distance pattern from
        ``counterfactual.evaluator._normalised_distance``.
        """
        results: list[dict[str, Any]] = []
        if not all_observations:
            return results

        for candidate in candidates:
            nearest = self._find_nearest_observation(
                candidate, all_observations, specs,
            )
            if nearest is not None:
                results.append({
                    "iteration": iteration,
                    "parameters": dict(nearest.parameters),
                    "kpi_values": dict(nearest.kpi_values),
                    "qc_passed": nearest.qc_passed,
                    "is_failure": nearest.is_failure,
                    "failure_reason": nearest.failure_reason,
                    "timestamp": nearest.timestamp,
                    "metadata": {
                        **nearest.metadata,
                        "source": "nearest_match",
                    },
                })

        return results

    @staticmethod
    def _evaluate_with_fn(
        candidates: list[dict[str, Any]],
        evaluate_fn: Callable[[dict[str, Any]], dict[str, float]],
        iteration: int,
    ) -> list[dict[str, Any]]:
        """Evaluate candidates using a provided evaluation function."""
        results: list[dict[str, Any]] = []
        for candidate in candidates:
            try:
                kpi_values = evaluate_fn(candidate)
                results.append({
                    "iteration": iteration,
                    "parameters": dict(candidate),
                    "kpi_values": dict(kpi_values),
                    "qc_passed": True,
                    "is_failure": False,
                    "failure_reason": None,
                    "timestamp": time.time(),
                    "metadata": {"source": "evaluate_fn"},
                })
            except Exception as exc:
                results.append({
                    "iteration": iteration,
                    "parameters": dict(candidate),
                    "kpi_values": {},
                    "qc_passed": False,
                    "is_failure": True,
                    "failure_reason": str(exc),
                    "timestamp": time.time(),
                    "metadata": {"source": "evaluate_fn"},
                })
        return results

    @staticmethod
    def _find_nearest_observation(
        candidate: dict[str, Any],
        observations: list[Observation],
        specs: list[ParameterSpec],
    ) -> Observation | None:
        """Find the observation nearest to a candidate in parameter space.

        Uses normalised Euclidean distance, consistent with the pattern
        in ``counterfactual.evaluator._normalised_distance``.

        Returns None if observations is empty.
        """
        if not observations:
            return None

        best_obs = observations[0]
        best_dist = _normalised_distance(candidate, best_obs, specs)

        for obs in observations[1:]:
            d = _normalised_distance(candidate, obs, specs)
            if d < best_dist:
                best_dist = d
                best_obs = obs

        return best_obs

    def _replay_with_backend(
        self,
        log: DecisionLog,
        backend_name: str,
        seed: int,
    ) -> DecisionLog:
        """Re-run a full log with an alternative backend.

        At each iteration, the alternative backend generates candidates
        and the nearest-match from original observations provides KPIs.
        """
        controller = MetaController(
            available_backends=self.registry.list_plugins(),
        )
        all_original_obs = self._collect_all_observations(log)
        specs = self._specs_from_log(log)

        alt_log = DecisionLog(
            campaign_id=f"{log.campaign_id}_alt_{backend_name}",
            spec=log.spec,
            base_seed=seed,
            metadata={
                "source_campaign": log.campaign_id,
                "alternative_backend": backend_name,
            },
        )

        for entry in log.entries:
            iteration = entry.iteration

            # Reconstruct snapshot from alt_log so far
            snapshot = self._reconstruct_snapshot_from_entries(
                alt_log, specs, log,
            )

            # Diagnostics and profiling
            diag_vector = self.diag_engine.compute(snapshot)
            diag_dict = diag_vector.to_dict()
            fingerprint = self.profiler.profile(snapshot)

            # Previous phase
            previous_phase = None
            if alt_log.entries:
                from optimization_copilot.core.models import Phase
                try:
                    previous_phase = Phase(alt_log.entries[-1].phase)
                except ValueError:
                    previous_phase = None

            # Decide (then override backend)
            decision = controller.decide(
                snapshot=snapshot,
                diagnostics=diag_dict,
                fingerprint=fingerprint,
                seed=seed + iteration,
                previous_phase=previous_phase,
            )
            decision.backend_name = backend_name
            decision.reason_codes.append(
                f"alternative_replay:forced_backend={backend_name}"
            )

            # Generate and evaluate via nearest-match
            candidates = self._generate_candidates(
                backend_name, snapshot, decision.batch_size, seed + iteration,
            )
            results = self._evaluate_nearest_match(
                candidates, all_original_obs, specs, iteration,
            )

            # If no candidates were generated (backend unavailable),
            # fall back to copying original results
            if not results and entry.ingested_results:
                results = entry.ingested_results

            new_entry = self.record_iteration(
                iteration=iteration,
                snapshot=snapshot,
                diagnostics_vector=diag_dict,
                fingerprint_dict=fingerprint.to_dict(),
                decision=decision,
                candidates=candidates,
                results=results,
                seed=seed + iteration,
            )
            alt_log.append(new_entry)

        return alt_log


# ---------------------------------------------------------------------------
# Distance helper (reuses pattern from counterfactual/evaluator.py)
# ---------------------------------------------------------------------------


def _normalised_distance(
    suggestion: dict[str, Any],
    obs: Observation,
    specs: list[ParameterSpec],
) -> float:
    """Normalised Euclidean distance between suggestion and observation.

    Continuous/discrete parameters are normalised by their range.
    Categorical parameters contribute 0.0 (match) or 1.0 (mismatch).
    Missing parameters contribute maximum distance (1.0) for that dimension.
    """
    total = 0.0
    n_dims = 0

    for spec in specs:
        s_val = suggestion.get(spec.name)
        o_val = obs.parameters.get(spec.name)

        if s_val is None or o_val is None:
            total += 1.0
            n_dims += 1
            continue

        if spec.type == VariableType.CATEGORICAL:
            total += 0.0 if s_val == o_val else 1.0
        else:
            lo = spec.lower if spec.lower is not None else 0.0
            hi = spec.upper if spec.upper is not None else 1.0
            rng = hi - lo if hi != lo else 1.0
            diff = (float(s_val) - float(o_val)) / rng
            total += diff * diff

        n_dims += 1

    if n_dims == 0:
        return 0.0
    return math.sqrt(total / n_dims)
