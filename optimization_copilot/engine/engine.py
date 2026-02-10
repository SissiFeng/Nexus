"""Main optimization execution engine.

Orchestrates optimization campaigns by coordinating the diagnostic engine,
problem profiler, meta-controller, plugin registry, and spec bridge into a
cohesive execution loop.  Supports both a generator-based pattern (for
interactive / external evaluators) and a convenience ``run_with_evaluator``
method for fully synchronous execution.

All execution is deterministic: given the same ``OptimizationSpec``, seed,
and evaluator function, the engine produces identical results.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generator

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    Phase,
    StrategyDecision,
)
from optimization_copilot.diagnostics.engine import DiagnosticEngine
from optimization_copilot.dsl.bridge import SpecBridge
from optimization_copilot.dsl.spec import OptimizationSpec
from optimization_copilot.engine.events import (
    EngineEvent,
    EventHook,
    EventPayload,
)
from optimization_copilot.engine.state import CampaignState
from optimization_copilot.engine.trial import Trial, TrialBatch, TrialState
from optimization_copilot.meta_controller.controller import MetaController
from optimization_copilot.plugins.registry import PluginRegistry
from optimization_copilot.profiler.profiler import ProblemProfiler

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────


@dataclass
class EngineConfig:
    """Runtime configuration for the optimization engine.

    Attributes:
        max_retries: Maximum number of retry attempts for a failed trial
            before it is marked ABANDONED.
        checkpoint_every: If > 0, automatically checkpoint state to disk
            every N iterations.  0 disables auto-checkpoint.
        checkpoint_path: Filesystem path for checkpoint files.  Required
            when ``checkpoint_every > 0``.
        available_backends: Explicit list of backend names to make
            available to the meta-controller.  ``None`` means use the
            registry defaults.
    """

    max_retries: int = 3
    checkpoint_every: int = 0
    checkpoint_path: str | None = None
    available_backends: list[str] | None = None


# ── Result ────────────────────────────────────────────────


@dataclass
class EngineResult:
    """Final outcome of an optimization campaign.

    Attributes:
        best_trial: Serialized dict of the best completed trial, or
            ``None`` if no trial completed successfully.
        best_kpi_values: KPI values from the best trial.
        total_iterations: Number of completed engine iterations.
        total_trials: Total number of trials created across all iterations.
        total_failures: Number of trials that ended in FAILED or ABANDONED.
        termination_reason: Human-readable reason the campaign ended.
        phase_history: Log of phase transitions with timestamps.
        decision_history: Log of strategy decisions per iteration.
        final_snapshot_dict: Serialized ``CampaignSnapshot`` at termination.
        audit_trail: Per-iteration decision log entries.
    """

    best_trial: dict[str, Any] | None
    best_kpi_values: dict[str, float]
    total_iterations: int
    total_trials: int
    total_failures: int
    termination_reason: str
    phase_history: list[dict[str, Any]]
    decision_history: list[dict[str, Any]]
    final_snapshot_dict: dict[str, Any]
    audit_trail: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a plain dictionary."""
        return {
            "best_trial": self.best_trial,
            "best_kpi_values": dict(self.best_kpi_values),
            "total_iterations": self.total_iterations,
            "total_trials": self.total_trials,
            "total_failures": self.total_failures,
            "termination_reason": self.termination_reason,
            "phase_history": list(self.phase_history),
            "decision_history": list(self.decision_history),
            "final_snapshot_dict": self.final_snapshot_dict,
            "audit_trail": list(self.audit_trail),
        }


# ── Engine ────────────────────────────────────────────────


class OptimizationEngine:
    """Main orchestrator for optimization campaigns.

    Coordinates the full lifecycle of an optimization campaign: diagnostic
    computation, problem profiling, strategy selection via the meta-controller,
    candidate generation via plugins, trial management, and result ingestion.

    Supports two usage patterns:

    **Generator pattern** (for external / async evaluators)::

        engine = OptimizationEngine(spec, registry)
        for batch in engine.run():
            for trial in batch.trials:
                result = my_evaluate(trial.parameters)
                trial.complete(kpi_values=result)
        result = engine.result()

    **Evaluator callback** (for fully synchronous execution)::

        engine = OptimizationEngine(spec, registry)
        result = engine.run_with_evaluator(my_evaluate_fn)
    """

    def __init__(
        self,
        spec: OptimizationSpec,
        registry: PluginRegistry,
        config: EngineConfig | None = None,
        state: CampaignState | None = None,
    ) -> None:
        # Validate spec has at least one parameter and one objective.
        if not spec.parameters:
            raise ValueError("OptimizationSpec must define at least one parameter")
        if not spec.objectives:
            raise ValueError("OptimizationSpec must define at least one objective")

        self._spec = spec
        self._registry = registry
        self._config = config or EngineConfig()

        # Core sub-systems.
        self._bridge = SpecBridge()
        self._diagnostic_engine = DiagnosticEngine()
        self._profiler = ProblemProfiler()
        self._meta_controller = MetaController(
            available_backends=(
                self._config.available_backends
                or registry.list_plugins()
                or ["random"]
            ),
        )

        # Event system.
        self._event_hook = EventHook()

        # Mutable campaign state: restore from checkpoint or create fresh.
        if state is not None:
            self._state = state
        else:
            snapshot = SpecBridge.to_campaign_snapshot(spec)
            self._state = CampaignState(
                spec=spec,
                snapshot=snapshot,
                seed=spec.seed,
            )

        # Internal bookkeeping.
        self._start_time: float = 0.0
        self._total_trials: int = 0
        self._total_failures: int = 0
        self._audit_trail: list[dict[str, Any]] = []
        self._best_trial: Trial | None = None
        self._stopped: bool = False
        self._stop_reason: str = ""
        self._previous_phase: Phase | None = None

    # ── Event registration ────────────────────────────────

    def on(self, event: EngineEvent, callback: Callable[[EventPayload], None]) -> None:
        """Register a callback for a specific engine event.

        Args:
            event: The event type to listen for.
            callback: A callable that accepts an ``EventPayload``.
        """
        self._event_hook.on(event, callback)

    # ── Main execution loop (generator) ───────────────────

    def run(self) -> Generator[TrialBatch, None, None]:
        """Generator that yields :class:`TrialBatch` objects per iteration.

        Each iteration performs the following steps:

        1. Check termination conditions.
        2. Compute diagnostics from the current snapshot.
        3. Profile the problem to produce a fingerprint.
        4. Ask the meta-controller for a strategy decision.
        5. Get the selected plugin, fit it, and generate candidates.
        6. Apply frozen parameter values via the spec bridge.
        7. Create Trial objects and form a TrialBatch.
        8. Yield the batch (caller evaluates trials).
        9. After yield: ingest results into the snapshot.
        10. Handle failures (retry queue).
        11. Record audit trail entry.
        12. Fire events (ITERATION_COMPLETE, PHASE_CHANGE if changed).
        13. Auto-checkpoint if configured.
        """
        self._start_time = time.monotonic()

        while True:
            # Step 1: Check termination.
            should_terminate, reason = self._check_termination()
            if should_terminate:
                self._state.terminated = True
                self._state.termination_reason = reason
                self._emit(EngineEvent.TERMINATION, {"reason": reason})
                return

            iteration = self._state.iteration

            # Step 2: Compute diagnostics.
            diagnostics_vec = self._diagnostic_engine.compute(self._state.snapshot)
            diagnostics = diagnostics_vec.to_dict()

            # Step 3: Profile problem.
            fingerprint = self._profiler.profile(self._state.snapshot)

            # Step 4: Meta-controller decision.
            decision = self._meta_controller.decide(
                snapshot=self._state.snapshot,
                diagnostics=diagnostics,
                fingerprint=fingerprint,
                seed=self._iteration_seed(iteration),
                previous_phase=self._previous_phase,
            )

            # Detect phase change.
            phase_changed = (
                self._previous_phase is not None
                and decision.phase != self._previous_phase
            )
            if phase_changed:
                self._state.phase_history.append({
                    "iteration": iteration,
                    "from_phase": self._previous_phase.value if self._previous_phase else None,
                    "to_phase": decision.phase.value,
                    "reason_codes": list(decision.reason_codes),
                    "timestamp": time.monotonic() - self._start_time,
                })

            # Record decision.
            self._state.decision_history.append(decision.to_dict())
            self._previous_phase = decision.phase

            # Step 5: Get plugin, fit, suggest.
            plugin = self._registry.get(decision.backend_name)

            plugin.fit(
                observations=list(self._state.snapshot.observations),
                parameter_specs=list(self._state.snapshot.parameter_specs),
            )

            batch_size = decision.batch_size
            seed = self._iteration_seed(iteration)
            candidates = plugin.suggest(n_suggestions=batch_size, seed=seed)

            # Step 6: Apply frozen values.
            candidates = SpecBridge.apply_frozen_values(self._spec, candidates)

            # Include any pending retries in this batch.
            retry_trials: list[Trial] = []
            remaining_retries: list[dict[str, Any]] = []
            for retry_data in self._state.pending_retries:
                retry_trial = Trial.from_dict(retry_data)
                retry_trial.iteration = iteration
                retry_trial.state = TrialState.PENDING
                retry_trials.append(retry_trial)
            self._state.pending_retries = remaining_retries  # cleared

            # Step 7: Create Trial objects and TrialBatch.
            now = time.monotonic() - self._start_time
            trials: list[Trial] = []
            for idx, params in enumerate(candidates):
                trial = Trial(
                    trial_id=self._create_trial_id(iteration, idx),
                    iteration=iteration,
                    parameters=dict(params),
                    state=TrialState.PENDING,
                    timestamp=now,
                )
                trials.append(trial)

            # Append retry trials.
            for rt in retry_trials:
                rt.timestamp = now
                trials.append(rt)

            self._total_trials += len(trials)

            batch = TrialBatch(
                batch_id=f"batch-{iteration:04d}",
                iteration=iteration,
                trials=trials,
                strategy_decision=decision,
            )

            # Step 8: Yield batch -- caller evaluates.
            yield batch

            # Step 9: Ingest results.
            if batch.all_failed:
                # Step: Rollback -- do not add all-failed batch to snapshot.
                self._rollback_batch(batch)
            else:
                self.ingest_batch(batch)

            # Step 10: Handle individual failures.
            for trial in batch.trials:
                if trial.state == TrialState.FAILED:
                    self._total_failures += 1
                    self._handle_failed_trial(trial)
                    self._emit(
                        EngineEvent.TRIAL_FAILED,
                        {
                            "trial_id": trial.trial_id,
                            "reason": trial.failure_reason,
                            "attempt": trial.attempt,
                        },
                    )
                elif trial.state == TrialState.COMPLETED:
                    self._emit(
                        EngineEvent.TRIAL_COMPLETE,
                        {
                            "trial_id": trial.trial_id,
                            "kpi_values": dict(trial.kpi_values),
                        },
                    )
                    # Track best trial.
                    self._update_best_trial(trial)

            # Record completed trials.
            for trial in batch.trials:
                if trial.state in (TrialState.COMPLETED, TrialState.FAILED):
                    self._state.completed_trials.append(trial.to_dict())

            # Step 11: Audit trail.
            audit_entry = {
                "iteration": iteration,
                "decision": decision.to_dict(),
                "candidates": [dict(c) for c in candidates],
                "phase": decision.phase.value,
                "backend": decision.backend_name,
                "reason_codes": list(decision.reason_codes),
                "seed": seed,
                "n_completed": batch.n_completed,
                "n_failed": batch.n_failed,
            }
            self._audit_trail.append(audit_entry)

            # Step 12: Fire events.
            if batch.all_failed:
                self._emit(
                    EngineEvent.BATCH_FAILED,
                    {"batch_id": batch.batch_id, "iteration": iteration},
                )
            else:
                self._emit(
                    EngineEvent.BATCH_COMPLETE,
                    {
                        "batch_id": batch.batch_id,
                        "iteration": iteration,
                        "n_completed": batch.n_completed,
                        "n_failed": batch.n_failed,
                    },
                )

            self._emit(
                EngineEvent.ITERATION_COMPLETE,
                {
                    "iteration": iteration,
                    "phase": decision.phase.value,
                    "backend": decision.backend_name,
                    "n_completed": batch.n_completed,
                    "n_failed": batch.n_failed,
                },
            )

            if phase_changed:
                self._emit(
                    EngineEvent.PHASE_CHANGE,
                    {
                        "iteration": iteration,
                        "from_phase": (
                            self._state.phase_history[-1]["from_phase"]
                            if self._state.phase_history
                            else None
                        ),
                        "to_phase": decision.phase.value,
                    },
                )

            # Step 13: Auto-checkpoint.
            if (
                self._config.checkpoint_every > 0
                and self._config.checkpoint_path
                and (iteration + 1) % self._config.checkpoint_every == 0
            ):
                self.checkpoint()
                self._emit(
                    EngineEvent.CHECKPOINT_SAVED,
                    {
                        "iteration": iteration,
                        "path": self._config.checkpoint_path,
                    },
                )

            # Advance iteration counter.
            self._state.iteration += 1
            self._state.snapshot.current_iteration = self._state.iteration

    # ── Batch and trial ingestion ─────────────────────────

    def ingest_batch(self, batch: TrialBatch) -> None:
        """Ingest all completed and failed trials from a batch into the snapshot.

        Converts each eligible trial to an :class:`Observation` and appends
        it to the campaign snapshot.

        Args:
            batch: A batch whose trials have been evaluated.
        """
        for trial in batch.trials:
            if trial.state in (TrialState.COMPLETED, TrialState.FAILED):
                self.ingest_trial(trial)

    def ingest_trial(self, trial: Trial) -> None:
        """Ingest a single completed or failed trial into the snapshot.

        Args:
            trial: A trial in COMPLETED or FAILED state.

        Raises:
            ValueError: If the trial cannot be converted to an observation
                (e.g. still PENDING or ABANDONED).
        """
        observation = trial.to_observation()
        self._state.snapshot.observations.append(observation)

    # ── Convenience: run with evaluator ───────────────────

    def run_with_evaluator(
        self,
        evaluate_fn: Callable[[dict[str, Any]], dict[str, float]],
    ) -> EngineResult:
        """Run a full campaign using a synchronous evaluator function.

        Iterates the generator, calls ``evaluate_fn(params_dict)`` for each
        trial, and handles exceptions as trial failures.

        Args:
            evaluate_fn: A function that accepts a parameter dictionary and
                returns a KPI dictionary.  Raising any exception marks the
                trial as failed.

        Returns:
            The final :class:`EngineResult`.
        """
        for batch in self.run():
            for trial in batch.trials:
                trial.state = TrialState.RUNNING
                try:
                    kpi_values = evaluate_fn(trial.parameters)
                    trial.complete(kpi_values=kpi_values)
                except Exception as exc:
                    trial.fail(reason=str(exc))
        return self.result()

    # ── Termination ───────────────────────────────────────

    def _check_termination(self) -> tuple[bool, str]:
        """Check all termination conditions.

        Returns:
            A tuple of (should_terminate, reason).
        """
        # Manual stop.
        if self._stopped:
            return True, self._stop_reason

        # Already terminated (e.g. restored from checkpoint in terminated state).
        if self._state.terminated:
            return True, self._state.termination_reason or "already_terminated"

        budget = self._spec.budget
        iteration = self._state.iteration

        # Max iterations.
        if budget.max_iterations is not None and iteration >= budget.max_iterations:
            return True, f"max_iterations_reached:{budget.max_iterations}"

        # Max samples (total observations).
        if budget.max_samples is not None:
            n_obs = self._state.snapshot.n_observations
            if n_obs >= budget.max_samples:
                return True, f"max_samples_reached:{n_obs}/{budget.max_samples}"

        # Max time.
        if budget.max_time_seconds is not None and self._start_time > 0:
            elapsed = time.monotonic() - self._start_time
            if elapsed >= budget.max_time_seconds:
                return True, f"max_time_reached:{elapsed:.1f}s/{budget.max_time_seconds}s"

        # TERMINATION phase from meta-controller (check previous phase).
        if self._previous_phase == Phase.TERMINATION:
            return True, "phase_termination"

        return False, ""

    def stop(self, reason: str = "manual_stop") -> None:
        """Request a graceful stop at the next iteration boundary.

        Args:
            reason: Human-readable reason for stopping.
        """
        self._stopped = True
        self._stop_reason = reason

    # ── Failure handling ──────────────────────────────────

    def _handle_failed_trial(self, trial: Trial) -> None:
        """Handle a failed trial: re-queue for retry or mark abandoned.

        Args:
            trial: A trial in FAILED state.
        """
        if trial.attempt < self._config.max_retries:
            # Re-queue with incremented attempt.
            retry = Trial(
                trial_id=f"{trial.trial_id}-retry{trial.attempt + 1}",
                iteration=trial.iteration,
                parameters=dict(trial.parameters),
                state=TrialState.PENDING,
                attempt=trial.attempt + 1,
                metadata=dict(trial.metadata),
            )
            self._state.pending_retries.append(retry.to_dict())
            logger.debug(
                "Trial %s queued for retry (attempt %d/%d)",
                trial.trial_id,
                retry.attempt,
                self._config.max_retries,
            )
        else:
            trial.abandon(
                reason=(
                    f"max_retries_exceeded:{self._config.max_retries} "
                    f"(last_failure: {trial.failure_reason})"
                )
            )
            self._total_failures += 1  # count abandonment as additional failure
            logger.debug(
                "Trial %s abandoned after %d attempts",
                trial.trial_id,
                trial.attempt,
            )

    def _rollback_batch(self, batch: TrialBatch) -> None:
        """Rollback a batch where all trials failed.

        Instead of ingesting observations, we handle each failed trial
        for retry/abandonment but do not add them to the snapshot.

        Args:
            batch: A batch where ``all_failed`` is True.
        """
        logger.warning(
            "Batch %s (iteration %d): all %d trials failed, rolling back",
            batch.batch_id,
            batch.iteration,
            len(batch.trials),
        )
        # Failed trials are still handled individually for retries in the
        # main loop; we just skip adding observations to the snapshot.

    # ── Checkpoint / resume ───────────────────────────────

    def checkpoint(self) -> CampaignState:
        """Checkpoint the current campaign state.

        If a ``checkpoint_path`` is configured, also writes to disk.

        Returns:
            The current :class:`CampaignState`.
        """
        if self._config.checkpoint_path:
            self._state.checkpoint_to_file(self._config.checkpoint_path)
            logger.info(
                "Checkpoint saved at iteration %d to %s",
                self._state.iteration,
                self._config.checkpoint_path,
            )
        return self._state

    # ── Result assembly ───────────────────────────────────

    def result(self) -> EngineResult:
        """Assemble the final campaign result.

        Returns:
            An :class:`EngineResult` summarizing the entire campaign.
        """
        best_trial_dict: dict[str, Any] | None = None
        best_kpi: dict[str, float] = {}

        if self._best_trial is not None:
            best_trial_dict = self._best_trial.to_dict()
            best_kpi = dict(self._best_trial.kpi_values)

        termination_reason = self._state.termination_reason or "not_terminated"
        if self._stopped and not self._state.termination_reason:
            termination_reason = self._stop_reason

        return EngineResult(
            best_trial=best_trial_dict,
            best_kpi_values=best_kpi,
            total_iterations=self._state.iteration,
            total_trials=self._total_trials,
            total_failures=self._total_failures,
            termination_reason=termination_reason,
            phase_history=list(self._state.phase_history),
            decision_history=list(self._state.decision_history),
            final_snapshot_dict=self._state.snapshot.to_dict(),
            audit_trail=list(self._audit_trail),
        )

    # ── Best trial tracking ───────────────────────────────

    def _update_best_trial(self, trial: Trial) -> None:
        """Update the tracked best trial if this trial is better.

        Uses the primary objective (first objective in the spec) to
        compare trials.  Respects the direction (minimize / maximize).

        Args:
            trial: A completed trial to compare against the current best.
        """
        if trial.state != TrialState.COMPLETED:
            return

        if not self._spec.objectives:
            return

        primary_obj = self._spec.objectives[0]
        primary_name = primary_obj.name
        maximize = primary_obj.direction.value == "maximize"

        trial_value = trial.kpi_values.get(primary_name)
        if trial_value is None:
            return

        if self._best_trial is None:
            self._best_trial = trial
            return

        best_value = self._best_trial.kpi_values.get(primary_name)
        if best_value is None:
            self._best_trial = trial
            return

        if maximize:
            if trial_value > best_value:
                self._best_trial = trial
        else:
            if trial_value < best_value:
                self._best_trial = trial

    # ── ID and seed generation ────────────────────────────

    def _create_trial_id(self, iteration: int, index: int) -> str:
        """Create a deterministic trial identifier.

        Args:
            iteration: Current iteration number.
            index: Trial index within the iteration.

        Returns:
            A string identifier of the form ``t-IIII-JJ``.
        """
        return f"t-{iteration:04d}-{index:02d}"

    def _iteration_seed(self, iteration: int) -> int:
        """Compute a deterministic per-iteration seed.

        Args:
            iteration: Current iteration number.

        Returns:
            ``spec.seed + iteration * 1000``.
        """
        return self._spec.seed + iteration * 1000

    # ── Event emission helper ─────────────────────────────

    def _emit(self, event: EngineEvent, data: dict[str, Any]) -> None:
        """Emit an event through the event hook.

        Args:
            event: The event type to emit.
            data: Event-specific payload data.
        """
        payload = EventPayload(
            event=event,
            iteration=self._state.iteration,
            data=data,
        )
        try:
            self._event_hook.emit(payload)
        except Exception:
            logger.exception(
                "Event handler raised an exception for %s at iteration %d",
                event.value,
                self._state.iteration,
            )
