"""Asynchronous trial scheduling with batch management.

Manages the lifecycle of optimization trials across multiple workers,
supporting synchronous batch generation, asynchronous completion
handling, greedy backfill for idle workers, and full trial state
tracking.

Features:
1. Synchronous batch: generate k candidates at once
2. Async updates: worker completion triggers model update
3. Greedy backfill: idle workers get new tasks immediately
4. Trial lifecycle management (pending -> running -> completed/failed)
5. Full serialization for persistence
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Trial status
# ---------------------------------------------------------------------------

class TrialStatus(Enum):
    """Lifecycle states for an optimization trial."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AsyncTrial:
    """A single asynchronous optimization trial.

    Attributes:
        trial_id: Unique identifier for this trial.
        params: Parameter dict suggested by the optimizer.
        status: Current lifecycle status.
        worker_id: Identifier of the worker executing this trial,
            or ``None`` if not yet assigned.
        submitted_at: Timestamp (seconds since epoch) when the trial
            was created.
        completed_at: Timestamp when the trial completed or failed,
            or ``None`` if still in progress.
        result: Result dict returned by the worker on completion.
            Typically contains ``"objective"`` and possibly other
            metric keys.
        error: Error description if the trial failed.
    """

    trial_id: str
    params: dict[str, Any]
    status: TrialStatus = TrialStatus.PENDING
    worker_id: str | None = None
    submitted_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    result: dict[str, Any] | None = None
    error: str = ""

    # -- convenience ------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True when the trial is pending or running."""
        return self.status in (TrialStatus.PENDING, TrialStatus.RUNNING)

    @property
    def is_terminal(self) -> bool:
        """True when the trial has completed or failed."""
        return self.status in (TrialStatus.COMPLETED, TrialStatus.FAILED)

    @property
    def duration(self) -> float | None:
        """Wall-clock duration in seconds, or ``None`` if not completed."""
        if self.completed_at is None:
            return None
        return self.completed_at - self.submitted_at

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "trial_id": self.trial_id,
            "params": dict(self.params),
            "status": self.status.value,
            "worker_id": self.worker_id,
            "submitted_at": self.submitted_at,
            "completed_at": self.completed_at,
            "result": dict(self.result) if self.result else None,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AsyncTrial:
        """Deserialize from a dict."""
        return cls(
            trial_id=data["trial_id"],
            params=data.get("params", {}),
            status=TrialStatus(data.get("status", "pending")),
            worker_id=data.get("worker_id"),
            submitted_at=data.get("submitted_at", 0.0),
            completed_at=data.get("completed_at"),
            result=data.get("result"),
            error=data.get("error", ""),
        )


# ---------------------------------------------------------------------------
# Batch scheduler
# ---------------------------------------------------------------------------

class BatchScheduler:
    """Batch and asynchronous trial scheduler.

    Manages trial lifecycle across a pool of workers.  Supports both
    synchronous batch mode (generate *k* candidates at once) and
    asynchronous mode where workers report completions independently
    and idle workers can be immediately backfilled.

    Batch strategies:

    * ``"simple"``: Trials are issued 1-to-1 from suggestions with
      no decorrelation.
    * ``"round_robin"``: Suggestions are assigned to workers in
      round-robin order (useful for load-balancing metadata).
    * ``"greedy"``: Always fills idle workers first, creating new
      trials on demand.

    Args:
        n_workers: Number of parallel workers available.
        batch_strategy: One of ``"simple"``, ``"round_robin"``,
            ``"greedy"``.
    """

    _VALID_STRATEGIES = ("simple", "round_robin", "greedy")

    def __init__(
        self,
        n_workers: int = 1,
        batch_strategy: str = "simple",
    ) -> None:
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        if batch_strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"batch_strategy must be one of {self._VALID_STRATEGIES}, "
                f"got {batch_strategy!r}"
            )
        self._n_workers = n_workers
        self._strategy = batch_strategy
        self._active_trials: dict[str, AsyncTrial] = {}
        self._completed_trials: list[AsyncTrial] = []
        self._trial_counter: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_workers(self) -> int:
        """Number of parallel workers."""
        return self._n_workers

    @property
    def batch_strategy(self) -> str:
        """Current batch strategy name."""
        return self._strategy

    @property
    def total_trials(self) -> int:
        """Total number of trials created (active + completed)."""
        return len(self._active_trials) + len(self._completed_trials)

    # ------------------------------------------------------------------
    # Batch creation
    # ------------------------------------------------------------------

    def request_batch(
        self,
        suggestions: list[dict[str, Any]],
    ) -> list[AsyncTrial]:
        """Create :class:`AsyncTrial` instances from suggested parameter dicts.

        Each suggestion becomes a new trial in ``PENDING`` status.

        Args:
            suggestions: List of parameter dicts from the optimizer.

        Returns:
            List of newly created :class:`AsyncTrial` objects.
        """
        trials: list[AsyncTrial] = []
        for params in suggestions:
            trial = self._create_trial(params)
            self._active_trials[trial.trial_id] = trial
            trials.append(trial)
        return trials

    # ------------------------------------------------------------------
    # Trial lifecycle
    # ------------------------------------------------------------------

    def submit_trial(self, trial_id: str, worker_id: str) -> None:
        """Mark a trial as submitted to a worker.

        Transitions the trial from ``PENDING`` to ``RUNNING``.

        Args:
            trial_id: Identifier of the trial to submit.
            worker_id: Identifier of the worker that will execute it.

        Raises:
            KeyError: If *trial_id* is not an active trial.
            ValueError: If the trial is not in ``PENDING`` status.
        """
        trial = self._get_active_trial(trial_id)
        if trial.status != TrialStatus.PENDING:
            raise ValueError(
                f"Trial {trial_id} is {trial.status.value}, expected pending"
            )
        trial.status = TrialStatus.RUNNING
        trial.worker_id = worker_id

    def complete_trial(self, trial_id: str, result: dict[str, Any]) -> None:
        """Record successful trial completion.

        Transitions the trial to ``COMPLETED`` and moves it from
        the active set to the completed list.

        Args:
            trial_id: Identifier of the completed trial.
            result: Result dict (should include ``"objective"`` key).

        Raises:
            KeyError: If *trial_id* is not an active trial.
            ValueError: If the trial is not ``PENDING`` or ``RUNNING``.
        """
        trial = self._get_active_trial(trial_id)
        if trial.status not in (TrialStatus.PENDING, TrialStatus.RUNNING):
            raise ValueError(
                f"Trial {trial_id} is {trial.status.value}, "
                f"cannot complete"
            )
        trial.status = TrialStatus.COMPLETED
        trial.completed_at = time.time()
        trial.result = dict(result)
        self._move_to_completed(trial_id)

    def fail_trial(self, trial_id: str, reason: str = "") -> None:
        """Record trial failure.

        Transitions the trial to ``FAILED`` and moves it from the
        active set to the completed list.

        Args:
            trial_id: Identifier of the failed trial.
            reason: Human-readable failure description.

        Raises:
            KeyError: If *trial_id* is not an active trial.
            ValueError: If the trial is already terminal.
        """
        trial = self._get_active_trial(trial_id)
        if trial.is_terminal:
            raise ValueError(
                f"Trial {trial_id} is already {trial.status.value}"
            )
        trial.status = TrialStatus.FAILED
        trial.completed_at = time.time()
        trial.error = reason
        self._move_to_completed(trial_id)

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_pending_results(self) -> list[dict[str, Any]]:
        """Return completed trial results ready for model update.

        Each result dict merges the trial parameters with the result
        dict, adding ``_trial_id`` for provenance.

        Returns:
            List of merged param+result dicts for completed trials.
        """
        results: list[dict[str, Any]] = []
        for trial in self._completed_trials:
            if trial.status != TrialStatus.COMPLETED:
                continue
            if trial.result is None:
                continue
            merged: dict[str, Any] = dict(trial.params)
            merged.update(trial.result)
            merged["_trial_id"] = trial.trial_id
            results.append(merged)
        return results

    def get_pending_trials(self) -> list[AsyncTrial]:
        """Return active trials in ``PENDING`` status."""
        return [
            t for t in self._active_trials.values()
            if t.status == TrialStatus.PENDING
        ]

    def get_running_trials(self) -> list[AsyncTrial]:
        """Return active trials in ``RUNNING`` status."""
        return [
            t for t in self._active_trials.values()
            if t.status == TrialStatus.RUNNING
        ]

    def get_active_trials(self) -> list[AsyncTrial]:
        """Return all active (non-terminal) trials."""
        return list(self._active_trials.values())

    def get_completed_trials(self) -> list[AsyncTrial]:
        """Return all terminal (completed or failed) trials."""
        return list(self._completed_trials)

    def get_trial(self, trial_id: str) -> AsyncTrial | None:
        """Look up a trial by ID across active and completed sets.

        Returns:
            The :class:`AsyncTrial` if found, otherwise ``None``.
        """
        if trial_id in self._active_trials:
            return self._active_trials[trial_id]
        for trial in self._completed_trials:
            if trial.trial_id == trial_id:
                return trial
        return None

    # ------------------------------------------------------------------
    # Worker management
    # ------------------------------------------------------------------

    def count_idle_workers(self) -> int:
        """Count workers not currently running any trial.

        Returns:
            Number of idle workers (``n_workers`` minus running
            trial count).
        """
        running = sum(
            1 for t in self._active_trials.values()
            if t.status == TrialStatus.RUNNING
        )
        return max(0, self._n_workers - running)

    def count_running(self) -> int:
        """Count trials currently running."""
        return sum(
            1 for t in self._active_trials.values()
            if t.status == TrialStatus.RUNNING
        )

    def active_worker_ids(self) -> list[str]:
        """Return IDs of workers currently running trials."""
        return [
            t.worker_id
            for t in self._active_trials.values()
            if t.status == TrialStatus.RUNNING and t.worker_id is not None
        ]

    def needs_backfill(self) -> bool:
        """True when there are idle workers and no pending trials.

        In this state the optimizer should generate new suggestions
        to fill idle worker slots.
        """
        has_pending = any(
            t.status == TrialStatus.PENDING
            for t in self._active_trials.values()
        )
        return self.count_idle_workers() > 0 and not has_pending

    def backfill_count(self) -> int:
        """Number of new suggestions needed to fill idle workers.

        Takes into account pending trials that have not yet been
        submitted.
        """
        pending = sum(
            1 for t in self._active_trials.values()
            if t.status == TrialStatus.PENDING
        )
        idle = self.count_idle_workers()
        return max(0, idle - pending)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Summary statistics for the scheduler state.

        Returns:
            Dict with keys: ``n_active``, ``n_pending``, ``n_running``,
            ``n_completed``, ``n_failed``, ``n_idle_workers``,
            ``n_workers``, ``total_trials``, ``batch_strategy``.
        """
        n_pending = sum(
            1 for t in self._active_trials.values()
            if t.status == TrialStatus.PENDING
        )
        n_running = sum(
            1 for t in self._active_trials.values()
            if t.status == TrialStatus.RUNNING
        )
        n_completed = sum(
            1 for t in self._completed_trials
            if t.status == TrialStatus.COMPLETED
        )
        n_failed = sum(
            1 for t in self._completed_trials
            if t.status == TrialStatus.FAILED
        )
        return {
            "n_active": len(self._active_trials),
            "n_pending": n_pending,
            "n_running": n_running,
            "n_completed": n_completed,
            "n_failed": n_failed,
            "n_idle_workers": self.count_idle_workers(),
            "n_workers": self._n_workers,
            "total_trials": self.total_trials,
            "batch_strategy": self._strategy,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_trial(self, params: dict[str, Any]) -> AsyncTrial:
        """Create a new trial with a unique ID."""
        self._trial_counter += 1
        trial_id = f"trial_{self._trial_counter:04d}_{uuid.uuid4().hex[:8]}"
        return AsyncTrial(
            trial_id=trial_id,
            params=dict(params),
            status=TrialStatus.PENDING,
            submitted_at=time.time(),
        )

    def _get_active_trial(self, trial_id: str) -> AsyncTrial:
        """Look up an active trial, raising KeyError if missing."""
        if trial_id not in self._active_trials:
            raise KeyError(f"No active trial with id {trial_id!r}")
        return self._active_trials[trial_id]

    def _move_to_completed(self, trial_id: str) -> None:
        """Move a trial from active to completed."""
        trial = self._active_trials.pop(trial_id, None)
        if trial is not None:
            self._completed_trials.append(trial)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize scheduler state to a plain dict."""
        return {
            "n_workers": self._n_workers,
            "batch_strategy": self._strategy,
            "trial_counter": self._trial_counter,
            "active_trials": {
                tid: t.to_dict() for tid, t in self._active_trials.items()
            },
            "completed_trials": [t.to_dict() for t in self._completed_trials],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchScheduler:
        """Restore scheduler state from a dict produced by :meth:`to_dict`.

        Args:
            data: Serialized scheduler state dict.

        Returns:
            Restored :class:`BatchScheduler` instance.
        """
        scheduler = cls(
            n_workers=data.get("n_workers", 1),
            batch_strategy=data.get("batch_strategy", "simple"),
        )
        scheduler._trial_counter = data.get("trial_counter", 0)

        for tid, td in data.get("active_trials", {}).items():
            trial = AsyncTrial.from_dict(td)
            scheduler._active_trials[tid] = trial

        for td in data.get("completed_trials", []):
            trial = AsyncTrial.from_dict(td)
            scheduler._completed_trials.append(trial)

        return scheduler

    def __repr__(self) -> str:
        return (
            f"BatchScheduler(n_workers={self._n_workers}, "
            f"strategy={self._strategy!r}, "
            f"active={len(self._active_trials)}, "
            f"completed={len(self._completed_trials)})"
        )
