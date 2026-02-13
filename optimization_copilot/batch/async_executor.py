"""Asynchronous batch execution with partial result incorporation.

PipeBO-style async executor that manages concurrent trial execution,
tracks trial lifecycles, and supports re-ranking candidates based on
partial results as they arrive.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class TrialState(Enum):
    """Lifecycle states for an async trial."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AsyncTrial:
    """A single asynchronous optimization trial.

    Attributes
    ----------
    trial_id:
        Unique identifier for this trial.
    parameters:
        Parameter configuration for evaluation.
    state:
        Current lifecycle state.
    submitted_at:
        Timestamp when the trial was created, or None if not yet submitted.
    completed_at:
        Timestamp when the trial finished, or None if still active.
    result:
        Result dict returned on completion (e.g. {"objective": 0.5}).
    duration_estimate:
        Estimated wall-clock duration in seconds, or None if unknown.
    error:
        Error description if the trial failed.
    """

    trial_id: str
    parameters: dict[str, float]
    state: TrialState = TrialState.PENDING
    submitted_at: float | None = None
    completed_at: float | None = None
    result: dict[str, float] | None = None
    duration_estimate: float | None = None
    error: str = ""


class AsyncBatchExecutor:
    """Asynchronous batch executor with partial result incorporation.

    Manages concurrent trial submissions and tracks their lifecycle.
    Supports polling for completed trials and re-ranking candidates
    based on partial results (PipeBO-style).

    Parameters
    ----------
    max_concurrent:
        Maximum number of trials that can be pending or running
        simultaneously.
    """

    def __init__(self, max_concurrent: int = 4) -> None:
        self._max_concurrent = max_concurrent
        self._trials: dict[str, AsyncTrial] = {}
        self._polled: set[str] = set()

    # -- submission --------------------------------------------------------

    def submit(
        self,
        params: dict[str, float],
        duration_estimate: float | None = None,
    ) -> str:
        """Create and register a new trial.

        Parameters
        ----------
        params:
            Parameter configuration to evaluate.
        duration_estimate:
            Optional estimated duration in seconds.

        Returns
        -------
        str
            The unique trial ID (12-char hex string).
        """
        trial_id = uuid.uuid4().hex[:12]
        trial = AsyncTrial(
            trial_id=trial_id,
            parameters=dict(params),
            state=TrialState.PENDING,
            submitted_at=time.time(),
            duration_estimate=duration_estimate,
        )
        self._trials[trial_id] = trial
        return trial_id

    # -- lifecycle transitions ---------------------------------------------

    def complete(self, trial_id: str, result: dict[str, float]) -> None:
        """Mark a trial as completed with its result.

        Parameters
        ----------
        trial_id:
            ID of the trial to complete.
        result:
            Result dict (e.g. {"objective": 0.5}).

        Raises
        ------
        KeyError
            If trial_id is not found.
        """
        trial = self._get_trial(trial_id)
        trial.state = TrialState.COMPLETED
        trial.completed_at = time.time()
        trial.result = dict(result)

    def fail(self, trial_id: str, error: str = "") -> None:
        """Mark a trial as failed.

        Parameters
        ----------
        trial_id:
            ID of the trial to fail.
        error:
            Description of the failure.

        Raises
        ------
        KeyError
            If trial_id is not found.
        """
        trial = self._get_trial(trial_id)
        trial.state = TrialState.FAILED
        trial.completed_at = time.time()
        trial.error = error

    def cancel(self, trial_id: str) -> None:
        """Mark a trial as cancelled.

        Parameters
        ----------
        trial_id:
            ID of the trial to cancel.

        Raises
        ------
        KeyError
            If trial_id is not found.
        """
        trial = self._get_trial(trial_id)
        trial.state = TrialState.CANCELLED
        trial.completed_at = time.time()

    # -- query methods -----------------------------------------------------

    def poll(self) -> list[AsyncTrial]:
        """Return newly completed trials not yet retrieved by poll.

        Each completed trial is returned at most once by successive
        poll calls.

        Returns
        -------
        list[AsyncTrial]
            Completed trials that have not been polled before.
        """
        newly_completed: list[AsyncTrial] = []
        for trial in self._trials.values():
            if trial.state == TrialState.COMPLETED and trial.trial_id not in self._polled:
                newly_completed.append(trial)
                self._polled.add(trial.trial_id)
        return newly_completed

    def pending_count(self) -> int:
        """Return the number of pending or running trials."""
        return sum(
            1 for t in self._trials.values()
            if t.state in (TrialState.PENDING, TrialState.RUNNING)
        )

    def active_trials(self) -> list[AsyncTrial]:
        """Return trials that are PENDING or RUNNING."""
        return [
            t for t in self._trials.values()
            if t.state in (TrialState.PENDING, TrialState.RUNNING)
        ]

    def completed_trials(self) -> list[AsyncTrial]:
        """Return all completed trials (regardless of poll status)."""
        return [
            t for t in self._trials.values()
            if t.state == TrialState.COMPLETED
        ]

    # -- partial result incorporation --------------------------------------

    def incorporate_partial(
        self,
        completed: list[AsyncTrial],
        candidates: list[dict[str, float]],
        score_fn: Callable[..., float] | None = None,
    ) -> list[dict[str, float]]:
        """Re-rank candidates based on completed partial results.

        If *score_fn* is provided, each candidate is scored and the
        list is returned sorted by score in ascending order (lower is
        better). If *score_fn* is ``None``, candidates are returned
        unchanged.

        Parameters
        ----------
        completed:
            Completed trials whose results inform the re-ranking.
        candidates:
            Candidate parameter configurations to re-rank.
        score_fn:
            Callable that takes ``(candidate_params, completed_trials)``
            and returns a float score. Lower scores are preferred.

        Returns
        -------
        list[dict[str, float]]
            Re-ranked candidate list (ascending score order).
        """
        if score_fn is None:
            return list(candidates)

        scored = [(score_fn(c, completed), c) for c in candidates]
        scored.sort(key=lambda pair: pair[0])
        return [c for _, c in scored]

    # -- internal helpers --------------------------------------------------

    def _get_trial(self, trial_id: str) -> AsyncTrial:
        """Look up a trial by ID, raising KeyError if missing."""
        if trial_id not in self._trials:
            raise KeyError(f"No trial with id {trial_id!r}")
        return self._trials[trial_id]
