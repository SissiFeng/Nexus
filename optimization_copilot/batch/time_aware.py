"""Time-aware scheduling to minimize wall-clock time.

Groups experiments into batches that respect parallelism limits and
uses greedy bin-packing (longest-job-first) to balance wall-clock
time across batches.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TimeEstimate:
    """Estimated duration for a candidate trial.

    Attributes
    ----------
    trial_id:
        Optional reference to an existing trial.
    parameters:
        Parameter configuration for the candidate.
    estimated_duration:
        Estimated wall-clock duration in seconds.
    variance:
        Variance of the duration estimate.
    """

    trial_id: str | None = None
    parameters: dict[str, float] = field(default_factory=dict)
    estimated_duration: float = 1.0
    variance: float = 0.0


@dataclass
class ScheduledBatch:
    """A batch of trials scheduled for parallel execution.

    Attributes
    ----------
    batch_index:
        Zero-based batch ordinal.
    trials:
        Time estimates for the trials in this batch.
    estimated_wall_time:
        Maximum duration across trials in this batch (determines
        when the batch finishes).
    estimated_total_time:
        Sum of all individual trial durations in this batch.
    """

    batch_index: int
    trials: list[TimeEstimate]
    estimated_wall_time: float
    estimated_total_time: float


class TimeAwareScheduler:
    """Schedule candidates into batches to minimize total wall-clock time.

    Uses a greedy longest-job-first strategy: sort candidates by
    estimated duration (descending), then fill each batch up to
    *max_parallel* slots. An optional deadline stops scheduling when
    cumulative wall time would be exceeded.

    Parameters
    ----------
    max_parallel:
        Maximum number of trials that run concurrently in a single
        batch.
    """

    def __init__(self, max_parallel: int = 4) -> None:
        self._max_parallel = max_parallel

    def schedule(
        self,
        candidates: list[TimeEstimate],
        deadline: float | None = None,
    ) -> list[ScheduledBatch]:
        """Group candidates into time-balanced batches.

        Strategy:
        1. Sort candidates by estimated_duration descending (longest first).
        2. Fill batches of size max_parallel greedily.
        3. If a deadline is set, stop when adding another batch would
           exceed the cumulative wall time.

        Parameters
        ----------
        candidates:
            List of time estimates to schedule.
        deadline:
            Optional wall-time budget in seconds. Batches are added
            only while the cumulative wall time stays within this limit.

        Returns
        -------
        list[ScheduledBatch]
            Ordered list of scheduled batches.
        """
        if not candidates:
            return []

        # Sort by estimated_duration descending (longest jobs first)
        sorted_candidates = sorted(
            candidates,
            key=lambda te: te.estimated_duration,
            reverse=True,
        )

        batches: list[ScheduledBatch] = []
        cumulative_wall = 0.0
        idx = 0

        while idx < len(sorted_candidates):
            batch_trials = sorted_candidates[idx : idx + self._max_parallel]
            idx += self._max_parallel

            wall_time = max(te.estimated_duration for te in batch_trials)
            total_time = sum(te.estimated_duration for te in batch_trials)

            # Check deadline constraint before adding this batch
            if deadline is not None and cumulative_wall + wall_time > deadline:
                break

            batch = ScheduledBatch(
                batch_index=len(batches),
                trials=batch_trials,
                estimated_wall_time=wall_time,
                estimated_total_time=total_time,
            )
            batches.append(batch)
            cumulative_wall += wall_time

        return batches

    def estimated_total_wall_time(self, batches: list[ScheduledBatch]) -> float:
        """Sum wall times across all batches.

        Parameters
        ----------
        batches:
            List of scheduled batches.

        Returns
        -------
        float
            Total estimated wall-clock time to execute all batches
            sequentially.
        """
        return sum(b.estimated_wall_time for b in batches)
