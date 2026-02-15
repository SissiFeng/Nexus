"""Tests for optimization_copilot.batch package enhancements."""

from __future__ import annotations

import pytest

from optimization_copilot.batch import (
    TrialState,
    BatchAsyncTrial,
    AsyncBatchExecutor,
    TimeEstimate,
    ScheduledBatch,
    TimeAwareScheduler,
)


# ---------------------------------------------------------------------------
# TrialState tests
# ---------------------------------------------------------------------------


class TestTrialState:
    """Tests for the TrialState enum."""

    def test_enum_values(self) -> None:
        assert TrialState.PENDING.value == "pending"
        assert TrialState.RUNNING.value == "running"
        assert TrialState.COMPLETED.value == "completed"
        assert TrialState.FAILED.value == "failed"
        assert TrialState.CANCELLED.value == "cancelled"


# ---------------------------------------------------------------------------
# AsyncBatchExecutor tests
# ---------------------------------------------------------------------------


class TestAsyncBatchExecutor:
    """Tests for AsyncBatchExecutor lifecycle and queries."""

    def test_submit_returns_unique_trial_ids(self) -> None:
        executor = AsyncBatchExecutor()
        id1 = executor.submit({"x": 1.0})
        id2 = executor.submit({"x": 2.0})
        assert isinstance(id1, str)
        assert isinstance(id2, str)
        assert id1 != id2

    def test_submit_complete_poll_lifecycle(self) -> None:
        executor = AsyncBatchExecutor()
        tid = executor.submit({"x": 1.0})
        # Before completion, poll returns nothing
        assert executor.poll() == []
        # Complete the trial
        executor.complete(tid, {"objective": 0.5})
        # Poll returns the completed trial
        completed = executor.poll()
        assert len(completed) == 1
        assert completed[0].trial_id == tid
        assert completed[0].result == {"objective": 0.5}
        assert completed[0].state == TrialState.COMPLETED

    def test_poll_returns_each_completed_trial_only_once(self) -> None:
        executor = AsyncBatchExecutor()
        tid = executor.submit({"x": 1.0})
        executor.complete(tid, {"objective": 0.5})
        first_poll = executor.poll()
        assert len(first_poll) == 1
        # Second poll should return empty
        second_poll = executor.poll()
        assert len(second_poll) == 0

    def test_fail_marks_trial_as_failed(self) -> None:
        executor = AsyncBatchExecutor()
        tid = executor.submit({"x": 1.0})
        executor.fail(tid, error="boom")
        trial = executor._trials[tid]
        assert trial.state == TrialState.FAILED
        assert trial.error == "boom"

    def test_cancel_marks_trial_as_cancelled(self) -> None:
        executor = AsyncBatchExecutor()
        tid = executor.submit({"x": 1.0})
        executor.cancel(tid)
        trial = executor._trials[tid]
        assert trial.state == TrialState.CANCELLED

    def test_pending_count_tracks_active_trials(self) -> None:
        executor = AsyncBatchExecutor()
        assert executor.pending_count() == 0
        tid1 = executor.submit({"x": 1.0})
        tid2 = executor.submit({"x": 2.0})
        assert executor.pending_count() == 2
        executor.complete(tid1, {"objective": 0.5})
        assert executor.pending_count() == 1

    def test_active_trials_returns_pending_and_running_only(self) -> None:
        executor = AsyncBatchExecutor()
        tid1 = executor.submit({"x": 1.0})
        tid2 = executor.submit({"x": 2.0})
        executor.complete(tid2, {"objective": 0.5})
        active = executor.active_trials()
        active_ids = {t.trial_id for t in active}
        assert tid1 in active_ids
        assert tid2 not in active_ids

    def test_completed_trials_returns_all_completed(self) -> None:
        executor = AsyncBatchExecutor()
        tid1 = executor.submit({"x": 1.0})
        tid2 = executor.submit({"x": 2.0})
        tid3 = executor.submit({"x": 3.0})
        executor.complete(tid1, {"objective": 0.1})
        executor.complete(tid3, {"objective": 0.3})
        completed = executor.completed_trials()
        completed_ids = {t.trial_id for t in completed}
        assert completed_ids == {tid1, tid3}

    def test_incorporate_partial_with_score_fn_reorders(self) -> None:
        executor = AsyncBatchExecutor()
        tid = executor.submit({"x": 1.0})
        executor.complete(tid, {"objective": 0.5})
        completed = executor.completed_trials()

        candidates = [{"x": 3.0}, {"x": 1.0}, {"x": 2.0}]

        def score_fn(candidate: dict, trials: list) -> float:
            return candidate["x"]

        reordered = executor.incorporate_partial(completed, candidates, score_fn)
        assert [c["x"] for c in reordered] == [1.0, 2.0, 3.0]

    def test_incorporate_partial_without_score_fn_returns_unchanged(self) -> None:
        executor = AsyncBatchExecutor()
        candidates = [{"x": 3.0}, {"x": 1.0}, {"x": 2.0}]
        result = executor.incorporate_partial([], candidates, score_fn=None)
        assert [c["x"] for c in result] == [3.0, 1.0, 2.0]


# ---------------------------------------------------------------------------
# TimeEstimate tests
# ---------------------------------------------------------------------------


class TestTimeEstimate:
    """Tests for TimeEstimate dataclass."""

    def test_creation_with_defaults(self) -> None:
        te = TimeEstimate()
        assert te.trial_id is None
        assert te.parameters == {}
        assert te.estimated_duration == 1.0
        assert te.variance == 0.0


# ---------------------------------------------------------------------------
# TimeAwareScheduler tests
# ---------------------------------------------------------------------------


class TestTimeAwareScheduler:
    """Tests for TimeAwareScheduler scheduling logic."""

    def test_schedule_groups_into_batches(self) -> None:
        scheduler = TimeAwareScheduler(max_parallel=2)
        candidates = [
            TimeEstimate(estimated_duration=10.0),
            TimeEstimate(estimated_duration=5.0),
            TimeEstimate(estimated_duration=8.0),
            TimeEstimate(estimated_duration=3.0),
        ]
        batches = scheduler.schedule(candidates)
        assert len(batches) == 2
        # Longest-first sorting: [10, 8, 5, 3]
        # Batch 0: [10, 8], Batch 1: [5, 3]
        assert len(batches[0].trials) == 2
        assert len(batches[1].trials) == 2

    def test_schedule_respects_max_parallel(self) -> None:
        scheduler = TimeAwareScheduler(max_parallel=3)
        candidates = [
            TimeEstimate(estimated_duration=1.0) for _ in range(7)
        ]
        batches = scheduler.schedule(candidates)
        # 7 candidates, max 3 per batch -> 3 batches (3, 3, 1)
        assert len(batches) == 3
        assert len(batches[0].trials) == 3
        assert len(batches[1].trials) == 3
        assert len(batches[2].trials) == 1

    def test_schedule_with_deadline_stops_early(self) -> None:
        scheduler = TimeAwareScheduler(max_parallel=1)
        candidates = [
            TimeEstimate(estimated_duration=10.0),
            TimeEstimate(estimated_duration=10.0),
            TimeEstimate(estimated_duration=10.0),
        ]
        # Deadline of 15s: only the first batch (10s) fits
        batches = scheduler.schedule(candidates, deadline=15.0)
        assert len(batches) == 1

    def test_estimated_total_wall_time_sums_batch_wall_times(self) -> None:
        scheduler = TimeAwareScheduler(max_parallel=2)
        candidates = [
            TimeEstimate(estimated_duration=10.0),
            TimeEstimate(estimated_duration=5.0),
            TimeEstimate(estimated_duration=8.0),
            TimeEstimate(estimated_duration=3.0),
        ]
        batches = scheduler.schedule(candidates)
        total = scheduler.estimated_total_wall_time(batches)
        # Batch 0 wall = 10.0, Batch 1 wall = 5.0 -> total = 15.0
        assert total == pytest.approx(15.0)
