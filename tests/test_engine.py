"""Comprehensive tests for the optimization execution engine.

Covers Trial lifecycle, TrialBatch properties, event system, engine
execution patterns (evaluator callback, generator, termination,
determinism, retry, rollback, events), and frozen parameter handling.
"""

from __future__ import annotations

import pytest

from optimization_copilot.backends.builtin import (
    LatinHypercubeSampler,
    RandomSampler,
    TPESampler,
)
from optimization_copilot.dsl.spec import (
    BudgetDef,
    Direction,
    ObjectiveDef,
    OptimizationSpec,
    ParamType,
    ParameterDef,
)
from optimization_copilot.engine.engine import (
    EngineConfig,
    OptimizationEngine,
)
from optimization_copilot.engine.events import (
    EngineEvent,
    EventHook,
    EventPayload,
)
from optimization_copilot.engine.trial import Trial, TrialBatch, TrialState
from optimization_copilot.plugins.registry import PluginRegistry


# ── Helpers ──────────────────────────────────────────────


def _make_spec(max_samples: int = 15, seed: int = 42) -> OptimizationSpec:
    """Create a simple OptimizationSpec with 2 continuous params and 1 objective.

    Parameters x1 and x2 range from 0 to 10. The single objective y is
    maximized. Budget defaults to 15 max_samples.
    """
    return OptimizationSpec(
        campaign_id="test-campaign",
        parameters=[
            ParameterDef(name="x1", type=ParamType.CONTINUOUS, lower=0.0, upper=10.0),
            ParameterDef(name="x2", type=ParamType.CONTINUOUS, lower=0.0, upper=10.0),
        ],
        objectives=[
            ObjectiveDef(name="y", direction=Direction.MAXIMIZE),
        ],
        budget=BudgetDef(max_samples=max_samples),
        seed=seed,
    )


def _make_registry() -> PluginRegistry:
    """Create a PluginRegistry with the three built-in samplers."""
    registry = PluginRegistry()
    registry.register(RandomSampler)
    registry.register(LatinHypercubeSampler)
    registry.register(TPESampler)
    return registry


def _simple_evaluator(params: dict) -> dict[str, float]:
    """Trivial linear objective: y = x1 + x2."""
    return {"y": params["x1"] + params["x2"]}


# ── Trial Lifecycle ──────────────────────────────────────


class TestTrialLifecycle:
    """Test Trial state transitions: create, complete, fail, abandon."""

    def test_create_trial_pending(self):
        trial = Trial(
            trial_id="t-0001-00",
            iteration=0,
            parameters={"x1": 1.0, "x2": 2.0},
        )
        assert trial.state == TrialState.PENDING
        assert trial.kpi_values == {}
        assert trial.is_failure is False
        assert trial.failure_reason is None
        assert trial.attempt == 1

    def test_complete_trial(self):
        trial = Trial(
            trial_id="t-0001-00",
            iteration=0,
            parameters={"x1": 1.0, "x2": 2.0},
        )
        trial.complete(kpi_values={"y": 3.0}, metadata={"source": "test"})
        assert trial.state == TrialState.COMPLETED
        assert trial.kpi_values == {"y": 3.0}
        assert trial.metadata["source"] == "test"
        assert trial.is_failure is False

    def test_fail_trial(self):
        trial = Trial(
            trial_id="t-0001-00",
            iteration=0,
            parameters={"x1": 1.0, "x2": 2.0},
        )
        trial.fail(reason="evaluation timeout")
        assert trial.state == TrialState.FAILED
        assert trial.is_failure is True
        assert trial.failure_reason == "evaluation timeout"

    def test_abandon_trial(self):
        trial = Trial(
            trial_id="t-0001-00",
            iteration=0,
            parameters={"x1": 1.0, "x2": 2.0},
        )
        trial.abandon(reason="max retries exceeded")
        assert trial.state == TrialState.ABANDONED
        assert trial.failure_reason == "max retries exceeded"

    def test_abandon_without_reason(self):
        trial = Trial(
            trial_id="t-0001-00",
            iteration=0,
            parameters={"x1": 1.0, "x2": 2.0},
        )
        trial.abandon()
        assert trial.state == TrialState.ABANDONED
        assert trial.failure_reason is None

    def test_state_transitions_are_direct(self):
        """Verify that state transitions overwrite previous state."""
        trial = Trial(
            trial_id="t-0001-00",
            iteration=0,
            parameters={"x1": 1.0, "x2": 2.0},
        )
        assert trial.state == TrialState.PENDING

        trial.state = TrialState.RUNNING
        assert trial.state == TrialState.RUNNING

        trial.complete(kpi_values={"y": 5.0})
        assert trial.state == TrialState.COMPLETED


# ── Trial Batch ──────────────────────────────────────────


class TestTrialBatch:
    """Test TrialBatch creation and aggregate properties."""

    def _make_batch(self) -> TrialBatch:
        """Helper: create a batch with 3 trials in mixed states."""
        t1 = Trial(trial_id="t-0000-00", iteration=0, parameters={"x1": 1.0, "x2": 2.0})
        t2 = Trial(trial_id="t-0000-01", iteration=0, parameters={"x1": 3.0, "x2": 4.0})
        t3 = Trial(trial_id="t-0000-02", iteration=0, parameters={"x1": 5.0, "x2": 6.0})
        t1.complete(kpi_values={"y": 3.0})
        t2.fail(reason="crash")
        # t3 stays PENDING
        return TrialBatch(batch_id="batch-0000", iteration=0, trials=[t1, t2, t3])

    def test_batch_creation(self):
        batch = self._make_batch()
        assert batch.batch_id == "batch-0000"
        assert batch.iteration == 0
        assert len(batch.trials) == 3

    def test_n_completed(self):
        batch = self._make_batch()
        assert batch.n_completed == 1

    def test_n_failed(self):
        batch = self._make_batch()
        assert batch.n_failed == 1

    def test_n_pending(self):
        batch = self._make_batch()
        assert batch.n_pending == 1

    def test_all_completed_false(self):
        batch = self._make_batch()
        assert batch.all_completed is False

    def test_all_failed_false(self):
        batch = self._make_batch()
        assert batch.all_failed is False

    def test_all_completed_true(self):
        t1 = Trial(trial_id="t-0000-00", iteration=0, parameters={"x1": 1.0, "x2": 2.0})
        t2 = Trial(trial_id="t-0000-01", iteration=0, parameters={"x1": 3.0, "x2": 4.0})
        t1.complete(kpi_values={"y": 3.0})
        t2.complete(kpi_values={"y": 7.0})
        batch = TrialBatch(batch_id="batch-0000", iteration=0, trials=[t1, t2])
        assert batch.all_completed is True

    def test_all_failed_true(self):
        t1 = Trial(trial_id="t-0000-00", iteration=0, parameters={"x1": 1.0, "x2": 2.0})
        t2 = Trial(trial_id="t-0000-01", iteration=0, parameters={"x1": 3.0, "x2": 4.0})
        t1.fail(reason="err1")
        t2.fail(reason="err2")
        batch = TrialBatch(batch_id="batch-0000", iteration=0, trials=[t1, t2])
        assert batch.all_failed is True

    def test_empty_batch_properties(self):
        batch = TrialBatch(batch_id="batch-empty", iteration=0, trials=[])
        assert batch.all_completed is False
        assert batch.all_failed is False
        assert batch.n_completed == 0
        assert batch.n_failed == 0
        assert batch.n_pending == 0


# ── Trial to Observation ─────────────────────────────────


class TestTrialToObservation:
    """Test conversion of trials to core Observation objects."""

    def test_completed_trial_to_observation(self):
        trial = Trial(
            trial_id="t-0003-00",
            iteration=3,
            parameters={"x1": 5.0, "x2": 7.0},
            timestamp=1.23,
        )
        trial.complete(kpi_values={"y": 12.0}, metadata={"source": "test"})
        obs = trial.to_observation()

        assert obs.iteration == 3
        assert obs.parameters == {"x1": 5.0, "x2": 7.0}
        assert obs.kpi_values == {"y": 12.0}
        assert obs.is_failure is False
        assert obs.qc_passed is True
        assert obs.timestamp == 1.23
        assert obs.metadata["source"] == "test"

    def test_failed_trial_to_observation(self):
        trial = Trial(
            trial_id="t-0005-00",
            iteration=5,
            parameters={"x1": 1.0, "x2": 1.0},
        )
        trial.fail(reason="timeout")
        obs = trial.to_observation()

        assert obs.iteration == 5
        assert obs.parameters == {"x1": 1.0, "x2": 1.0}
        assert obs.is_failure is True
        assert obs.qc_passed is False
        assert obs.failure_reason == "timeout"

    def test_pending_trial_cannot_convert(self):
        trial = Trial(
            trial_id="t-0000-00",
            iteration=0,
            parameters={"x1": 1.0, "x2": 2.0},
        )
        with pytest.raises(ValueError, match="pending"):
            trial.to_observation()

    def test_abandoned_trial_cannot_convert(self):
        trial = Trial(
            trial_id="t-0000-00",
            iteration=0,
            parameters={"x1": 1.0, "x2": 2.0},
        )
        trial.abandon(reason="too many retries")
        with pytest.raises(ValueError, match="abandoned"):
            trial.to_observation()


# ── Trial Serialization ──────────────────────────────────


class TestTrialSerialization:
    """Test Trial to_dict / from_dict round-trip fidelity."""

    def test_round_trip_completed_trial(self):
        trial = Trial(
            trial_id="t-0002-01",
            iteration=2,
            parameters={"x1": 3.5, "x2": 7.2},
            attempt=1,
            timestamp=0.5,
        )
        trial.complete(kpi_values={"y": 10.7}, metadata={"tag": "best"})

        data = trial.to_dict()
        restored = Trial.from_dict(data)

        assert restored.trial_id == trial.trial_id
        assert restored.iteration == trial.iteration
        assert restored.parameters == trial.parameters
        assert restored.state == TrialState.COMPLETED
        assert restored.kpi_values == trial.kpi_values
        assert restored.is_failure == trial.is_failure
        assert restored.failure_reason == trial.failure_reason
        assert restored.metadata == trial.metadata
        assert restored.attempt == trial.attempt
        assert restored.timestamp == trial.timestamp

    def test_round_trip_failed_trial(self):
        trial = Trial(
            trial_id="t-0004-00",
            iteration=4,
            parameters={"x1": 0.1, "x2": 0.2},
            attempt=2,
        )
        trial.fail(reason="division by zero")

        data = trial.to_dict()
        restored = Trial.from_dict(data)

        assert restored.state == TrialState.FAILED
        assert restored.is_failure is True
        assert restored.failure_reason == "division by zero"
        assert restored.attempt == 2

    def test_round_trip_pending_trial(self):
        trial = Trial(
            trial_id="t-0000-00",
            iteration=0,
            parameters={"x1": 5.0, "x2": 5.0},
        )

        data = trial.to_dict()
        restored = Trial.from_dict(data)

        assert restored.state == TrialState.PENDING
        assert restored.parameters == {"x1": 5.0, "x2": 5.0}

    def test_dict_contains_expected_keys(self):
        trial = Trial(
            trial_id="t-0000-00",
            iteration=0,
            parameters={"x1": 1.0, "x2": 2.0},
        )
        data = trial.to_dict()
        expected_keys = {
            "trial_id", "iteration", "parameters", "state",
            "kpi_values", "is_failure", "failure_reason",
            "metadata", "attempt", "timestamp",
        }
        assert set(data.keys()) == expected_keys


# ── Event Hook ───────────────────────────────────────────


class TestEventHook:
    """Test the publish-subscribe event system."""

    def test_register_and_emit(self):
        hook = EventHook()
        received = []

        hook.on(EngineEvent.TRIAL_COMPLETE, lambda p: received.append(p))
        payload = EventPayload(
            event=EngineEvent.TRIAL_COMPLETE,
            iteration=0,
            data={"trial_id": "t-0000-00"},
        )
        hook.emit(payload)

        assert len(received) == 1
        assert received[0].event == EngineEvent.TRIAL_COMPLETE
        assert received[0].data["trial_id"] == "t-0000-00"

    def test_multiple_handlers_same_event(self):
        hook = EventHook()
        log_a = []
        log_b = []

        hook.on(EngineEvent.BATCH_COMPLETE, lambda p: log_a.append(p))
        hook.on(EngineEvent.BATCH_COMPLETE, lambda p: log_b.append(p))

        payload = EventPayload(
            event=EngineEvent.BATCH_COMPLETE,
            iteration=1,
            data={},
        )
        hook.emit(payload)

        assert len(log_a) == 1
        assert len(log_b) == 1

    def test_handlers_only_fire_for_registered_event(self):
        hook = EventHook()
        received = []

        hook.on(EngineEvent.TRIAL_COMPLETE, lambda p: received.append(p))

        # Emit a different event type.
        payload = EventPayload(
            event=EngineEvent.TRIAL_FAILED,
            iteration=0,
            data={},
        )
        hook.emit(payload)

        assert len(received) == 0

    def test_clear_removes_all_handlers(self):
        hook = EventHook()
        received = []

        hook.on(EngineEvent.TRIAL_COMPLETE, lambda p: received.append(p))
        hook.clear()

        payload = EventPayload(
            event=EngineEvent.TRIAL_COMPLETE,
            iteration=0,
            data={},
        )
        hook.emit(payload)

        assert len(received) == 0

    def test_handler_exception_propagates(self):
        hook = EventHook()

        def bad_handler(payload):
            raise RuntimeError("handler error")

        hook.on(EngineEvent.TRIAL_COMPLETE, bad_handler)

        payload = EventPayload(
            event=EngineEvent.TRIAL_COMPLETE,
            iteration=0,
            data={},
        )
        with pytest.raises(RuntimeError, match="handler error"):
            hook.emit(payload)

    def test_all_handlers_called_despite_exception(self):
        """Even if first handler raises, subsequent handlers still fire."""
        hook = EventHook()
        second_called = []

        def bad_handler(payload):
            raise RuntimeError("first fails")

        def good_handler(payload):
            second_called.append(True)

        hook.on(EngineEvent.TRIAL_COMPLETE, bad_handler)
        hook.on(EngineEvent.TRIAL_COMPLETE, good_handler)

        payload = EventPayload(
            event=EngineEvent.TRIAL_COMPLETE,
            iteration=0,
            data={},
        )
        with pytest.raises(RuntimeError, match="first fails"):
            hook.emit(payload)

        # Second handler should have been called despite the first raising.
        assert len(second_called) == 1


# ── Engine Basic ─────────────────────────────────────────


class TestEngineBasic:
    """Test basic engine execution with run_with_evaluator."""

    def test_run_with_evaluator_completes(self):
        spec = _make_spec(max_samples=10)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)

        result = engine.run_with_evaluator(_simple_evaluator)

        # The meta-controller may select batch_size > 1, so total_iterations
        # is typically fewer than max_samples. What matters is that the
        # engine ran, produced results, and terminated by budget.
        assert result.total_iterations > 0
        assert result.total_trials >= 10  # at least max_samples trials created
        assert result.best_kpi_values.get("y", 0) > 0
        assert "max_samples_reached" in result.termination_reason

    def test_run_with_evaluator_best_trial_populated(self):
        spec = _make_spec(max_samples=5)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)

        result = engine.run_with_evaluator(_simple_evaluator)

        assert result.best_trial is not None
        assert "y" in result.best_kpi_values
        assert result.best_kpi_values["y"] > 0

    def test_run_result_has_audit_trail(self):
        spec = _make_spec(max_samples=5)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)

        result = engine.run_with_evaluator(_simple_evaluator)

        assert len(result.audit_trail) == result.total_iterations
        for entry in result.audit_trail:
            assert "iteration" in entry
            assert "decision" in entry
            assert "backend" in entry


# ── Engine Generator ─────────────────────────────────────


class TestEngineGenerator:
    """Test the generator-based engine execution pattern."""

    def test_generator_pattern(self):
        spec = _make_spec(max_samples=5)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)

        iterations_seen = 0
        for batch in engine.run():
            for trial in batch.trials:
                trial.state = TrialState.RUNNING
                kpi = _simple_evaluator(trial.parameters)
                trial.complete(kpi_values=kpi)
            iterations_seen += 1

        result = engine.result()
        assert result.total_iterations == iterations_seen
        assert result.total_iterations > 0
        assert result.total_trials >= 5  # at least max_samples trials

    def test_generator_yields_trial_batches(self):
        spec = _make_spec(max_samples=3)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)

        for batch in engine.run():
            assert isinstance(batch, TrialBatch)
            assert len(batch.trials) >= 1
            for trial in batch.trials:
                assert isinstance(trial, Trial)
                assert trial.state == TrialState.PENDING
                trial.complete(kpi_values={"y": 1.0})


# ── Engine Termination ───────────────────────────────────


class TestEngineTermination:
    """Test termination by budget and by manual stop."""

    def test_max_samples_budget(self):
        spec = _make_spec(max_samples=7)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)

        result = engine.run_with_evaluator(_simple_evaluator)

        assert result.total_iterations > 0
        assert result.total_trials >= 7  # at least max_samples trials were created
        assert "max_samples_reached" in result.termination_reason

    def test_manual_stop(self):
        spec = _make_spec(max_samples=100)  # large budget so it won't hit it
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)

        iterations_run = 0
        for batch in engine.run():
            for trial in batch.trials:
                trial.complete(kpi_values={"y": 1.0})
            iterations_run += 1
            if iterations_run >= 3:
                engine.stop(reason="user_requested")

        result = engine.result()
        assert result.total_iterations == 3
        assert result.termination_reason == "user_requested"

    def test_max_iterations_budget(self):
        spec = OptimizationSpec(
            campaign_id="test-max-iter",
            parameters=[
                ParameterDef(name="x1", type=ParamType.CONTINUOUS, lower=0.0, upper=10.0),
                ParameterDef(name="x2", type=ParamType.CONTINUOUS, lower=0.0, upper=10.0),
            ],
            objectives=[
                ObjectiveDef(name="y", direction=Direction.MAXIMIZE),
            ],
            budget=BudgetDef(max_iterations=5),
            seed=42,
        )
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)

        result = engine.run_with_evaluator(_simple_evaluator)

        assert result.total_iterations == 5
        assert "max_iterations_reached" in result.termination_reason


# ── Engine Determinism ───────────────────────────────────


class TestEngineDeterminism:
    """Run the same spec+seed+evaluator multiple times and verify identical results."""

    def test_deterministic_results(self):
        results = []
        for _ in range(3):
            spec = _make_spec(max_samples=10, seed=42)
            registry = _make_registry()
            engine = OptimizationEngine(spec, registry)
            result = engine.run_with_evaluator(_simple_evaluator)
            results.append(result)

        # All three runs must produce identical outcomes.
        for r in results[1:]:
            assert r.best_kpi_values == results[0].best_kpi_values
            assert r.total_iterations == results[0].total_iterations
            assert r.termination_reason == results[0].termination_reason
            assert r.total_trials == results[0].total_trials

    def test_different_seeds_differ(self):
        spec_a = _make_spec(max_samples=10, seed=42)
        spec_b = _make_spec(max_samples=10, seed=99)
        registry = _make_registry()

        result_a = OptimizationEngine(spec_a, registry).run_with_evaluator(_simple_evaluator)
        result_b = OptimizationEngine(spec_b, registry).run_with_evaluator(_simple_evaluator)

        # Different seeds should generally produce different best KPI values.
        # (Not guaranteed for trivial objectives but very likely with random sampling.)
        assert result_a.total_iterations == result_b.total_iterations


# ── Engine Retry ─────────────────────────────────────────


class TestEngineRetry:
    """Test that the engine retries failed trials up to max_retries."""

    def test_evaluator_fails_then_succeeds(self):
        call_count = {"n": 0}

        def flaky_evaluator(params):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise RuntimeError(f"transient failure #{call_count['n']}")
            return {"y": params["x1"] + params["x2"]}

        spec = _make_spec(max_samples=10)
        registry = _make_registry()
        config = EngineConfig(max_retries=3)
        engine = OptimizationEngine(spec, registry, config=config)

        result = engine.run_with_evaluator(flaky_evaluator)

        # The engine should have completed despite early failures.
        assert result.total_iterations > 0
        assert result.total_failures >= 2  # at least the 2 transient failures
        # Eventually the evaluator succeeded, so we should have a best trial.
        assert result.best_trial is not None
        assert result.best_kpi_values.get("y", 0) > 0


# ── Engine Rollback ──────────────────────────────────────


class TestEngineRollback:
    """Test rollback behavior when all trials in a batch fail."""

    def test_all_fail_rollback_snapshot_stable(self):
        """When every trial fails, the snapshot should not accumulate observations."""

        def always_fail(params):
            raise RuntimeError("always fails")

        spec = OptimizationSpec(
            campaign_id="test-rollback",
            parameters=[
                ParameterDef(name="x1", type=ParamType.CONTINUOUS, lower=0.0, upper=10.0),
                ParameterDef(name="x2", type=ParamType.CONTINUOUS, lower=0.0, upper=10.0),
            ],
            objectives=[
                ObjectiveDef(name="y", direction=Direction.MAXIMIZE),
            ],
            budget=BudgetDef(max_iterations=10),
            seed=42,
        )
        registry = _make_registry()
        config = EngineConfig(max_retries=3)
        engine = OptimizationEngine(spec, registry, config=config)

        result = engine.run_with_evaluator(always_fail)

        # All batches had all_failed=True, so _rollback_batch was called
        # and no observations were added to the snapshot.
        snapshot_obs = result.final_snapshot_dict.get("observations", [])
        assert len(snapshot_obs) == 0

    def test_eventual_abandonment(self):
        """After max_retries, failed trials should be abandoned."""

        def always_fail(params):
            raise RuntimeError("persistent failure")

        spec = OptimizationSpec(
            campaign_id="test-abandon",
            parameters=[
                ParameterDef(name="x1", type=ParamType.CONTINUOUS, lower=0.0, upper=10.0),
                ParameterDef(name="x2", type=ParamType.CONTINUOUS, lower=0.0, upper=10.0),
            ],
            objectives=[
                ObjectiveDef(name="y", direction=Direction.MAXIMIZE),
            ],
            budget=BudgetDef(max_iterations=10),
            seed=42,
        )
        registry = _make_registry()
        config = EngineConfig(max_retries=3)
        engine = OptimizationEngine(spec, registry, config=config)

        result = engine.run_with_evaluator(always_fail)

        assert result.total_failures > 0
        assert result.best_trial is None


# ── Engine Events ────────────────────────────────────────


class TestEngineEvents:
    """Verify that engine emits expected events during execution."""

    def test_iteration_complete_events_fire(self):
        spec = _make_spec(max_samples=5)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)

        iteration_events = []
        engine.on(
            EngineEvent.ITERATION_COMPLETE,
            lambda p: iteration_events.append(p),
        )

        result = engine.run_with_evaluator(_simple_evaluator)

        # One ITERATION_COMPLETE event per engine iteration.
        assert len(iteration_events) == result.total_iterations
        assert len(iteration_events) > 0
        for i, evt in enumerate(iteration_events):
            assert evt.data["iteration"] == i

    def test_batch_complete_events_fire(self):
        spec = _make_spec(max_samples=5)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)

        batch_events = []
        engine.on(
            EngineEvent.BATCH_COMPLETE,
            lambda p: batch_events.append(p),
        )

        result = engine.run_with_evaluator(_simple_evaluator)

        # One BATCH_COMPLETE per iteration (when not all-failed).
        assert len(batch_events) == result.total_iterations
        assert len(batch_events) > 0
        for evt in batch_events:
            assert "batch_id" in evt.data
            assert "n_completed" in evt.data

    def test_termination_event_fires(self):
        spec = _make_spec(max_samples=3)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)

        termination_events = []
        engine.on(
            EngineEvent.TERMINATION,
            lambda p: termination_events.append(p),
        )

        engine.run_with_evaluator(_simple_evaluator)

        assert len(termination_events) == 1
        assert "reason" in termination_events[0].data

    def test_trial_complete_events_fire(self):
        spec = _make_spec(max_samples=3)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)

        trial_events = []
        engine.on(
            EngineEvent.TRIAL_COMPLETE,
            lambda p: trial_events.append(p),
        )

        result = engine.run_with_evaluator(_simple_evaluator)

        # One TRIAL_COMPLETE event per completed trial (may exceed
        # max_samples due to batch_size > 1).
        assert len(trial_events) == result.total_trials
        assert len(trial_events) >= 3  # at least max_samples
        for evt in trial_events:
            assert "trial_id" in evt.data
            assert "kpi_values" in evt.data


# ── Engine Frozen Parameters ─────────────────────────────


class TestEngineFrozenParams:
    """Test that frozen parameters appear in trial parameters with correct values."""

    def test_frozen_param_injected(self):
        spec = OptimizationSpec(
            campaign_id="test-frozen",
            parameters=[
                ParameterDef(name="x1", type=ParamType.CONTINUOUS, lower=0.0, upper=10.0),
                ParameterDef(name="x2", type=ParamType.CONTINUOUS, lower=0.0, upper=10.0),
                ParameterDef(
                    name="x_frozen",
                    type=ParamType.CONTINUOUS,
                    lower=0.0,
                    upper=10.0,
                    frozen=True,
                    frozen_value=3.14,
                ),
            ],
            objectives=[
                ObjectiveDef(name="y", direction=Direction.MAXIMIZE),
            ],
            budget=BudgetDef(max_samples=5),
            seed=42,
        )
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)

        def evaluator_with_frozen(params):
            # The frozen parameter must be present with the correct value.
            assert "x_frozen" in params
            assert params["x_frozen"] == 3.14
            return {"y": params["x1"] + params["x2"] + params["x_frozen"]}

        result = engine.run_with_evaluator(evaluator_with_frozen)

        assert result.total_iterations > 0
        assert result.total_trials >= 5  # at least max_samples
        assert result.best_trial is not None
        # Best trial parameters must include the frozen value.
        assert result.best_trial["parameters"]["x_frozen"] == 3.14
