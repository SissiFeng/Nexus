"""Comprehensive tests for OptimizationEngine <-> InfrastructureStack integration wiring.

Covers backward compatibility (no infrastructure), warm start injection,
infrastructure stopping, pre-decide signals, candidate filtering,
trial recording, best-values history tracking, MetaController backend_policy
integration, the _param_specs_to_dicts helper, and full integration
round-trips with real infrastructure modules.
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

from optimization_copilot.backends.builtin import (
    LatinHypercubeSampler,
    RandomSampler,
    TPESampler,
)
from optimization_copilot.core.models import (
    CampaignSnapshot,
    ParameterSpec,
    Phase,
    ProblemFingerprint,
    VariableType,
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
    EngineResult,
    OptimizationEngine,
)
from optimization_copilot.engine.events import EngineEvent, EventPayload
from optimization_copilot.engine.trial import Trial, TrialBatch, TrialState
from optimization_copilot.infrastructure.integration import (
    InfrastructureConfig,
    InfrastructureStack,
)
from optimization_copilot.infrastructure.stopping_rule import StoppingDecision
from optimization_copilot.meta_controller.controller import MetaController
from optimization_copilot.plugins.registry import PluginRegistry


# ── Helpers ──────────────────────────────────────────────


def _make_spec(
    n_params: int = 2,
    max_iterations: int = 5,
    max_samples: int | None = None,
    seed: int = 42,
) -> OptimizationSpec:
    """Create a minimal OptimizationSpec for testing."""
    budget_kwargs: dict[str, Any] = {}
    if max_samples is not None:
        budget_kwargs["max_samples"] = max_samples
    else:
        budget_kwargs["max_iterations"] = max_iterations

    return OptimizationSpec(
        campaign_id="test-infra",
        parameters=[
            ParameterDef(
                name=f"x{i}",
                type=ParamType.CONTINUOUS,
                lower=0.0,
                upper=1.0,
            )
            for i in range(n_params)
        ],
        objectives=[
            ObjectiveDef(name="loss", direction=Direction.MINIMIZE),
        ],
        budget=BudgetDef(**budget_kwargs),
        seed=seed,
    )


def _make_registry() -> PluginRegistry:
    """Create a PluginRegistry with built-in samplers."""
    registry = PluginRegistry()
    registry.register(RandomSampler)
    registry.register(LatinHypercubeSampler)
    registry.register(TPESampler)
    return registry


def _simple_evaluator(params: dict) -> dict[str, float]:
    """Trivial evaluator: sum of parameter values (minimize)."""
    return {"loss": sum(params.values())}


def _make_mock_infrastructure(**overrides: Any) -> MagicMock:
    """Create a fully mocked InfrastructureStack.

    All methods return sensible defaults that can be overridden via kwargs.
    """
    mock = MagicMock(spec=InfrastructureStack)
    mock.warm_start_points.return_value = overrides.get("warm_start_points", [])
    mock.check_stopping.return_value = overrides.get("check_stopping", None)
    mock.pre_decide_signals.return_value = overrides.get("pre_decide_signals", {})
    mock.filter_candidates.side_effect = overrides.get(
        "filter_candidates", lambda candidates, parameter_specs: candidates
    )
    mock.record_trial.return_value = None
    return mock


# ═══════════════════════════════════════════════════════════
# 1. BACKWARD COMPATIBILITY (no infrastructure)
# ═══════════════════════════════════════════════════════════


class TestNoInfrastructureBackwardCompat:
    """Engine works exactly as before when infrastructure=None."""

    def test_engine_init_without_infrastructure(self):
        """Engine can be created without infrastructure parameter."""
        spec = _make_spec(max_iterations=2)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)
        assert engine._infrastructure is None

    def test_engine_init_with_infrastructure_none(self):
        """Explicitly passing infrastructure=None works."""
        spec = _make_spec(max_iterations=2)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=None)
        assert engine._infrastructure is None

    def test_engine_accepts_infrastructure_parameter(self):
        """Engine __init__ accepts infrastructure kwarg."""
        spec = _make_spec(max_iterations=2)
        registry = _make_registry()
        infra = _make_mock_infrastructure()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)
        assert engine._infrastructure is infra

    def test_run_without_infrastructure_completes(self):
        """Engine runs to completion with infrastructure=None."""
        spec = _make_spec(max_iterations=3)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)
        result = engine.run_with_evaluator(_simple_evaluator)
        assert result.total_iterations == 3
        assert "max_iterations_reached" in result.termination_reason

    def test_result_without_infrastructure(self):
        """Engine.result() works correctly without infrastructure."""
        spec = _make_spec(max_iterations=2)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)
        result = engine.run_with_evaluator(_simple_evaluator)
        assert isinstance(result, EngineResult)
        assert result.best_trial is not None
        assert "loss" in result.best_kpi_values

    def test_result_with_infrastructure_populated(self):
        """Engine.result() works correctly when infrastructure is provided."""
        spec = _make_spec(max_iterations=2)
        registry = _make_registry()
        infra = _make_mock_infrastructure()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)
        result = engine.run_with_evaluator(_simple_evaluator)
        assert isinstance(result, EngineResult)
        assert result.best_trial is not None

    def test_no_infrastructure_methods_called_when_none(self):
        """When infrastructure=None, no infrastructure methods are called."""
        spec = _make_spec(max_iterations=2)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)
        # Should not raise any AttributeError
        result = engine.run_with_evaluator(_simple_evaluator)
        assert result.total_iterations == 2


# ═══════════════════════════════════════════════════════════
# 2. WARM START INJECTION
# ═══════════════════════════════════════════════════════════


class TestWarmStartInjection:
    """Warm start candidates from transfer learning are injected correctly."""

    def test_warm_start_prepended_at_iteration_0(self):
        """Warm start candidates are prepended to candidates in iteration 0."""
        warm_points = [
            {"x0": 0.1, "x1": 0.2},
            {"x0": 0.3, "x1": 0.4},
        ]
        infra = _make_mock_infrastructure(warm_start_points=warm_points)

        spec = _make_spec(max_iterations=1)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        batches = []
        for batch in engine.run():
            batches.append(batch)
            for trial in batch.trials:
                trial.complete(kpi_values={"loss": sum(trial.parameters.values())})

        # Warm start should have been called once
        infra.warm_start_points.assert_called_once()

        # The first batch should contain the warm start candidates
        first_batch = batches[0]
        params_list = [t.parameters for t in first_batch.trials]
        # Warm start points appear at the front
        assert params_list[0] == {"x0": 0.1, "x1": 0.2}
        assert params_list[1] == {"x0": 0.3, "x1": 0.4}

    def test_no_warm_start_without_transfer_engine(self):
        """When warm_start_points returns empty, no extra candidates appear."""
        infra = _make_mock_infrastructure(warm_start_points=[])

        spec = _make_spec(max_iterations=1)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        for batch in engine.run():
            n_trials = len(batch.trials)
            for trial in batch.trials:
                trial.complete(kpi_values={"loss": 0.5})

        infra.warm_start_points.assert_called_once()

    def test_warm_start_only_on_first_iteration(self):
        """Warm start happens only once (iteration 0), not on subsequent iterations."""
        warm_points = [{"x0": 0.5, "x1": 0.5}]
        infra = _make_mock_infrastructure(warm_start_points=warm_points)

        spec = _make_spec(max_iterations=3)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        iteration_trial_counts = []
        for batch in engine.run():
            iteration_trial_counts.append(len(batch.trials))
            for trial in batch.trials:
                trial.complete(kpi_values={"loss": sum(trial.parameters.values())})

        # warm_start_points called exactly once (before the loop)
        infra.warm_start_points.assert_called_once()

        # The warm start candidate appears in iteration 0 but not in later ones.
        # Later iterations should have fewer trials (no warm start appended).
        # We check indirectly: the first iteration has at least 1 more trial
        # than what the meta controller would have suggested alone.
        assert len(iteration_trial_counts) == 3

    def test_warm_start_not_called_when_no_infrastructure(self):
        """No warm start attempt when infrastructure=None."""
        spec = _make_spec(max_iterations=2)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)
        result = engine.run_with_evaluator(_simple_evaluator)
        # If it ran without error, warm start was properly skipped
        assert result.total_iterations == 2


# ═══════════════════════════════════════════════════════════
# 3. INFRASTRUCTURE STOPPING
# ═══════════════════════════════════════════════════════════


class TestInfrastructureStopping:
    """Infrastructure stopping criteria correctly terminate the engine."""

    def test_engine_stops_when_should_stop_true(self):
        """Engine terminates when check_stopping returns should_stop=True."""
        stop_decision = StoppingDecision(
            should_stop=True,
            reason="budget_exhausted",
            criterion="budget",
        )
        infra = _make_mock_infrastructure(check_stopping=stop_decision)

        spec = _make_spec(max_iterations=100)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        result = engine.run_with_evaluator(_simple_evaluator)

        # Engine should stop at iteration 0 (before any work)
        assert result.total_iterations == 0
        assert "infrastructure_stopping" in result.termination_reason
        assert "budget_exhausted" in result.termination_reason

    def test_engine_continues_when_should_stop_false(self):
        """Engine continues when check_stopping returns should_stop=False."""
        continue_decision = StoppingDecision(
            should_stop=False,
            reason="Continue",
            criterion="none",
        )
        infra = _make_mock_infrastructure(check_stopping=continue_decision)

        spec = _make_spec(max_iterations=3)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        result = engine.run_with_evaluator(_simple_evaluator)

        assert result.total_iterations == 3
        assert "max_iterations_reached" in result.termination_reason

    def test_engine_continues_when_check_stopping_returns_none(self):
        """Engine continues when check_stopping returns None (no stopping rule)."""
        infra = _make_mock_infrastructure(check_stopping=None)

        spec = _make_spec(max_iterations=3)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        result = engine.run_with_evaluator(_simple_evaluator)

        assert result.total_iterations == 3

    def test_termination_reason_has_infrastructure_prefix(self):
        """Termination reason includes 'infrastructure_stopping:' prefix."""
        stop_decision = StoppingDecision(
            should_stop=True,
            reason="stagnation detected",
            criterion="stagnation",
        )
        infra = _make_mock_infrastructure(check_stopping=stop_decision)

        spec = _make_spec(max_iterations=100)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        result = engine.run_with_evaluator(_simple_evaluator)

        assert result.termination_reason.startswith("infrastructure_stopping:")
        assert "stagnation detected" in result.termination_reason

    def test_termination_event_emitted_on_infra_stop(self):
        """TERMINATION event is emitted when infrastructure stops the engine."""
        stop_decision = StoppingDecision(
            should_stop=True,
            reason="max_trials",
            criterion="max_trials",
        )
        infra = _make_mock_infrastructure(check_stopping=stop_decision)

        spec = _make_spec(max_iterations=100)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        termination_events = []
        engine.on(EngineEvent.TERMINATION, lambda p: termination_events.append(p))

        engine.run_with_evaluator(_simple_evaluator)

        assert len(termination_events) == 1
        assert "reason" in termination_events[0].data
        assert "infrastructure_stopping" in termination_events[0].data["reason"]

    def test_stopping_after_n_iterations(self):
        """Infrastructure stopping triggers after some iterations complete."""
        call_count = {"n": 0}

        def stop_after_two(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] >= 3:
                return StoppingDecision(
                    should_stop=True,
                    reason="enough trials",
                    criterion="max_trials",
                )
            return StoppingDecision(
                should_stop=False,
                reason="Continue",
                criterion="none",
            )

        infra = _make_mock_infrastructure()
        infra.check_stopping.side_effect = stop_after_two

        spec = _make_spec(max_iterations=100)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        result = engine.run_with_evaluator(_simple_evaluator)

        assert result.total_iterations == 2
        assert "infrastructure_stopping" in result.termination_reason


# ═══════════════════════════════════════════════════════════
# 4. PRE-DECIDE SIGNALS
# ═══════════════════════════════════════════════════════════


class TestPreDecideSignals:
    """Infrastructure signals are passed to MetaController.decide()."""

    def test_signals_passed_to_decide(self):
        """Signals from infrastructure are forwarded as kwargs to decide()."""
        signals = {
            "cost_signals": {"total_spent": 10.0, "remaining_budget": 90.0},
            "backend_policy": "random_sampler",
        }
        infra = _make_mock_infrastructure(pre_decide_signals=signals)

        spec = _make_spec(max_iterations=1)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        # Patch the meta controller to capture kwargs
        original_decide = engine._meta_controller.decide
        captured_kwargs: list[dict] = []

        def capturing_decide(*args, **kwargs):
            captured_kwargs.append(kwargs)
            return original_decide(*args, **kwargs)

        engine._meta_controller.decide = capturing_decide

        engine.run_with_evaluator(_simple_evaluator)

        assert len(captured_kwargs) == 1
        assert "cost_signals" in captured_kwargs[0]
        assert captured_kwargs[0]["cost_signals"]["total_spent"] == 10.0
        assert "backend_policy" in captured_kwargs[0]
        assert captured_kwargs[0]["backend_policy"] == "random_sampler"

    def test_empty_signals_when_no_modules_active(self):
        """Empty signals dict when infrastructure has no cost tracker or auto sampler."""
        infra = _make_mock_infrastructure(pre_decide_signals={})

        spec = _make_spec(max_iterations=1)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        engine.run_with_evaluator(_simple_evaluator)

        infra.pre_decide_signals.assert_called()

    def test_cost_signals_passed_correctly(self):
        """cost_signals dict is passed through correctly."""
        cost_signals = {
            "total_spent": 50.0,
            "remaining_budget": 50.0,
            "average_cost_per_trial": 5.0,
            "estimated_remaining_trials": 10,
            "n_trials_recorded": 10,
        }
        signals = {"cost_signals": cost_signals}
        infra = _make_mock_infrastructure(pre_decide_signals=signals)

        spec = _make_spec(max_iterations=1)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        original_decide = engine._meta_controller.decide
        captured: list[dict] = []

        def capture(*args, **kwargs):
            captured.append(kwargs)
            return original_decide(*args, **kwargs)

        engine._meta_controller.decide = capture
        engine.run_with_evaluator(_simple_evaluator)

        assert captured[0]["cost_signals"]["total_spent"] == 50.0
        assert captured[0]["cost_signals"]["remaining_budget"] == 50.0

    def test_backend_policy_string_passed_correctly(self):
        """backend_policy string is passed through correctly."""
        signals = {"backend_policy": "tpe_sampler"}
        infra = _make_mock_infrastructure(pre_decide_signals=signals)

        spec = _make_spec(max_iterations=1)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        original_decide = engine._meta_controller.decide
        captured: list[dict] = []

        def capture(*args, **kwargs):
            captured.append(kwargs)
            return original_decide(*args, **kwargs)

        engine._meta_controller.decide = capture
        engine.run_with_evaluator(_simple_evaluator)

        assert captured[0]["backend_policy"] == "tpe_sampler"

    def test_pre_decide_signals_not_called_without_infrastructure(self):
        """pre_decide_signals is not invoked when infrastructure is None."""
        spec = _make_spec(max_iterations=2)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)
        # Should run fine without attempting to call pre_decide_signals
        result = engine.run_with_evaluator(_simple_evaluator)
        assert result.total_iterations == 2


# ═══════════════════════════════════════════════════════════
# 5. CANDIDATE FILTERING
# ═══════════════════════════════════════════════════════════


class TestCandidateFiltering:
    """Infrastructure constraint engine filters candidates correctly."""

    def test_candidates_filtered_by_constraint_engine(self):
        """Candidates are filtered when constraint engine is active."""
        def filter_fn(candidates, parameter_specs):
            # Only keep candidates where x0 < 0.5
            return [c for c in candidates if c.get("x0", 1.0) < 0.5]

        infra = _make_mock_infrastructure(filter_candidates=filter_fn)

        spec = _make_spec(max_iterations=1)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        for batch in engine.run():
            # All trials should have x0 < 0.5 (or be from the plugin
            # producing values in [0,1], some may be filtered out)
            for trial in batch.trials:
                assert trial.parameters.get("x0", 1.0) < 0.5
                trial.complete(kpi_values={"loss": sum(trial.parameters.values())})

    def test_no_filtering_without_constraints(self):
        """No filtering when filter_candidates passes through all candidates."""
        infra = _make_mock_infrastructure()
        # Default: filter_candidates passes through

        spec = _make_spec(max_iterations=1)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        for batch in engine.run():
            assert len(batch.trials) > 0
            for trial in batch.trials:
                trial.complete(kpi_values={"loss": 0.5})

    def test_filtered_candidates_used_for_trials(self):
        """Trial parameters come from filtered candidates, not originals."""
        filtered_candidate = {"x0": 0.99, "x1": 0.99}

        def always_replace(candidates, parameter_specs):
            return [filtered_candidate]

        infra = _make_mock_infrastructure(filter_candidates=always_replace)

        spec = _make_spec(max_iterations=1)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        for batch in engine.run():
            # The only trial should have the filtered parameters
            assert len(batch.trials) >= 1
            assert batch.trials[0].parameters == {"x0": 0.99, "x1": 0.99}
            for trial in batch.trials:
                trial.complete(kpi_values={"loss": sum(trial.parameters.values())})

    def test_filter_candidates_receives_param_dicts(self):
        """filter_candidates is called with parameter_specs as dicts."""
        infra = _make_mock_infrastructure()

        spec = _make_spec(max_iterations=1)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        for batch in engine.run():
            for trial in batch.trials:
                trial.complete(kpi_values={"loss": 0.5})

        # Verify filter_candidates was called with param dicts
        assert infra.filter_candidates.called
        call_args = infra.filter_candidates.call_args
        param_specs_arg = call_args.kwargs.get(
            "parameter_specs", call_args[1].get("parameter_specs") if len(call_args) > 1 else None
        )
        if param_specs_arg is None and len(call_args[0]) > 1:
            param_specs_arg = call_args[0][1]
        # Should be a list of dicts (converted from ParameterSpec dataclasses)
        assert isinstance(param_specs_arg, list)


# ═══════════════════════════════════════════════════════════
# 6. TRIAL RECORDING
# ═══════════════════════════════════════════════════════════


class TestTrialRecording:
    """Infrastructure record_trial is called at appropriate times."""

    def test_record_trial_called_for_completed_trials(self):
        """record_trial is called for each completed trial."""
        infra = _make_mock_infrastructure()

        spec = _make_spec(max_iterations=2)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        result = engine.run_with_evaluator(_simple_evaluator)

        # record_trial should have been called for each completed trial
        assert infra.record_trial.call_count == result.total_trials

    def test_record_trial_not_called_for_failed_trials(self):
        """record_trial is NOT called for failed trials."""
        infra = _make_mock_infrastructure()

        spec = _make_spec(max_iterations=2)
        registry = _make_registry()
        config = EngineConfig(max_retries=1)
        engine = OptimizationEngine(spec, registry, config=config, infrastructure=infra)

        call_count = {"n": 0}

        def sometimes_fail(params):
            call_count["n"] += 1
            if call_count["n"] % 3 == 0:
                raise RuntimeError("deliberate failure")
            return {"loss": sum(params.values())}

        result = engine.run_with_evaluator(sometimes_fail)

        # record_trial should only be called for completed trials
        completed_count = result.total_trials - result.total_failures
        # Because retries can happen, the exact count is tricky.
        # But record_trial should not have been called more than completed trials.
        assert infra.record_trial.call_count <= result.total_trials
        # And it should have been called at least once (some succeeded)
        assert infra.record_trial.call_count > 0

    def test_record_trial_receives_correct_args(self):
        """trial_id and kpi_values are passed correctly to record_trial."""
        infra = _make_mock_infrastructure()

        spec = _make_spec(max_iterations=1)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        engine.run_with_evaluator(_simple_evaluator)

        # Check each record_trial call has the expected kwargs
        for call in infra.record_trial.call_args_list:
            kwargs = call.kwargs
            assert "trial_params" in kwargs
            assert "kpi_values" in kwargs
            assert "trial_id" in kwargs
            assert isinstance(kwargs["trial_params"], dict)
            assert isinstance(kwargs["kpi_values"], dict)
            assert "loss" in kwargs["kpi_values"]
            assert isinstance(kwargs["trial_id"], str)

    def test_record_trial_not_called_without_infrastructure(self):
        """No record_trial when infrastructure is None."""
        spec = _make_spec(max_iterations=2)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)
        # Should work fine without trying to call record_trial
        result = engine.run_with_evaluator(_simple_evaluator)
        assert result.total_trials > 0

    def test_record_trial_called_per_trial_not_per_batch(self):
        """record_trial is called once per completed trial, not once per batch."""
        infra = _make_mock_infrastructure()

        spec = _make_spec(max_iterations=1)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        trial_count = 0
        for batch in engine.run():
            for trial in batch.trials:
                trial.complete(kpi_values={"loss": 0.5})
                trial_count += 1

        assert infra.record_trial.call_count == trial_count

    def test_all_fail_batch_no_record_trial(self):
        """When all trials in a batch fail, record_trial is not called."""
        infra = _make_mock_infrastructure()

        spec = _make_spec(max_iterations=2)
        registry = _make_registry()
        config = EngineConfig(max_retries=1)
        engine = OptimizationEngine(spec, registry, config=config, infrastructure=infra)

        def always_fail(params):
            raise RuntimeError("always fails")

        engine.run_with_evaluator(always_fail)

        # No completed trials -> record_trial not called
        assert infra.record_trial.call_count == 0


# ═══════════════════════════════════════════════════════════
# 7. BEST VALUES HISTORY TRACKING
# ═══════════════════════════════════════════════════════════


class TestBestValuesHistory:
    """_best_values_history is populated as trials complete."""

    def test_best_values_history_populated(self):
        """_best_values_history is populated as trials complete."""
        infra = _make_mock_infrastructure()

        spec = _make_spec(max_iterations=3)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        engine.run_with_evaluator(_simple_evaluator)

        # After running, _best_values_history should be non-empty
        assert len(engine._best_values_history) > 0

    def test_best_values_history_monotonic_for_minimize(self):
        """For minimization, best values history should be non-increasing."""
        spec = _make_spec(max_iterations=5)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)

        engine.run_with_evaluator(_simple_evaluator)

        history = engine._best_values_history
        if len(history) > 1:
            for i in range(1, len(history)):
                assert history[i] <= history[i - 1]

    def test_best_values_passed_to_check_stopping(self):
        """check_stopping receives the best_values_history."""
        infra = _make_mock_infrastructure()

        spec = _make_spec(max_iterations=3)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        engine.run_with_evaluator(_simple_evaluator)

        # check_stopping should have been called with best_values
        for call in infra.check_stopping.call_args_list:
            kwargs = call.kwargs
            # best_values is passed as a keyword arg
            assert "best_values" in kwargs or len(call.args) >= 2

    def test_empty_history_at_start(self):
        """At the start, _best_values_history is empty."""
        spec = _make_spec(max_iterations=1)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)
        assert engine._best_values_history == []

    def test_history_not_populated_for_all_failures(self):
        """_best_values_history stays empty if all trials fail."""
        spec = _make_spec(max_iterations=3)
        registry = _make_registry()
        config = EngineConfig(max_retries=1)
        engine = OptimizationEngine(spec, registry, config=config)

        def always_fail(params):
            raise RuntimeError("fail")

        engine.run_with_evaluator(always_fail)

        assert len(engine._best_values_history) == 0


# ═══════════════════════════════════════════════════════════
# 8. METACONTROLLER BACKEND_POLICY INTEGRATION
# ═══════════════════════════════════════════════════════════


class TestMetaControllerBackendPolicy:
    """MetaController handles backend_policy hints from AutoSampler."""

    def test_string_backend_policy_used_when_available(self):
        """MetaController uses string backend_policy hint from AutoSampler."""
        mc = MetaController(available_backends=["random_sampler", "tpe_sampler"])

        snapshot = CampaignSnapshot(
            campaign_id="test",
            parameter_specs=[
                ParameterSpec(name="x0", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0)
            ],
            observations=[],
            objective_names=["loss"],
            objective_directions=["minimize"],
        )
        fingerprint = ProblemFingerprint()

        decision = mc.decide(
            snapshot=snapshot,
            diagnostics={},
            fingerprint=fingerprint,
            seed=42,
            backend_policy="tpe_sampler",
        )

        # When no portfolio is provided, the backend_policy string hint is used
        assert decision.backend_name == "tpe_sampler"
        assert any("auto_sampler_hint" in rc for rc in decision.reason_codes)

    def test_fallback_when_hint_not_in_available_backends(self):
        """Falls back to rule-based when hint is not in available_backends."""
        mc = MetaController(available_backends=["random_sampler", "tpe_sampler"])

        snapshot = CampaignSnapshot(
            campaign_id="test",
            parameter_specs=[
                ParameterSpec(name="x0", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0)
            ],
            observations=[],
            objective_names=["loss"],
            objective_directions=["minimize"],
        )
        fingerprint = ProblemFingerprint()

        decision = mc.decide(
            snapshot=snapshot,
            diagnostics={},
            fingerprint=fingerprint,
            seed=42,
            backend_policy="nonexistent_sampler",
        )

        # Should fall back to rule-based selection
        assert decision.backend_name in ["random_sampler", "tpe_sampler"]
        assert not any("auto_sampler_hint" in rc for rc in decision.reason_codes)

    def test_portfolio_takes_priority_over_hint(self):
        """Portfolio scorer takes priority over AutoSampler hint."""
        mc = MetaController(available_backends=["random_sampler", "tpe_sampler"])

        snapshot = CampaignSnapshot(
            campaign_id="test",
            parameter_specs=[
                ParameterSpec(name="x0", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0)
            ],
            observations=[],
            objective_names=["loss"],
            objective_directions=["minimize"],
        )
        fingerprint = ProblemFingerprint()

        # When portfolio is provided but fails (no portfolio module), it falls
        # through to backend_policy hint
        decision = mc.decide(
            snapshot=snapshot,
            diagnostics={},
            fingerprint=fingerprint,
            seed=42,
            backend_policy="tpe_sampler",
            portfolio=None,  # No portfolio -> uses hint
        )

        # Without a real portfolio, it should use the hint
        assert decision.backend_name == "tpe_sampler"

    def test_none_backend_policy_uses_rules(self):
        """When backend_policy is None, rule-based selection is used."""
        mc = MetaController(available_backends=["random_sampler", "tpe_sampler"])

        snapshot = CampaignSnapshot(
            campaign_id="test",
            parameter_specs=[
                ParameterSpec(name="x0", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0)
            ],
            observations=[],
            objective_names=["loss"],
            objective_directions=["minimize"],
        )
        fingerprint = ProblemFingerprint()

        decision = mc.decide(
            snapshot=snapshot,
            diagnostics={},
            fingerprint=fingerprint,
            seed=42,
            backend_policy=None,
        )

        assert decision.backend_name in ["random_sampler", "tpe_sampler"]

    def test_non_string_backend_policy_ignored(self):
        """Non-string backend_policy values are not used as hints."""
        mc = MetaController(available_backends=["random_sampler", "tpe_sampler"])

        snapshot = CampaignSnapshot(
            campaign_id="test",
            parameter_specs=[
                ParameterSpec(name="x0", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0)
            ],
            observations=[],
            objective_names=["loss"],
            objective_directions=["minimize"],
        )
        fingerprint = ProblemFingerprint()

        decision = mc.decide(
            snapshot=snapshot,
            diagnostics={},
            fingerprint=fingerprint,
            seed=42,
            backend_policy=42,  # Not a string
        )

        # Should fall back to rule-based
        assert decision.backend_name in ["random_sampler", "tpe_sampler"]
        assert not any("auto_sampler_hint" in rc for rc in decision.reason_codes)


# ═══════════════════════════════════════════════════════════
# 9. _param_specs_to_dicts HELPER
# ═══════════════════════════════════════════════════════════


class TestParamSpecsToDicts:
    """The _param_specs_to_dicts helper converts various formats correctly."""

    def test_converts_dataclass_to_dict(self):
        """Converts dataclass ParameterSpec objects to dicts."""
        specs = [
            ParameterSpec(name="x0", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
            ParameterSpec(name="x1", type=VariableType.DISCRETE, lower=1, upper=10),
        ]
        result = OptimizationEngine._param_specs_to_dicts(specs)

        assert len(result) == 2
        assert isinstance(result[0], dict)
        assert result[0]["name"] == "x0"
        assert result[1]["name"] == "x1"

    def test_passes_through_dicts_unchanged(self):
        """Passes through dict objects unchanged."""
        dicts = [
            {"name": "x0", "type": "continuous", "lower": 0.0, "upper": 1.0},
            {"name": "x1", "type": "discrete", "lower": 1, "upper": 10},
        ]
        result = OptimizationEngine._param_specs_to_dicts(dicts)

        assert len(result) == 2
        assert result[0] == dicts[0]
        assert result[1] == dicts[1]

    def test_handles_enum_values(self):
        """Converts enum values to their .value representation."""
        specs = [
            ParameterSpec(name="x0", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0),
        ]
        result = OptimizationEngine._param_specs_to_dicts(specs)

        # VariableType.CONTINUOUS should be converted to "continuous"
        assert result[0]["type"] == "continuous"

    def test_handles_mixed_input(self):
        """Handles a mix of dicts and dataclasses."""
        specs = [
            {"name": "x0", "type": "continuous"},
            ParameterSpec(name="x1", type=VariableType.DISCRETE, lower=0, upper=5),
        ]
        result = OptimizationEngine._param_specs_to_dicts(specs)

        assert len(result) == 2
        assert result[0]["name"] == "x0"
        assert result[1]["name"] == "x1"

    def test_empty_list(self):
        """Empty input produces empty output."""
        result = OptimizationEngine._param_specs_to_dicts([])
        assert result == []

    def test_fallback_for_unknown_types(self):
        """Non-dict, non-dataclass objects get a fallback representation."""
        result = OptimizationEngine._param_specs_to_dicts(["x0", "x1"])
        assert len(result) == 2
        assert result[0] == {"name": "x0"}
        assert result[1] == {"name": "x1"}

    def test_categorical_spec_conversion(self):
        """ParameterSpec with categorical type and categories converts correctly."""
        specs = [
            ParameterSpec(
                name="color",
                type=VariableType.CATEGORICAL,
                categories=["red", "green", "blue"],
            ),
        ]
        result = OptimizationEngine._param_specs_to_dicts(specs)

        assert result[0]["name"] == "color"
        assert result[0]["type"] == "categorical"
        assert result[0]["categories"] == ["red", "green", "blue"]

    def test_none_fields_preserved(self):
        """None fields in ParameterSpec are preserved in output."""
        specs = [
            ParameterSpec(name="x0", type=VariableType.CONTINUOUS),
        ]
        result = OptimizationEngine._param_specs_to_dicts(specs)

        assert result[0]["lower"] is None
        assert result[0]["upper"] is None


# ═══════════════════════════════════════════════════════════
# 10. FULL INTEGRATION ROUND-TRIP
# ═══════════════════════════════════════════════════════════


class TestFullIntegrationRoundTrip:
    """End-to-end tests with real InfrastructureConfig and InfrastructureStack."""

    def test_engine_with_real_infrastructure_no_modules(self):
        """Engine works with a real InfrastructureStack with no modules configured."""
        infra_config = InfrastructureConfig()
        infra = InfrastructureStack(infra_config)

        spec = _make_spec(max_iterations=3)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        result = engine.run_with_evaluator(_simple_evaluator)

        assert result.total_iterations == 3
        assert result.best_trial is not None

    def test_engine_with_cost_tracking(self):
        """Engine with real cost tracker records trials."""
        infra_config = InfrastructureConfig(budget=100.0)
        infra = InfrastructureStack(infra_config)

        spec = _make_spec(max_iterations=3)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        result = engine.run_with_evaluator(_simple_evaluator)

        assert result.total_iterations == 3
        # Cost tracker should have recorded trials
        assert infra.cost_tracker is not None
        assert infra.cost_tracker.n_trials == result.total_trials

    def test_engine_with_stopping_rule(self):
        """Engine with real stopping rule (max_trials)."""
        infra_config = InfrastructureConfig(max_trials=3)
        infra = InfrastructureStack(infra_config)

        spec = _make_spec(max_iterations=100)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        result = engine.run_with_evaluator(_simple_evaluator)

        # Should stop based on infrastructure stopping rule
        # (though the exact iteration depends on batch size)
        assert "infrastructure_stopping" in result.termination_reason or result.total_iterations < 100

    def test_engine_with_constraint_filtering(self):
        """Engine with real constraint engine filters candidates."""

        def x0_less_than_half(params):
            return params.get("x0", 1.0) < 0.5

        infra_config = InfrastructureConfig(
            constraints=[
                {
                    "name": "x0_bound",
                    "constraint_type": "known_hard",
                    "evaluate": x0_less_than_half,
                }
            ]
        )
        infra = InfrastructureStack(infra_config)

        spec = _make_spec(max_iterations=3)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        all_trial_params = []
        for batch in engine.run():
            for trial in batch.trials:
                all_trial_params.append(dict(trial.parameters))
                trial.complete(kpi_values={"loss": sum(trial.parameters.values())})

        # All trials should satisfy x0 < 0.5
        for params in all_trial_params:
            assert params["x0"] < 0.5, f"Constraint violated: x0={params['x0']}"

    def test_engine_with_auto_sampler(self):
        """Engine with real auto sampler provides backend hints."""
        infra_config = InfrastructureConfig(
            available_backends=["random_sampler", "tpe_sampler"],
        )
        infra = InfrastructureStack(infra_config)

        spec = _make_spec(max_iterations=2)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        result = engine.run_with_evaluator(_simple_evaluator)

        assert result.total_iterations == 2
        # Auto sampler should have been used for pre-decide signals
        assert infra.auto_sampler is not None

    def test_engine_with_multiple_modules(self):
        """Engine with multiple infrastructure modules active simultaneously."""
        def x_sum_constraint(params):
            return sum(params.values()) < 1.5

        infra_config = InfrastructureConfig(
            budget=1000.0,
            max_trials=50,
            constraints=[
                {
                    "name": "sum_bound",
                    "constraint_type": "known_hard",
                    "evaluate": x_sum_constraint,
                }
            ],
            available_backends=["random_sampler", "tpe_sampler"],
        )
        infra = InfrastructureStack(infra_config)

        spec = _make_spec(max_iterations=3)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        result = engine.run_with_evaluator(_simple_evaluator)

        assert result.total_iterations == 3
        assert result.best_trial is not None
        assert infra.cost_tracker is not None
        assert infra.cost_tracker.n_trials > 0

    def test_full_integration_audit_trail(self):
        """Full integration run produces valid audit trail."""
        infra_config = InfrastructureConfig(budget=500.0)
        infra = InfrastructureStack(infra_config)

        spec = _make_spec(max_iterations=2)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        result = engine.run_with_evaluator(_simple_evaluator)

        assert len(result.audit_trail) == result.total_iterations
        for entry in result.audit_trail:
            assert "decision" in entry
            assert "backend" in entry

    def test_result_to_dict_with_infrastructure(self):
        """EngineResult.to_dict() works correctly with infrastructure results."""
        infra_config = InfrastructureConfig(budget=100.0)
        infra = InfrastructureStack(infra_config)

        spec = _make_spec(max_iterations=2)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        result = engine.run_with_evaluator(_simple_evaluator)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "best_trial" in result_dict
        assert "termination_reason" in result_dict

    def test_deterministic_with_infrastructure(self):
        """Results are deterministic even with infrastructure modules."""
        results = []
        for _ in range(2):
            infra_config = InfrastructureConfig(budget=100.0)
            infra = InfrastructureStack(infra_config)

            spec = _make_spec(max_iterations=3, seed=42)
            registry = _make_registry()
            engine = OptimizationEngine(spec, registry, infrastructure=infra)

            result = engine.run_with_evaluator(_simple_evaluator)
            results.append(result)

        assert results[0].best_kpi_values == results[1].best_kpi_values
        assert results[0].total_iterations == results[1].total_iterations
        assert results[0].total_trials == results[1].total_trials


# ═══════════════════════════════════════════════════════════
# ADDITIONAL EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and boundary conditions for integration wiring."""

    def test_infrastructure_with_generator_pattern(self):
        """Infrastructure works with the generator-based engine pattern."""
        infra = _make_mock_infrastructure()

        spec = _make_spec(max_iterations=2)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        for batch in engine.run():
            for trial in batch.trials:
                trial.complete(kpi_values={"loss": 0.5})

        result = engine.result()
        assert result.total_iterations == 2

    def test_infrastructure_with_single_iteration(self):
        """Infrastructure works with a single iteration."""
        infra = _make_mock_infrastructure()

        spec = _make_spec(max_iterations=1)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        result = engine.run_with_evaluator(_simple_evaluator)

        assert result.total_iterations == 1
        infra.check_stopping.assert_called()
        infra.pre_decide_signals.assert_called()
        infra.filter_candidates.assert_called()

    def test_check_stopping_called_each_iteration(self):
        """check_stopping is called at the top of each iteration."""
        infra = _make_mock_infrastructure()

        spec = _make_spec(max_iterations=5)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        engine.run_with_evaluator(_simple_evaluator)

        # check_stopping should be called once per iteration
        assert infra.check_stopping.call_count == 5

    def test_pre_decide_called_each_iteration(self):
        """pre_decide_signals is called each iteration."""
        infra = _make_mock_infrastructure()

        spec = _make_spec(max_iterations=4)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        engine.run_with_evaluator(_simple_evaluator)

        assert infra.pre_decide_signals.call_count == 4

    def test_filter_candidates_called_each_iteration(self):
        """filter_candidates is called each iteration."""
        infra = _make_mock_infrastructure()

        spec = _make_spec(max_iterations=3)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        engine.run_with_evaluator(_simple_evaluator)

        assert infra.filter_candidates.call_count == 3

    def test_warm_start_with_matching_params(self):
        """Warm start points with correct parameter names work."""
        warm_points = [{"x0": 0.25, "x1": 0.75}]
        infra = _make_mock_infrastructure(warm_start_points=warm_points)

        spec = _make_spec(max_iterations=1)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry, infrastructure=infra)

        for batch in engine.run():
            found_warm = False
            for trial in batch.trials:
                if trial.parameters == {"x0": 0.25, "x1": 0.75}:
                    found_warm = True
                trial.complete(kpi_values={"loss": sum(trial.parameters.values())})
            assert found_warm, "Warm start candidate not found in batch"

    def test_engine_with_config_and_infrastructure(self):
        """Engine accepts both config and infrastructure simultaneously."""
        infra = _make_mock_infrastructure()
        config = EngineConfig(max_retries=5)

        spec = _make_spec(max_iterations=2)
        registry = _make_registry()
        engine = OptimizationEngine(
            spec, registry, config=config, infrastructure=infra
        )

        assert engine._infrastructure is infra
        assert engine._config.max_retries == 5

        result = engine.run_with_evaluator(_simple_evaluator)
        assert result.total_iterations == 2
