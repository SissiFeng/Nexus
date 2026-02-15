"""Tests for campaign state serialization, checkpoint, and resume.

Covers CampaignState to_json/from_json round-trip, engine checkpoint
and resume across separate engine instances, and file-based checkpoint
persistence.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from optimization_copilot.backends.builtin import (
    LatinHypercubeSampler,
    RandomSampler,
    TPESampler,
)
from optimization_copilot.dsl.bridge import SpecBridge
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
from optimization_copilot.engine.events import EngineEvent
from optimization_copilot.engine.state import CampaignState
from optimization_copilot.engine.trial import TrialState
from optimization_copilot.plugins.registry import PluginRegistry


# ── Helpers ──────────────────────────────────────────────


def _make_spec(max_samples: int = 15, seed: int = 42) -> OptimizationSpec:
    """Create a simple OptimizationSpec with 2 continuous params and 1 objective."""
    return OptimizationSpec(
        campaign_id="checkpoint-test",
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


# ── Campaign State Serialization ─────────────────────────


class TestCampaignStateSerialization:
    """Test CampaignState to_json / from_json round-trip fidelity."""

    def test_fresh_state_round_trip(self):
        """A freshly created CampaignState should survive JSON round-trip."""
        spec = _make_spec()
        snapshot = SpecBridge.to_campaign_snapshot(spec)
        state = CampaignState(spec=spec, snapshot=snapshot, seed=42)

        json_str = state.to_json()
        restored = CampaignState.from_json(json_str)

        assert restored.iteration == state.iteration
        assert restored.seed == state.seed
        assert restored.terminated == state.terminated
        assert restored.termination_reason == state.termination_reason
        assert restored.spec.campaign_id == spec.campaign_id
        assert len(restored.spec.parameters) == len(spec.parameters)
        assert len(restored.spec.objectives) == len(spec.objectives)
        assert restored.snapshot.campaign_id == snapshot.campaign_id

    def test_state_with_history_round_trip(self):
        """State with populated histories should round-trip correctly."""
        spec = _make_spec()
        snapshot = SpecBridge.to_campaign_snapshot(spec)
        state = CampaignState(
            spec=spec,
            snapshot=snapshot,
            iteration=5,
            phase_history=[
                {"iteration": 2, "from_phase": "cold_start", "to_phase": "learning"},
            ],
            decision_history=[
                {"backend_name": "random_sampler", "phase": "cold_start"},
                {"backend_name": "random_sampler", "phase": "learning"},
            ],
            completed_trials=[
                {"trial_id": "t-0000-00", "iteration": 0, "parameters": {"x1": 1.0, "x2": 2.0}},
            ],
            pending_retries=[],
            terminated=False,
            seed=42,
        )

        json_str = state.to_json()
        restored = CampaignState.from_json(json_str)

        assert restored.iteration == 5
        assert len(restored.phase_history) == 1
        assert restored.phase_history[0]["to_phase"] == "learning"
        assert len(restored.decision_history) == 2
        assert len(restored.completed_trials) == 1
        assert restored.completed_trials[0]["trial_id"] == "t-0000-00"

    def test_terminated_state_round_trip(self):
        """A terminated state should preserve terminated flag and reason."""
        spec = _make_spec()
        snapshot = SpecBridge.to_campaign_snapshot(spec)
        state = CampaignState(
            spec=spec,
            snapshot=snapshot,
            iteration=15,
            terminated=True,
            termination_reason="max_samples_reached:15/15",
            seed=42,
        )

        json_str = state.to_json()
        restored = CampaignState.from_json(json_str)

        assert restored.terminated is True
        assert restored.termination_reason == "max_samples_reached:15/15"
        assert restored.iteration == 15

    def test_json_is_valid(self):
        """to_json should produce valid JSON."""
        spec = _make_spec()
        snapshot = SpecBridge.to_campaign_snapshot(spec)
        state = CampaignState(spec=spec, snapshot=snapshot, seed=42)

        json_str = state.to_json()
        parsed = json.loads(json_str)

        assert isinstance(parsed, dict)
        assert "spec" in parsed
        assert "snapshot" in parsed
        assert "iteration" in parsed
        assert "seed" in parsed

    def test_to_dict_from_dict_round_trip(self):
        """to_dict / from_dict should also work correctly."""
        spec = _make_spec()
        snapshot = SpecBridge.to_campaign_snapshot(spec)
        state = CampaignState(spec=spec, snapshot=snapshot, iteration=3, seed=42)

        data = state.to_dict()
        restored = CampaignState.from_dict(data)

        assert restored.iteration == 3
        assert restored.seed == 42
        assert restored.spec.campaign_id == spec.campaign_id


# ── Checkpoint and Resume ────────────────────────────────


class TestCheckpointResume:
    """Test running part of a campaign, checkpointing, and resuming."""

    def test_checkpoint_resume_continues_correctly(self):
        """Run some iterations, checkpoint, resume, run more. Total matches."""
        # Use run_with_evaluator for phase 1 with a small budget so the
        # engine completes cleanly (no generator break issues).
        spec_phase1 = _make_spec(max_samples=5, seed=42)
        registry = _make_registry()
        engine = OptimizationEngine(spec_phase1, registry)

        result_phase1 = engine.run_with_evaluator(_simple_evaluator)
        phase1_iterations = result_phase1.total_iterations
        assert phase1_iterations > 0

        # Checkpoint the state.
        checkpoint_state = engine.checkpoint()
        assert checkpoint_state.iteration == phase1_iterations

        # Phase 2: Create a new engine from checkpointed state with
        # a larger budget so it can run more iterations.
        spec_phase2 = _make_spec(max_samples=50, seed=42)
        # Clear termination so the engine can continue.
        checkpoint_state.terminated = False
        checkpoint_state.termination_reason = ""
        checkpoint_state.spec = spec_phase2

        resumed_engine = OptimizationEngine(
            spec_phase2,
            _make_registry(),
            state=checkpoint_state,
        )

        # Run a few more iterations via generator, then stop manually.
        target_additional = 3
        iterations_phase2 = 0
        for batch in resumed_engine.run():
            for trial in batch.trials:
                trial.complete(kpi_values=_simple_evaluator(trial.parameters))
            iterations_phase2 += 1
            if iterations_phase2 >= target_additional:
                resumed_engine.stop(reason="phase2_done")

        result = resumed_engine.result()
        # Total iterations = phase1 iterations + phase2 iterations.
        assert result.total_iterations == phase1_iterations + target_additional

    def test_checkpoint_preserves_observations(self):
        """Observations from phase 1 should be present after resume."""
        spec = _make_spec(max_samples=8, seed=42)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)

        result = engine.run_with_evaluator(_simple_evaluator)

        checkpoint_state = engine.checkpoint()

        # After a complete run, the snapshot should have all observations.
        assert checkpoint_state.snapshot.n_observations >= 8
        assert checkpoint_state.snapshot.n_observations == result.total_trials

    def test_checkpoint_preserves_decision_history(self):
        """Decision history from phase 1 should be in the checkpoint."""
        spec = _make_spec(max_samples=15, seed=42)
        registry = _make_registry()
        engine = OptimizationEngine(spec, registry)

        iterations_run = 0
        for batch in engine.run():
            for trial in batch.trials:
                trial.complete(kpi_values={"y": 5.0})
            iterations_run += 1
            if iterations_run >= 4:
                engine.stop(reason="checkpoint")
                break

        checkpoint_state = engine.checkpoint()

        assert len(checkpoint_state.decision_history) == 4


# ── Checkpoint File ──────────────────────────────────────


class TestCheckpointFile:
    """Test file-based checkpoint and resume using CampaignState methods."""

    def test_checkpoint_to_file_and_resume(self):
        """Write checkpoint to a temp file, read it back, verify round-trip."""
        spec = _make_spec(max_samples=10, seed=42)
        snapshot = SpecBridge.to_campaign_snapshot(spec)
        trial_records = [
            {"trial_id": f"t-{i}", "iteration": i, "parameters": {"x1": float(i)}}
            for i in range(7)
        ]
        state = CampaignState(
            spec=spec,
            snapshot=snapshot,
            iteration=7,
            phase_history=[{"iteration": 3, "to_phase": "learning"}],
            decision_history=[{"backend": "random_sampler"}] * 7,
            completed_trials=trial_records,
            terminated=False,
            seed=42,
        )

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, dir="/tmp"
        ) as f:
            tmp_path = f.name

        try:
            state.checkpoint_to_file(tmp_path)
            assert os.path.exists(tmp_path)

            restored = CampaignState.resume_from_file(tmp_path)

            assert restored.iteration == 7
            assert restored.seed == 42
            assert len(restored.phase_history) == 1
            assert len(restored.decision_history) == 7
            assert len(restored.completed_trials) == 7
            assert restored.spec.campaign_id == spec.campaign_id
            assert restored.terminated is False
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_checkpoint_file_is_valid_json(self):
        """The checkpoint file should contain valid JSON."""
        spec = _make_spec()
        snapshot = SpecBridge.to_campaign_snapshot(spec)
        state = CampaignState(spec=spec, snapshot=snapshot, seed=42)

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, dir="/tmp"
        ) as f:
            tmp_path = f.name

        try:
            state.checkpoint_to_file(tmp_path)

            with open(tmp_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            assert isinstance(data, dict)
            assert "spec" in data
            assert "snapshot" in data
            assert data["seed"] == 42
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_resume_from_missing_file_raises(self):
        """Attempting to resume from a nonexistent file should raise."""
        with pytest.raises(FileNotFoundError):
            CampaignState.resume_from_file("/tmp/nonexistent_checkpoint_xyz.json")

    def test_engine_auto_checkpoint(self):
        """Engine with checkpoint_every should write files automatically."""
        tmp_path = os.path.join("/tmp", "auto_checkpoint_test.json")
        # Ensure clean slate.
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

        try:
            # Use max_iterations to get a predictable number of iterations.
            spec = OptimizationSpec(
                campaign_id="auto-ckpt-test",
                parameters=[
                    ParameterDef(name="x1", type=ParamType.CONTINUOUS, lower=0.0, upper=10.0),
                    ParameterDef(name="x2", type=ParamType.CONTINUOUS, lower=0.0, upper=10.0),
                ],
                objectives=[
                    ObjectiveDef(name="y", direction=Direction.MAXIMIZE),
                ],
                budget=BudgetDef(max_iterations=6),
                seed=42,
            )
            registry = _make_registry()
            config = EngineConfig(
                checkpoint_every=3,
                checkpoint_path=tmp_path,
            )
            engine = OptimizationEngine(spec, registry, config=config)

            checkpoint_events = []
            engine.on(
                EngineEvent.CHECKPOINT_SAVED,
                lambda p: checkpoint_events.append(p),
            )

            engine.run_with_evaluator(_simple_evaluator)

            # checkpoint_every=3 fires when (iteration+1) % 3 == 0.
            # With 6 iterations (0-5): fires at iteration 2 and 5.
            assert len(checkpoint_events) == 2
            assert os.path.exists(tmp_path)

            # Verify the file contains valid state.
            restored = CampaignState.resume_from_file(tmp_path)
            assert restored.spec.campaign_id == spec.campaign_id
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_full_engine_checkpoint_resume_cycle(self):
        """End-to-end: run engine partially, checkpoint to file, resume from file."""
        tmp_path = os.path.join("/tmp", "full_cycle_ckpt_test.json")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

        try:
            # Phase 1: Run with a small budget, checkpoint to file.
            spec_phase1 = _make_spec(max_samples=5, seed=42)
            registry = _make_registry()
            config = EngineConfig(checkpoint_path=tmp_path)
            engine = OptimizationEngine(spec_phase1, registry, config=config)

            result_phase1 = engine.run_with_evaluator(_simple_evaluator)
            phase1_iters = result_phase1.total_iterations
            assert phase1_iters > 0

            engine.checkpoint()
            assert os.path.exists(tmp_path)

            # Phase 2: Resume from file and continue with larger budget.
            restored_state = CampaignState.resume_from_file(tmp_path)
            assert restored_state.iteration == phase1_iters

            # Update state for continued execution.
            spec_phase2 = _make_spec(max_samples=50, seed=42)
            restored_state.terminated = False
            restored_state.termination_reason = ""
            restored_state.spec = spec_phase2

            resumed_engine = OptimizationEngine(
                spec_phase2,
                _make_registry(),
                config=config,
                state=restored_state,
            )

            # Run a few more iterations then stop.
            target_additional = 3
            additional_run = 0
            for batch in resumed_engine.run():
                for trial in batch.trials:
                    trial.complete(kpi_values=_simple_evaluator(trial.parameters))
                additional_run += 1
                if additional_run >= target_additional:
                    resumed_engine.stop(reason="done")

            result = resumed_engine.result()
            assert result.total_iterations == phase1_iters + target_additional
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
