"""Tests for the Algorithm Composer (pipeline selection and orchestration).

Verifies:
- PipelineStage construction, defaults, and serialization round-trips
- ComposerPipeline multi-stage construction, accessors, and serialization
- StageTransition and PipelineOutcome round-trip serialization
- PipelineRecord construction and serialization
- Compose heuristic: HIGH noise -> restart_on_stagnation,
  TINY+LOW -> exploit_heavy, budget>30 -> screening_then_optimize, default -> exploration_first
- Backend filtering and substitution for unavailable backends
- Learned pipeline preferred when >= 3 uses with positive win rate
- select_stage iteration-fraction transitions at correct boundaries
- select_stage diagnostic-threshold transitions
- min_iterations respected (won't exit before min)
- max_iterations forced exit
- Loop on stagnation restart
- Single-stage pipeline stays on same stage
- Recording outcomes updates records and ranking reflects win rates
- Low-use records don't affect compose
- Determinism: same (fingerprint, budget, seed) -> same pipeline
- Edge cases: empty pipeline raises ValueError, all backends unavailable,
  reset clears state
"""

from __future__ import annotations

import pytest

from optimization_copilot.core.models import (
    CampaignSnapshot,
    DataScale,
    FeasibleRegion,
    NoiseRegime,
    Observation,
    ParameterSpec,
    ProblemFingerprint,
    VariableType,
)
from optimization_copilot.composer.models import (
    ComposerPipeline,
    PipelineOutcome,
    PipelineRecord,
    PipelineStage,
    StageExitCondition,
    StageTransition,
)
from optimization_copilot.composer.composer import AlgorithmComposer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fp(**kwargs) -> ProblemFingerprint:
    """Build a ProblemFingerprint with sensible defaults, overridden by kwargs."""
    return ProblemFingerprint(**kwargs)


def _make_snapshot(n_obs: int = 20) -> CampaignSnapshot:
    """Build a minimal CampaignSnapshot with *n_obs* observations."""
    params = [ParameterSpec(name="x", type=VariableType.CONTINUOUS, lower=0.0, upper=1.0)]
    observations = [
        Observation(
            iteration=i,
            parameters={"x": float(i) / max(n_obs, 1)},
            kpi_values={"y": float(i)},
        )
        for i in range(n_obs)
    ]
    return CampaignSnapshot(
        campaign_id="test-campaign",
        parameter_specs=params,
        observations=observations,
        objective_names=["y"],
        objective_directions=["minimize"],
        current_iteration=n_obs,
    )


def _make_outcome(
    pipeline_name: str = "exploration_first",
    fp_key: str = "test_key",
    *,
    best_kpi: float = 1.0,
    convergence_speed: float = 0.5,
    failure_rate: float = 0.0,
    is_winner: bool = False,
) -> PipelineOutcome:
    """Build a PipelineOutcome for testing."""
    return PipelineOutcome(
        pipeline_name=pipeline_name,
        fingerprint_key=fp_key,
        n_iterations=100,
        best_kpi=best_kpi,
        convergence_speed=convergence_speed,
        failure_rate=failure_rate,
        is_winner=is_winner,
    )


# ---------------------------------------------------------------------------
# Tests: PipelineStage
# ---------------------------------------------------------------------------


class TestPipelineStage:
    def test_construction_defaults(self):
        """PipelineStage should have sensible defaults for optional fields."""
        stage = PipelineStage(stage_id="s1", backend_name="tpe")
        assert stage.stage_id == "s1"
        assert stage.backend_name == "tpe"
        assert stage.iteration_fraction == 0.0
        assert stage.min_iterations == 1
        assert stage.max_iterations == 0
        assert stage.exit_conditions == {}
        assert stage.exit_condition_type == StageExitCondition.ITERATION_FRACTION
        assert stage.phase_trigger is None
        assert stage.exploration_override is None
        assert stage.reason == ""

    def test_construction_with_all_fields(self):
        """PipelineStage can be constructed with all fields explicitly set."""
        stage = PipelineStage(
            stage_id="explore",
            backend_name="latin_hypercube",
            iteration_fraction=0.3,
            min_iterations=5,
            max_iterations=50,
            exit_conditions={"plateau": 10.0},
            exit_condition_type=StageExitCondition.DIAGNOSTIC_THRESHOLD,
            phase_trigger="learning",
            exploration_override=0.9,
            reason="Test reason",
        )
        assert stage.stage_id == "explore"
        assert stage.iteration_fraction == 0.3
        assert stage.min_iterations == 5
        assert stage.max_iterations == 50
        assert stage.exit_conditions == {"plateau": 10.0}
        assert stage.exit_condition_type == StageExitCondition.DIAGNOSTIC_THRESHOLD
        assert stage.phase_trigger == "learning"
        assert stage.exploration_override == 0.9
        assert stage.reason == "Test reason"

    def test_to_dict_from_dict_roundtrip(self):
        """PipelineStage survives to_dict -> from_dict round-trip."""
        original = PipelineStage(
            stage_id="s1",
            backend_name="tpe",
            iteration_fraction=0.5,
            min_iterations=3,
            max_iterations=20,
            exit_conditions={"kpi_plateau_length": 8.0},
            exit_condition_type=StageExitCondition.STAGNATION_DETECTED,
            phase_trigger="exploitation",
            exploration_override=0.2,
            reason="Test roundtrip",
        )
        rebuilt = PipelineStage.from_dict(original.to_dict())

        assert rebuilt.stage_id == original.stage_id
        assert rebuilt.backend_name == original.backend_name
        assert rebuilt.iteration_fraction == original.iteration_fraction
        assert rebuilt.min_iterations == original.min_iterations
        assert rebuilt.max_iterations == original.max_iterations
        assert rebuilt.exit_conditions == original.exit_conditions
        assert rebuilt.exit_condition_type == original.exit_condition_type
        assert rebuilt.phase_trigger == original.phase_trigger
        assert rebuilt.exploration_override == original.exploration_override
        assert rebuilt.reason == original.reason

    def test_to_dict_exit_condition_type_is_string(self):
        """to_dict should serialize exit_condition_type as its string value."""
        stage = PipelineStage(
            stage_id="s1",
            backend_name="tpe",
            exit_condition_type=StageExitCondition.DIAGNOSTIC_THRESHOLD,
        )
        d = stage.to_dict()
        assert d["exit_condition_type"] == "diagnostic_threshold"
        assert isinstance(d["exit_condition_type"], str)


# ---------------------------------------------------------------------------
# Tests: ComposerPipeline
# ---------------------------------------------------------------------------


class TestComposerPipeline:
    def test_multi_stage_construction(self):
        """ComposerPipeline holds multiple stages correctly."""
        s1 = PipelineStage(stage_id="a", backend_name="random", iteration_fraction=0.3)
        s2 = PipelineStage(stage_id="b", backend_name="tpe", iteration_fraction=0.7)
        pipeline = ComposerPipeline(
            name="test_pipe",
            description="A test pipeline",
            stages=[s1, s2],
        )
        assert pipeline.name == "test_pipe"
        assert pipeline.n_stages == 2
        assert pipeline.stage_ids == ["a", "b"]

    def test_get_stage_found(self):
        """get_stage returns the correct stage by id."""
        s1 = PipelineStage(stage_id="explore", backend_name="random")
        s2 = PipelineStage(stage_id="exploit", backend_name="tpe")
        pipeline = ComposerPipeline(name="p", description="d", stages=[s1, s2])

        found = pipeline.get_stage("exploit")
        assert found is not None
        assert found.stage_id == "exploit"
        assert found.backend_name == "tpe"

    def test_get_stage_not_found(self):
        """get_stage returns None for a missing stage id."""
        s1 = PipelineStage(stage_id="explore", backend_name="random")
        pipeline = ComposerPipeline(name="p", description="d", stages=[s1])

        assert pipeline.get_stage("nonexistent") is None

    def test_n_stages_empty(self):
        """n_stages is 0 for an empty pipeline."""
        pipeline = ComposerPipeline(name="empty", description="no stages")
        assert pipeline.n_stages == 0
        assert pipeline.stage_ids == []

    def test_defaults(self):
        """ComposerPipeline has correct defaults."""
        pipeline = ComposerPipeline(name="p", description="d")
        assert pipeline.loop_on_stagnation is False
        assert pipeline.restart_stage_id is None
        assert pipeline.metadata == {}

    def test_to_dict_from_dict_roundtrip(self):
        """ComposerPipeline survives to_dict -> from_dict."""
        original = ComposerPipeline(
            name="roundtrip",
            description="Testing serialization",
            stages=[
                PipelineStage(stage_id="a", backend_name="random", iteration_fraction=0.3),
                PipelineStage(stage_id="b", backend_name="tpe", iteration_fraction=0.7),
            ],
            loop_on_stagnation=True,
            restart_stage_id="a",
            metadata={"key": "value"},
        )
        rebuilt = ComposerPipeline.from_dict(original.to_dict())

        assert rebuilt.name == original.name
        assert rebuilt.description == original.description
        assert rebuilt.n_stages == original.n_stages
        assert rebuilt.stage_ids == original.stage_ids
        assert rebuilt.loop_on_stagnation == original.loop_on_stagnation
        assert rebuilt.restart_stage_id == original.restart_stage_id
        assert rebuilt.metadata == original.metadata

        # Check nested stages also survived.
        for orig_s, rebuilt_s in zip(original.stages, rebuilt.stages):
            assert orig_s.stage_id == rebuilt_s.stage_id
            assert orig_s.backend_name == rebuilt_s.backend_name
            assert orig_s.iteration_fraction == rebuilt_s.iteration_fraction


# ---------------------------------------------------------------------------
# Tests: StageTransition
# ---------------------------------------------------------------------------


class TestStageTransition:
    def test_construction(self):
        """StageTransition stores its fields correctly."""
        t = StageTransition(
            from_stage_id="a",
            to_stage_id="b",
            iteration=10,
            trigger="iteration_fraction",
            diagnostics_at_transition={"kpi": 0.5},
        )
        assert t.from_stage_id == "a"
        assert t.to_stage_id == "b"
        assert t.iteration == 10
        assert t.trigger == "iteration_fraction"
        assert t.diagnostics_at_transition == {"kpi": 0.5}

    def test_to_dict_from_dict_roundtrip(self):
        """StageTransition survives to_dict -> from_dict."""
        original = StageTransition(
            from_stage_id="x",
            to_stage_id="y",
            iteration=42,
            trigger="stagnation_detected",
            diagnostics_at_transition={"plateau": 12.0, "speed": 0.1},
        )
        rebuilt = StageTransition.from_dict(original.to_dict())
        assert rebuilt.from_stage_id == original.from_stage_id
        assert rebuilt.to_stage_id == original.to_stage_id
        assert rebuilt.iteration == original.iteration
        assert rebuilt.trigger == original.trigger
        assert rebuilt.diagnostics_at_transition == original.diagnostics_at_transition

    def test_default_diagnostics_empty(self):
        """Default diagnostics_at_transition is an empty dict."""
        t = StageTransition(from_stage_id="a", to_stage_id="b", iteration=0, trigger="t")
        assert t.diagnostics_at_transition == {}


# ---------------------------------------------------------------------------
# Tests: PipelineOutcome
# ---------------------------------------------------------------------------


class TestPipelineOutcome:
    def test_construction(self):
        """PipelineOutcome stores all fields."""
        o = PipelineOutcome(
            pipeline_name="p1",
            fingerprint_key="fp1",
            n_iterations=100,
            best_kpi=0.5,
            convergence_speed=0.8,
            failure_rate=0.05,
            is_winner=True,
        )
        assert o.pipeline_name == "p1"
        assert o.n_iterations == 100
        assert o.best_kpi == 0.5
        assert o.is_winner is True

    def test_to_dict_from_dict_roundtrip(self):
        """PipelineOutcome survives to_dict -> from_dict."""
        transition = StageTransition(
            from_stage_id="a", to_stage_id="b", iteration=10, trigger="test"
        )
        original = PipelineOutcome(
            pipeline_name="p1",
            fingerprint_key="fp1",
            n_iterations=50,
            best_kpi=1.5,
            convergence_speed=0.7,
            failure_rate=0.1,
            transitions=[transition],
            is_winner=True,
        )
        rebuilt = PipelineOutcome.from_dict(original.to_dict())
        assert rebuilt.pipeline_name == original.pipeline_name
        assert rebuilt.fingerprint_key == original.fingerprint_key
        assert rebuilt.n_iterations == original.n_iterations
        assert rebuilt.best_kpi == original.best_kpi
        assert rebuilt.convergence_speed == original.convergence_speed
        assert rebuilt.failure_rate == original.failure_rate
        assert rebuilt.is_winner == original.is_winner
        assert len(rebuilt.transitions) == 1
        assert rebuilt.transitions[0].from_stage_id == "a"

    def test_default_transitions_empty(self):
        """Default transitions is an empty list."""
        o = PipelineOutcome(
            pipeline_name="p",
            fingerprint_key="fp",
            n_iterations=10,
            best_kpi=0.0,
            convergence_speed=0.0,
            failure_rate=0.0,
        )
        assert o.transitions == []
        assert o.is_winner is False


# ---------------------------------------------------------------------------
# Tests: PipelineRecord
# ---------------------------------------------------------------------------


class TestPipelineRecord:
    def test_construction_defaults(self):
        """PipelineRecord has sensible defaults."""
        rec = PipelineRecord(pipeline_name="p", fingerprint_key="fp")
        assert rec.n_uses == 0
        assert rec.win_count == 0
        assert rec.avg_best_kpi == 0.0
        assert rec.avg_convergence_speed == 0.0
        assert rec.avg_failure_rate == 0.0

    def test_to_dict_from_dict_roundtrip(self):
        """PipelineRecord survives to_dict -> from_dict."""
        original = PipelineRecord(
            pipeline_name="p1",
            fingerprint_key="fp1",
            n_uses=5,
            win_count=3,
            avg_best_kpi=0.42,
            avg_convergence_speed=0.7,
            avg_failure_rate=0.05,
        )
        rebuilt = PipelineRecord.from_dict(original.to_dict())
        assert rebuilt.pipeline_name == original.pipeline_name
        assert rebuilt.fingerprint_key == original.fingerprint_key
        assert rebuilt.n_uses == original.n_uses
        assert rebuilt.win_count == original.win_count
        assert rebuilt.avg_best_kpi == pytest.approx(original.avg_best_kpi)
        assert rebuilt.avg_convergence_speed == pytest.approx(original.avg_convergence_speed)
        assert rebuilt.avg_failure_rate == pytest.approx(original.avg_failure_rate)


# ---------------------------------------------------------------------------
# Tests: Compose (heuristic + learning selection)
# ---------------------------------------------------------------------------


class TestCompose:
    def test_high_noise_selects_restart_on_stagnation(self):
        """HIGH noise fingerprint should select restart_on_stagnation."""
        fp = _make_fp(noise_regime=NoiseRegime.HIGH)
        composer = AlgorithmComposer()
        pipeline = composer.compose(fp, budget=20)
        assert pipeline.name == "restart_on_stagnation"

    def test_fragmented_region_selects_restart_on_stagnation(self):
        """FRAGMENTED feasible region should select restart_on_stagnation."""
        fp = _make_fp(feasible_region=FeasibleRegion.FRAGMENTED)
        composer = AlgorithmComposer()
        pipeline = composer.compose(fp, budget=20)
        assert pipeline.name == "restart_on_stagnation"

    def test_tiny_low_noise_selects_exploit_heavy(self):
        """TINY data + LOW noise should select exploit_heavy."""
        fp = _make_fp(data_scale=DataScale.TINY, noise_regime=NoiseRegime.LOW)
        composer = AlgorithmComposer()
        pipeline = composer.compose(fp, budget=20)
        assert pipeline.name == "exploit_heavy"

    def test_budget_over_30_selects_screening_then_optimize(self):
        """Budget > 30 should select screening_then_optimize (non-extreme fingerprint)."""
        fp = _make_fp(
            noise_regime=NoiseRegime.MEDIUM,
            data_scale=DataScale.MODERATE,
        )
        composer = AlgorithmComposer()
        pipeline = composer.compose(fp, budget=50)
        assert pipeline.name == "screening_then_optimize"

    def test_default_selects_exploration_first(self):
        """Default fingerprint with small budget should select exploration_first."""
        fp = _make_fp(
            noise_regime=NoiseRegime.MEDIUM,
            data_scale=DataScale.MODERATE,
        )
        composer = AlgorithmComposer()
        pipeline = composer.compose(fp, budget=20)
        assert pipeline.name == "exploration_first"

    def test_backend_filtering_substitutes_unavailable(self):
        """Unavailable backends should be substituted with the first available."""
        fp = _make_fp(noise_regime=NoiseRegime.MEDIUM, data_scale=DataScale.MODERATE)
        # Only "my_backend" is available; latin_hypercube and tpe are not.
        composer = AlgorithmComposer(available_backends=["my_backend"])
        pipeline = composer.compose(fp, budget=20)

        # All stages should have been substituted to "my_backend".
        for stage in pipeline.stages:
            assert stage.backend_name == "my_backend"

    def test_learned_pipeline_preferred_with_sufficient_uses(self):
        """A learned pipeline with >= 3 uses and positive win rate is preferred."""
        fp = _make_fp(
            noise_regime=NoiseRegime.MEDIUM,
            data_scale=DataScale.MODERATE,
        )
        composer = AlgorithmComposer()

        # Record outcomes to build a learned preference for exploit_heavy.
        for i in range(5):
            outcome = _make_outcome(
                pipeline_name="exploit_heavy",
                best_kpi=0.1,
                convergence_speed=0.9,
                failure_rate=0.0,
                is_winner=True,
            )
            # Get the actual pipeline to record against.
            pipeline = ComposerPipeline(
                name="exploit_heavy", description="test",
                stages=[PipelineStage(stage_id="s", backend_name="tpe")],
            )
            composer.record_outcome(pipeline, fp, outcome)

        # Now compose should prefer the learned pipeline.
        result = composer.compose(fp, budget=20)
        assert result.name == "exploit_heavy"

    def test_learned_pipeline_not_preferred_with_insufficient_uses(self):
        """A pipeline with < 3 uses should not override heuristic selection."""
        fp = _make_fp(
            noise_regime=NoiseRegime.MEDIUM,
            data_scale=DataScale.MODERATE,
        )
        composer = AlgorithmComposer()

        # Record only 2 outcomes (below the 3-use threshold).
        for _ in range(2):
            outcome = _make_outcome(
                pipeline_name="exploit_heavy",
                best_kpi=0.1,
                convergence_speed=0.9,
                failure_rate=0.0,
                is_winner=True,
            )
            pipeline = ComposerPipeline(
                name="exploit_heavy", description="test",
                stages=[PipelineStage(stage_id="s", backend_name="tpe")],
            )
            composer.record_outcome(pipeline, fp, outcome)

        # Heuristic should still win (default: exploration_first for budget <= 30).
        result = composer.compose(fp, budget=20)
        assert result.name == "exploration_first"


# ---------------------------------------------------------------------------
# Tests: select_stage
# ---------------------------------------------------------------------------


class TestSelectStage:
    def _two_stage_pipeline(self) -> ComposerPipeline:
        """Create a simple 2-stage pipeline with iteration_fraction exit."""
        return ComposerPipeline(
            name="two_stage",
            description="test",
            stages=[
                PipelineStage(
                    stage_id="explore",
                    backend_name="random",
                    iteration_fraction=0.4,
                    exit_condition_type=StageExitCondition.ITERATION_FRACTION,
                ),
                PipelineStage(
                    stage_id="exploit",
                    backend_name="tpe",
                    iteration_fraction=0.6,
                    exit_condition_type=StageExitCondition.ITERATION_FRACTION,
                ),
            ],
        )

    def test_iteration_fraction_stays_in_first_stage(self):
        """Before reaching the fraction boundary, should stay in the first stage."""
        composer = AlgorithmComposer()
        pipeline = self._two_stage_pipeline()
        budget = 100

        # Iteration 0: fraction = 0/100 = 0, threshold = 0.4 * 100 = 40
        stage = composer.select_stage(pipeline, iteration=0, budget=budget, diagnostics={}, snapshot=None)
        assert stage.stage_id == "explore"

    def test_iteration_fraction_transitions_at_boundary(self):
        """At the fraction boundary, should transition to the next stage."""
        composer = AlgorithmComposer()
        pipeline = self._two_stage_pipeline()
        budget = 100

        # Iterate through enough iterations to trigger transition.
        # Stage 1 fraction = 0.4 => exits at iteration 40 (40 iterations in stage).
        for i in range(40):
            composer.select_stage(pipeline, iteration=i, budget=budget, diagnostics={}, snapshot=None)

        # At iteration 40, the stage should have transitioned.
        stage = composer.select_stage(pipeline, iteration=40, budget=budget, diagnostics={}, snapshot=None)
        assert stage.stage_id == "exploit"

    def test_diagnostic_threshold_transition(self):
        """Diagnostic threshold exit condition triggers transition."""
        pipeline = ComposerPipeline(
            name="diag_test",
            description="test",
            stages=[
                PipelineStage(
                    stage_id="learn",
                    backend_name="tpe",
                    iteration_fraction=0.5,
                    exit_conditions={"kpi_plateau_length": 8.0},
                    exit_condition_type=StageExitCondition.DIAGNOSTIC_THRESHOLD,
                ),
                PipelineStage(
                    stage_id="polish",
                    backend_name="tpe",
                    iteration_fraction=0.5,
                    exit_condition_type=StageExitCondition.ITERATION_FRACTION,
                ),
            ],
        )
        composer = AlgorithmComposer()
        budget = 100

        # Iteration 1 (past min_iterations=1), with plateau >= 8 -> should exit.
        stage = composer.select_stage(
            pipeline, iteration=0, budget=budget, diagnostics={}, snapshot=None,
        )
        assert stage.stage_id == "learn"

        # Now at iteration 1 (1 iteration in stage), plateau signal triggers exit.
        stage = composer.select_stage(
            pipeline, iteration=1, budget=budget,
            diagnostics={"kpi_plateau_length": 10.0},
            snapshot=None,
        )
        assert stage.stage_id == "polish"

    def test_min_iterations_respected(self):
        """Stage should not exit before min_iterations."""
        pipeline = ComposerPipeline(
            name="min_iter_test",
            description="test",
            stages=[
                PipelineStage(
                    stage_id="first",
                    backend_name="random",
                    iteration_fraction=0.0,  # Would trigger immediate exit without min
                    min_iterations=5,
                    exit_condition_type=StageExitCondition.ITERATION_FRACTION,
                ),
                PipelineStage(
                    stage_id="second",
                    backend_name="tpe",
                    iteration_fraction=1.0,
                    exit_condition_type=StageExitCondition.ITERATION_FRACTION,
                ),
            ],
        )
        composer = AlgorithmComposer()
        budget = 100

        # Iterations 0 through 4 should stay on "first" due to min_iterations=5.
        for i in range(5):
            stage = composer.select_stage(
                pipeline, iteration=i, budget=budget, diagnostics={}, snapshot=None,
            )
            assert stage.stage_id == "first", f"Expected 'first' at iteration {i}"

        # Iteration 5 should be able to exit (5 iterations in stage >= min_iterations=5).
        stage = composer.select_stage(
            pipeline, iteration=5, budget=budget, diagnostics={}, snapshot=None,
        )
        assert stage.stage_id == "second"

    def test_max_iterations_forced_exit(self):
        """Stage should force-exit at max_iterations even if condition not met."""
        pipeline = ComposerPipeline(
            name="max_iter_test",
            description="test",
            stages=[
                PipelineStage(
                    stage_id="first",
                    backend_name="random",
                    iteration_fraction=1.0,  # Very high fraction, won't naturally exit
                    min_iterations=1,
                    max_iterations=3,
                    exit_condition_type=StageExitCondition.ITERATION_FRACTION,
                ),
                PipelineStage(
                    stage_id="second",
                    backend_name="tpe",
                    iteration_fraction=0.5,
                    exit_condition_type=StageExitCondition.ITERATION_FRACTION,
                ),
            ],
        )
        composer = AlgorithmComposer()
        budget = 100

        # Iterations 0, 1, 2 should stay (iterations_in_stage < max_iterations=3).
        for i in range(3):
            stage = composer.select_stage(
                pipeline, iteration=i, budget=budget, diagnostics={}, snapshot=None,
            )
            assert stage.stage_id == "first", f"Expected 'first' at iteration {i}"

        # At iteration 3, iterations_in_stage=3 >= max_iterations=3, force exit.
        stage = composer.select_stage(
            pipeline, iteration=3, budget=budget, diagnostics={}, snapshot=None,
        )
        assert stage.stage_id == "second"

    def test_loop_on_stagnation_restart(self):
        """When the last stage exits and loop_on_stagnation=True, should loop back."""
        pipeline = ComposerPipeline(
            name="loop_test",
            description="test",
            stages=[
                PipelineStage(
                    stage_id="init",
                    backend_name="random",
                    iteration_fraction=0.1,
                    exit_condition_type=StageExitCondition.ITERATION_FRACTION,
                ),
                PipelineStage(
                    stage_id="opt",
                    backend_name="tpe",
                    iteration_fraction=0.3,
                    exit_condition_type=StageExitCondition.ITERATION_FRACTION,
                ),
                PipelineStage(
                    stage_id="restart",
                    backend_name="random",
                    iteration_fraction=0.1,
                    exit_condition_type=StageExitCondition.ITERATION_FRACTION,
                ),
            ],
            loop_on_stagnation=True,
            restart_stage_id="init",
        )
        composer = AlgorithmComposer()
        budget = 100

        # Advance through all three stages.
        # Stage 1: exits at iteration 10 (0.1 * 100).
        for i in range(10):
            composer.select_stage(pipeline, iteration=i, budget=budget, diagnostics={}, snapshot=None)

        stage = composer.select_stage(pipeline, iteration=10, budget=budget, diagnostics={}, snapshot=None)
        assert stage.stage_id == "opt"

        # Stage 2: exits at iteration 10 + 30 = 40 (0.3 * 100 = 30 in stage).
        for i in range(11, 40):
            composer.select_stage(pipeline, iteration=i, budget=budget, diagnostics={}, snapshot=None)

        stage = composer.select_stage(pipeline, iteration=40, budget=budget, diagnostics={}, snapshot=None)
        assert stage.stage_id == "restart"

        # Stage 3: exits at iteration 40 + 10 = 50 (0.1 * 100 = 10 in stage).
        for i in range(41, 50):
            composer.select_stage(pipeline, iteration=i, budget=budget, diagnostics={}, snapshot=None)

        # At iteration 50, last stage exits and should loop back to "init".
        stage = composer.select_stage(pipeline, iteration=50, budget=budget, diagnostics={}, snapshot=None)
        assert stage.stage_id == "init"

    def test_single_stage_pipeline_stays(self):
        """A single-stage pipeline should always return the same stage."""
        pipeline = ComposerPipeline(
            name="single",
            description="test",
            stages=[
                PipelineStage(
                    stage_id="only",
                    backend_name="tpe",
                    iteration_fraction=0.5,
                    exit_condition_type=StageExitCondition.ITERATION_FRACTION,
                ),
            ],
        )
        composer = AlgorithmComposer()
        budget = 100

        # Even past the fraction boundary, should stay on the only stage.
        for i in range(0, 80, 10):
            stage = composer.select_stage(
                pipeline, iteration=i, budget=budget, diagnostics={}, snapshot=None,
            )
            assert stage.stage_id == "only"


# ---------------------------------------------------------------------------
# Tests: Record and Rank
# ---------------------------------------------------------------------------


class TestRecordAndRank:
    def test_recording_outcomes_updates_records(self):
        """Recording an outcome should update the internal record."""
        fp = _make_fp()
        composer = AlgorithmComposer()
        pipeline = ComposerPipeline(
            name="test_pipe", description="test",
            stages=[PipelineStage(stage_id="s", backend_name="tpe")],
        )
        outcome = _make_outcome(
            pipeline_name="test_pipe",
            best_kpi=2.0,
            convergence_speed=0.8,
            failure_rate=0.1,
            is_winner=True,
        )
        composer.record_outcome(pipeline, fp, outcome)

        rankings = composer.rank_pipelines(fp)
        assert len(rankings) == 1
        assert rankings[0][0] == "test_pipe"

    def test_ranking_reflects_win_rates(self):
        """Higher win rate should produce higher ranking score."""
        fp = _make_fp()
        composer = AlgorithmComposer()

        pipe_a = ComposerPipeline(
            name="winner", description="test",
            stages=[PipelineStage(stage_id="s", backend_name="tpe")],
        )
        pipe_b = ComposerPipeline(
            name="loser", description="test",
            stages=[PipelineStage(stage_id="s", backend_name="random")],
        )

        # Record several wins for pipe_a.
        for _ in range(5):
            composer.record_outcome(
                pipe_a, fp,
                _make_outcome(pipeline_name="winner", is_winner=True, failure_rate=0.0),
            )

        # Record several losses for pipe_b.
        for _ in range(5):
            composer.record_outcome(
                pipe_b, fp,
                _make_outcome(pipeline_name="loser", is_winner=False, failure_rate=0.3),
            )

        rankings = composer.rank_pipelines(fp)
        names = [name for name, _ in rankings]
        assert names[0] == "winner"

    def test_low_use_records_dont_affect_compose(self):
        """Records with < 3 uses should not override heuristic in compose."""
        fp = _make_fp(
            noise_regime=NoiseRegime.MEDIUM,
            data_scale=DataScale.MODERATE,
        )
        composer = AlgorithmComposer()

        pipe = ComposerPipeline(
            name="exploit_heavy", description="test",
            stages=[PipelineStage(stage_id="s", backend_name="tpe")],
        )
        # Only 2 uses - below the 3-use threshold.
        for _ in range(2):
            composer.record_outcome(
                pipe, fp,
                _make_outcome(pipeline_name="exploit_heavy", is_winner=True, failure_rate=0.0),
            )

        # Heuristic should still be used (budget <= 30 -> exploration_first).
        result = composer.compose(fp, budget=20)
        assert result.name == "exploration_first"


# ---------------------------------------------------------------------------
# Tests: Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_inputs_same_pipeline(self):
        """Same (fingerprint, budget, seed) should always produce same pipeline."""
        fp = _make_fp(noise_regime=NoiseRegime.MEDIUM, data_scale=DataScale.MODERATE)
        composer = AlgorithmComposer()

        p1 = composer.compose(fp, budget=50, seed=42)
        p2 = composer.compose(fp, budget=50, seed=42)

        assert p1.name == p2.name
        assert p1.n_stages == p2.n_stages
        assert p1.stage_ids == p2.stage_ids

    def test_different_fingerprints_may_differ(self):
        """Different fingerprints can produce different pipelines."""
        fp_high_noise = _make_fp(noise_regime=NoiseRegime.HIGH)
        fp_low_noise = _make_fp(
            noise_regime=NoiseRegime.MEDIUM, data_scale=DataScale.MODERATE,
        )
        composer = AlgorithmComposer()

        p1 = composer.compose(fp_high_noise, budget=20)
        p2 = composer.compose(fp_low_noise, budget=20)

        # HIGH noise -> restart_on_stagnation; MEDIUM+MODERATE+budget<=30 -> exploration_first
        assert p1.name != p2.name

    def test_multiple_calls_consistent(self):
        """Multiple compose calls with same inputs produce identical results."""
        fp = _make_fp()
        composer = AlgorithmComposer()

        results = [composer.compose(fp, budget=20, seed=0) for _ in range(10)]
        names = [r.name for r in results]
        assert len(set(names)) == 1


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_pipeline_raises_valueerror(self):
        """select_stage on an empty pipeline should raise ValueError."""
        pipeline = ComposerPipeline(name="empty", description="no stages")
        composer = AlgorithmComposer()

        with pytest.raises(ValueError, match="Pipeline has no stages"):
            composer.select_stage(
                pipeline, iteration=0, budget=100, diagnostics={}, snapshot=None,
            )

    def test_all_backends_unavailable_still_works(self):
        """Even if no template backends are available, substitution fills in."""
        fp = _make_fp()
        # "custom_only" is not in any template, but all stages will be substituted.
        composer = AlgorithmComposer(available_backends=["custom_only"])
        pipeline = composer.compose(fp, budget=20)

        # All stages should have been substituted to "custom_only".
        for stage in pipeline.stages:
            assert stage.backend_name == "custom_only"

    def test_empty_available_backends_no_substitution(self):
        """Empty available_backends list means no substitution occurs."""
        fp = _make_fp()
        composer = AlgorithmComposer(available_backends=[])
        pipeline = composer.compose(fp, budget=20)

        # With empty backends list, _filter_to_available returns pipeline unchanged.
        # The pipeline should still have its original backends.
        assert pipeline.n_stages > 0

    def test_reset_clears_state(self):
        """reset() should clear the active stage index, start iteration, and transitions."""
        composer = AlgorithmComposer()
        pipeline = ComposerPipeline(
            name="test",
            description="test",
            stages=[
                PipelineStage(stage_id="a", backend_name="random", iteration_fraction=0.3),
                PipelineStage(stage_id="b", backend_name="tpe", iteration_fraction=0.7),
            ],
        )
        budget = 100

        # Advance to second stage.
        for i in range(31):
            composer.select_stage(pipeline, iteration=i, budget=budget, diagnostics={}, snapshot=None)

        assert composer.active_stage_index > 0 or len(composer.transitions) > 0

        # Reset.
        composer.reset()
        assert composer.active_stage_index == 0
        assert composer.transitions == []

    def test_composer_serialization_roundtrip(self):
        """AlgorithmComposer survives to_dict -> from_dict."""
        fp = _make_fp()
        composer = AlgorithmComposer()

        # Record some outcomes to populate state.
        pipe = ComposerPipeline(
            name="exploration_first", description="test",
            stages=[PipelineStage(stage_id="s", backend_name="tpe")],
        )
        for _ in range(3):
            composer.record_outcome(
                pipe, fp,
                _make_outcome(pipeline_name="exploration_first", is_winner=True),
            )

        data = composer.to_dict()
        restored = AlgorithmComposer.from_dict(data)

        assert restored.active_stage_index == composer.active_stage_index
        assert len(restored.transitions) == len(composer.transitions)
        # Verify records were restored.
        rankings = restored.rank_pipelines(fp)
        assert len(rankings) > 0
