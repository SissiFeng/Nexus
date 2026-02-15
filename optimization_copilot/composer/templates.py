"""Built-in pipeline templates for common optimization strategies."""

from __future__ import annotations

from optimization_copilot.composer.models import (
    ComposerPipeline,
    PipelineStage,
    StageExitCondition,
)


# ── Pipeline Templates ─────────────────────────────────

PIPELINE_TEMPLATES: dict[str, ComposerPipeline] = {
    # ── 1. Exploration First ───────────────────────────
    "exploration_first": ComposerPipeline(
        name="exploration_first",
        description=(
            "Two-stage pipeline: broad space-filling exploration followed "
            "by model-guided exploitation."
        ),
        stages=[
            PipelineStage(
                stage_id="explore",
                backend_name="latin_hypercube",
                iteration_fraction=0.2,
                exit_condition_type=StageExitCondition.ITERATION_FRACTION,
                exploration_override=0.9,
                reason="Space-filling exploration to build initial surrogate.",
            ),
            PipelineStage(
                stage_id="exploit",
                backend_name="tpe",
                iteration_fraction=0.8,
                exit_condition_type=StageExitCondition.ITERATION_FRACTION,
                exploration_override=0.2,
                reason="Model-guided exploitation to converge on optimum.",
            ),
        ],
    ),

    # ── 2. Screening Then Optimize ─────────────────────
    "screening_then_optimize": ComposerPipeline(
        name="screening_then_optimize",
        description=(
            "Three-stage pipeline: screening to identify promising regions, "
            "learning phase with stagnation detection, then focused polishing."
        ),
        stages=[
            PipelineStage(
                stage_id="screen",
                backend_name="latin_hypercube",
                iteration_fraction=0.15,
                exit_condition_type=StageExitCondition.ITERATION_FRACTION,
                exploration_override=0.95,
                reason="Broad screening to identify promising parameter regions.",
            ),
            PipelineStage(
                stage_id="learn",
                backend_name="tpe",
                iteration_fraction=0.55,
                exit_conditions={"kpi_plateau_length": 8.0},
                exit_condition_type=StageExitCondition.DIAGNOSTIC_THRESHOLD,
                exploration_override=None,
                reason="Model-based learning with early exit on plateau.",
            ),
            PipelineStage(
                stage_id="polish",
                backend_name="tpe",
                iteration_fraction=0.3,
                exit_condition_type=StageExitCondition.ITERATION_FRACTION,
                exploration_override=0.1,
                reason="Low-exploration polishing around best-known region.",
            ),
        ],
    ),

    # ── 3. Restart On Stagnation ───────────────────────
    "restart_on_stagnation": ComposerPipeline(
        name="restart_on_stagnation",
        description=(
            "Three-stage pipeline with loop-back: explores, optimizes, and "
            "restarts from a fresh random sample when stagnation is detected."
        ),
        stages=[
            PipelineStage(
                stage_id="initial_explore",
                backend_name="latin_hypercube",
                iteration_fraction=0.15,
                exit_condition_type=StageExitCondition.ITERATION_FRACTION,
                exploration_override=0.9,
                reason="Initial space-filling exploration.",
            ),
            PipelineStage(
                stage_id="optimize",
                backend_name="tpe",
                iteration_fraction=0.7,
                exit_conditions={"kpi_plateau_length": 10.0},
                exit_condition_type=StageExitCondition.STAGNATION_DETECTED,
                exploration_override=None,
                reason="Model-guided optimization with stagnation escape.",
            ),
            PipelineStage(
                stage_id="restart_explore",
                backend_name="random",
                iteration_fraction=0.15,
                exit_condition_type=StageExitCondition.ITERATION_FRACTION,
                exploration_override=1.0,
                reason="Pure random restart to escape local optima.",
            ),
        ],
        loop_on_stagnation=True,
        restart_stage_id="initial_explore",
    ),

    # ── 4. Exploit Heavy ───────────────────────────────
    "exploit_heavy": ComposerPipeline(
        name="exploit_heavy",
        description=(
            "Two-stage pipeline: brief random warm-up followed by extended "
            "model-guided exploitation for low-noise, small-data problems."
        ),
        stages=[
            PipelineStage(
                stage_id="warmup",
                backend_name="random",
                iteration_fraction=0.1,
                exit_condition_type=StageExitCondition.ITERATION_FRACTION,
                exploration_override=0.8,
                reason="Brief random warm-up to seed the surrogate model.",
            ),
            PipelineStage(
                stage_id="exploit",
                backend_name="tpe",
                iteration_fraction=0.9,
                exit_condition_type=StageExitCondition.ITERATION_FRACTION,
                exploration_override=0.15,
                reason="Heavy exploitation for rapid convergence.",
            ),
        ],
    ),
}
