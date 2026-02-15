"""Tests for built-in pipeline templates (composer/templates.py).

Verifies:
- Each template exists and has valid structure (stages, backend_name)
- exploration_first: 2 stages, correct backends, fractions sum to ~1.0
- screening_then_optimize: 3 stages, correct backends, diagnostic threshold exit
- restart_on_stagnation: loop_on_stagnation=True, restart_stage_id set, 3 stages
- exploit_heavy: 2 stages, warmup fraction small, exploit fraction large
- Serialization round-trip for all templates
"""

from __future__ import annotations

import pytest

from optimization_copilot.composer.models import (
    ComposerPipeline,
    PipelineStage,
    StageExitCondition,
)
from optimization_copilot.composer.templates import PIPELINE_TEMPLATES


# ---------------------------------------------------------------------------
# Expected template names
# ---------------------------------------------------------------------------

EXPECTED_TEMPLATES = [
    "exploration_first",
    "screening_then_optimize",
    "restart_on_stagnation",
    "exploit_heavy",
]


# ---------------------------------------------------------------------------
# Tests: Template Structure
# ---------------------------------------------------------------------------


class TestTemplateStructure:
    def test_all_expected_templates_exist(self):
        """All expected template names should be present in PIPELINE_TEMPLATES."""
        for name in EXPECTED_TEMPLATES:
            assert name in PIPELINE_TEMPLATES, f"Template '{name}' not found"

    def test_templates_are_composer_pipelines(self):
        """Each template should be a ComposerPipeline instance."""
        for name, template in PIPELINE_TEMPLATES.items():
            assert isinstance(template, ComposerPipeline), (
                f"Template '{name}' is not a ComposerPipeline"
            )

    def test_all_templates_have_stages(self):
        """Every template should have at least one stage."""
        for name, template in PIPELINE_TEMPLATES.items():
            assert template.n_stages > 0, f"Template '{name}' has no stages"

    def test_all_stages_have_backend_name(self):
        """Every stage in every template should have a non-empty backend_name."""
        for name, template in PIPELINE_TEMPLATES.items():
            for stage in template.stages:
                assert stage.backend_name, (
                    f"Stage '{stage.stage_id}' in '{name}' has empty backend_name"
                )

    def test_all_stages_have_stage_id(self):
        """Every stage in every template should have a non-empty stage_id."""
        for name, template in PIPELINE_TEMPLATES.items():
            for stage in template.stages:
                assert stage.stage_id, (
                    f"Stage in '{name}' has empty stage_id"
                )

    def test_stage_ids_unique_within_template(self):
        """Stage IDs should be unique within each template."""
        for name, template in PIPELINE_TEMPLATES.items():
            ids = [s.stage_id for s in template.stages]
            assert len(ids) == len(set(ids)), (
                f"Template '{name}' has duplicate stage IDs: {ids}"
            )

    def test_template_name_matches_key(self):
        """Each template's name field should match its dictionary key."""
        for key, template in PIPELINE_TEMPLATES.items():
            assert template.name == key, (
                f"Template key '{key}' does not match name '{template.name}'"
            )

    def test_templates_have_descriptions(self):
        """Each template should have a non-empty description."""
        for name, template in PIPELINE_TEMPLATES.items():
            assert template.description, f"Template '{name}' has empty description"

    def test_all_stages_have_valid_exit_condition_type(self):
        """Every stage should have a valid StageExitCondition type."""
        for name, template in PIPELINE_TEMPLATES.items():
            for stage in template.stages:
                assert isinstance(stage.exit_condition_type, StageExitCondition), (
                    f"Stage '{stage.stage_id}' in '{name}' has invalid exit_condition_type"
                )


# ---------------------------------------------------------------------------
# Tests: exploration_first Template
# ---------------------------------------------------------------------------


class TestExplorationFirst:
    @pytest.fixture
    def template(self) -> ComposerPipeline:
        return PIPELINE_TEMPLATES["exploration_first"]

    def test_has_two_stages(self, template: ComposerPipeline):
        """exploration_first should have exactly 2 stages."""
        assert template.n_stages == 2

    def test_first_stage_is_latin_hypercube(self, template: ComposerPipeline):
        """First stage should use latin_hypercube backend."""
        assert template.stages[0].backend_name == "latin_hypercube"

    def test_second_stage_is_tpe(self, template: ComposerPipeline):
        """Second stage should use tpe backend."""
        assert template.stages[1].backend_name == "tpe"

    def test_fractions_sum_to_one(self, template: ComposerPipeline):
        """Iteration fractions should sum to approximately 1.0."""
        total = sum(s.iteration_fraction for s in template.stages)
        assert total == pytest.approx(1.0, abs=0.05)

    def test_first_stage_high_exploration(self, template: ComposerPipeline):
        """First stage should have high exploration override."""
        assert template.stages[0].exploration_override is not None
        assert template.stages[0].exploration_override >= 0.8

    def test_second_stage_low_exploration(self, template: ComposerPipeline):
        """Second stage should have low exploration override."""
        assert template.stages[1].exploration_override is not None
        assert template.stages[1].exploration_override <= 0.3

    def test_no_loop(self, template: ComposerPipeline):
        """exploration_first should not loop on stagnation."""
        assert template.loop_on_stagnation is False
        assert template.restart_stage_id is None

    def test_stage_ids(self, template: ComposerPipeline):
        """exploration_first stage IDs should be 'explore' and 'exploit'."""
        assert template.stage_ids == ["explore", "exploit"]

    def test_both_use_iteration_fraction_exit(self, template: ComposerPipeline):
        """Both stages should use ITERATION_FRACTION exit condition."""
        for stage in template.stages:
            assert stage.exit_condition_type == StageExitCondition.ITERATION_FRACTION


# ---------------------------------------------------------------------------
# Tests: screening_then_optimize Template
# ---------------------------------------------------------------------------


class TestScreeningThenOptimize:
    @pytest.fixture
    def template(self) -> ComposerPipeline:
        return PIPELINE_TEMPLATES["screening_then_optimize"]

    def test_has_three_stages(self, template: ComposerPipeline):
        """screening_then_optimize should have exactly 3 stages."""
        assert template.n_stages == 3

    def test_correct_backends(self, template: ComposerPipeline):
        """Stages should use latin_hypercube, tpe, tpe."""
        backends = [s.backend_name for s in template.stages]
        assert backends == ["latin_hypercube", "tpe", "tpe"]

    def test_has_diagnostic_threshold_exit(self, template: ComposerPipeline):
        """At least one stage should use DIAGNOSTIC_THRESHOLD exit condition."""
        exit_types = [s.exit_condition_type for s in template.stages]
        assert StageExitCondition.DIAGNOSTIC_THRESHOLD in exit_types

    def test_learn_stage_has_exit_conditions(self, template: ComposerPipeline):
        """The 'learn' stage should have exit_conditions for diagnostic threshold."""
        learn_stage = template.get_stage("learn")
        assert learn_stage is not None
        assert len(learn_stage.exit_conditions) > 0
        assert "kpi_plateau_length" in learn_stage.exit_conditions

    def test_fractions_sum_to_one(self, template: ComposerPipeline):
        """Iteration fractions should sum to approximately 1.0."""
        total = sum(s.iteration_fraction for s in template.stages)
        assert total == pytest.approx(1.0, abs=0.05)

    def test_screen_stage_high_exploration(self, template: ComposerPipeline):
        """Screen stage should have very high exploration override."""
        screen = template.get_stage("screen")
        assert screen is not None
        assert screen.exploration_override is not None
        assert screen.exploration_override >= 0.9

    def test_polish_stage_low_exploration(self, template: ComposerPipeline):
        """Polish stage should have low exploration override."""
        polish = template.get_stage("polish")
        assert polish is not None
        assert polish.exploration_override is not None
        assert polish.exploration_override <= 0.2

    def test_stage_ids(self, template: ComposerPipeline):
        """Stage IDs should be 'screen', 'learn', 'polish'."""
        assert template.stage_ids == ["screen", "learn", "polish"]

    def test_no_loop(self, template: ComposerPipeline):
        """screening_then_optimize should not loop on stagnation."""
        assert template.loop_on_stagnation is False


# ---------------------------------------------------------------------------
# Tests: restart_on_stagnation Template
# ---------------------------------------------------------------------------


class TestRestartOnStagnation:
    @pytest.fixture
    def template(self) -> ComposerPipeline:
        return PIPELINE_TEMPLATES["restart_on_stagnation"]

    def test_has_three_stages(self, template: ComposerPipeline):
        """restart_on_stagnation should have exactly 3 stages."""
        assert template.n_stages == 3

    def test_loop_on_stagnation_enabled(self, template: ComposerPipeline):
        """loop_on_stagnation should be True."""
        assert template.loop_on_stagnation is True

    def test_restart_stage_id_set(self, template: ComposerPipeline):
        """restart_stage_id should be set and reference a valid stage."""
        assert template.restart_stage_id is not None
        assert template.restart_stage_id in template.stage_ids

    def test_restart_stage_id_is_initial_explore(self, template: ComposerPipeline):
        """restart_stage_id should be 'initial_explore'."""
        assert template.restart_stage_id == "initial_explore"

    def test_has_stagnation_detected_exit(self, template: ComposerPipeline):
        """At least one stage should use STAGNATION_DETECTED exit condition."""
        exit_types = [s.exit_condition_type for s in template.stages]
        assert StageExitCondition.STAGNATION_DETECTED in exit_types

    def test_optimize_stage_has_stagnation_exit(self, template: ComposerPipeline):
        """The 'optimize' stage should use STAGNATION_DETECTED exit."""
        opt_stage = template.get_stage("optimize")
        assert opt_stage is not None
        assert opt_stage.exit_condition_type == StageExitCondition.STAGNATION_DETECTED
        assert "kpi_plateau_length" in opt_stage.exit_conditions

    def test_restart_explore_uses_random(self, template: ComposerPipeline):
        """The restart_explore stage should use random backend."""
        restart = template.get_stage("restart_explore")
        assert restart is not None
        assert restart.backend_name == "random"

    def test_restart_explore_full_exploration(self, template: ComposerPipeline):
        """The restart_explore stage should have exploration_override=1.0."""
        restart = template.get_stage("restart_explore")
        assert restart is not None
        assert restart.exploration_override == 1.0

    def test_correct_backends(self, template: ComposerPipeline):
        """Stages should use latin_hypercube, tpe, random."""
        backends = [s.backend_name for s in template.stages]
        assert backends == ["latin_hypercube", "tpe", "random"]

    def test_stage_ids(self, template: ComposerPipeline):
        """Stage IDs should be 'initial_explore', 'optimize', 'restart_explore'."""
        assert template.stage_ids == ["initial_explore", "optimize", "restart_explore"]


# ---------------------------------------------------------------------------
# Tests: exploit_heavy Template
# ---------------------------------------------------------------------------


class TestExploitHeavy:
    @pytest.fixture
    def template(self) -> ComposerPipeline:
        return PIPELINE_TEMPLATES["exploit_heavy"]

    def test_has_two_stages(self, template: ComposerPipeline):
        """exploit_heavy should have exactly 2 stages."""
        assert template.n_stages == 2

    def test_warmup_fraction_small(self, template: ComposerPipeline):
        """Warmup fraction should be small (< 0.2)."""
        warmup = template.stages[0]
        assert warmup.iteration_fraction <= 0.2

    def test_exploit_fraction_large(self, template: ComposerPipeline):
        """Exploit fraction should be large (> 0.7)."""
        exploit = template.stages[1]
        assert exploit.iteration_fraction >= 0.7

    def test_fractions_sum_to_one(self, template: ComposerPipeline):
        """Iteration fractions should sum to approximately 1.0."""
        total = sum(s.iteration_fraction for s in template.stages)
        assert total == pytest.approx(1.0, abs=0.05)

    def test_warmup_uses_random(self, template: ComposerPipeline):
        """Warmup stage should use random backend."""
        assert template.stages[0].backend_name == "random"

    def test_exploit_uses_tpe(self, template: ComposerPipeline):
        """Exploit stage should use tpe backend."""
        assert template.stages[1].backend_name == "tpe"

    def test_warmup_moderate_exploration(self, template: ComposerPipeline):
        """Warmup stage should have moderate-to-high exploration."""
        warmup = template.stages[0]
        assert warmup.exploration_override is not None
        assert warmup.exploration_override >= 0.5

    def test_exploit_low_exploration(self, template: ComposerPipeline):
        """Exploit stage should have low exploration override."""
        exploit = template.stages[1]
        assert exploit.exploration_override is not None
        assert exploit.exploration_override <= 0.3

    def test_no_loop(self, template: ComposerPipeline):
        """exploit_heavy should not loop on stagnation."""
        assert template.loop_on_stagnation is False
        assert template.restart_stage_id is None

    def test_stage_ids(self, template: ComposerPipeline):
        """Stage IDs should be 'warmup' and 'exploit'."""
        assert template.stage_ids == ["warmup", "exploit"]


# ---------------------------------------------------------------------------
# Tests: Serialization Round-Trip for All Templates
# ---------------------------------------------------------------------------


class TestSerializationRoundTrip:
    @pytest.mark.parametrize("template_name", EXPECTED_TEMPLATES)
    def test_template_survives_roundtrip(self, template_name: str):
        """Each template should survive to_dict -> from_dict without data loss."""
        original = PIPELINE_TEMPLATES[template_name]
        rebuilt = ComposerPipeline.from_dict(original.to_dict())

        assert rebuilt.name == original.name
        assert rebuilt.description == original.description
        assert rebuilt.n_stages == original.n_stages
        assert rebuilt.stage_ids == original.stage_ids
        assert rebuilt.loop_on_stagnation == original.loop_on_stagnation
        assert rebuilt.restart_stage_id == original.restart_stage_id
        assert rebuilt.metadata == original.metadata

        for orig_s, rebuilt_s in zip(original.stages, rebuilt.stages):
            assert rebuilt_s.stage_id == orig_s.stage_id
            assert rebuilt_s.backend_name == orig_s.backend_name
            assert rebuilt_s.iteration_fraction == orig_s.iteration_fraction
            assert rebuilt_s.min_iterations == orig_s.min_iterations
            assert rebuilt_s.max_iterations == orig_s.max_iterations
            assert rebuilt_s.exit_conditions == orig_s.exit_conditions
            assert rebuilt_s.exit_condition_type == orig_s.exit_condition_type
            assert rebuilt_s.phase_trigger == orig_s.phase_trigger
            assert rebuilt_s.exploration_override == orig_s.exploration_override
            assert rebuilt_s.reason == orig_s.reason

    @pytest.mark.parametrize("template_name", EXPECTED_TEMPLATES)
    def test_double_roundtrip_stable(self, template_name: str):
        """Double round-trip (to_dict -> from_dict -> to_dict) is stable."""
        original = PIPELINE_TEMPLATES[template_name]
        first_dict = original.to_dict()
        rebuilt = ComposerPipeline.from_dict(first_dict)
        second_dict = rebuilt.to_dict()

        assert first_dict == second_dict
