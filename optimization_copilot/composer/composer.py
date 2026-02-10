"""Algorithm Composer: multi-stage pipeline selection and orchestration."""

from __future__ import annotations

import copy
from typing import Any

from optimization_copilot.core.models import (
    CampaignSnapshot,
    DataScale,
    FeasibleRegion,
    NoiseRegime,
    ProblemFingerprint,
)
from optimization_copilot.composer.models import (
    ComposerPipeline,
    PipelineOutcome,
    PipelineRecord,
    PipelineStage,
    StageExitCondition,
    StageTransition,
)
from optimization_copilot.composer.templates import PIPELINE_TEMPLATES


# ── Helpers ────────────────────────────────────────────


def _fingerprint_key(fp: ProblemFingerprint) -> str:
    """Serialize a ProblemFingerprint to a stable string key."""
    return str(fp.to_tuple())


def _incremental_mean(old_mean: float, old_count: int, new_value: float) -> float:
    """Compute the new mean after adding one observation."""
    return (old_mean * old_count + new_value) / (old_count + 1)


# ── AlgorithmComposer ─────────────────────────────────


class AlgorithmComposer:
    """Selects, orchestrates, and learns from multi-stage optimization pipelines.

    The composer picks a pipeline template based on the problem fingerprint
    and budget, manages stage transitions during a run, and records outcomes
    to improve future selections.
    """

    DEFAULT_BACKENDS: list[str] = ["random", "latin_hypercube", "tpe"]

    def __init__(
        self,
        available_backends: list[str] | None = None,
        templates: dict[str, ComposerPipeline] | None = None,
    ) -> None:
        self._available_backends: list[str] = list(
            available_backends if available_backends is not None else self.DEFAULT_BACKENDS
        )
        self._templates: dict[str, ComposerPipeline] = dict(
            templates if templates is not None else PIPELINE_TEMPLATES
        )

        # Learning state: keyed by (pipeline_name, fingerprint_key)
        self._records: dict[tuple[str, str], PipelineRecord] = {}

        # Active run state
        self._active_stage_index: int = 0
        self._stage_start_iteration: int = 0
        self._transitions: list[StageTransition] = []

    # ── Pipeline Composition ──────────────────────────

    def compose(
        self,
        fingerprint: ProblemFingerprint,
        budget: int,
        seed: int = 42,
    ) -> ComposerPipeline:
        """Select and configure a pipeline for the given problem context.

        Parameters
        ----------
        fingerprint:
            Classification of the optimization problem.
        budget:
            Total iteration budget for the campaign.
        seed:
            Random seed (reserved for future stochastic selection).

        Returns
        -------
        A :class:`ComposerPipeline` configured for the problem.
        """
        fp_key = _fingerprint_key(fingerprint)

        # 1. Check learned preferences (need >= 3 uses to trust).
        learned = self._best_learned_pipeline(fp_key)
        if learned is not None:
            pipeline = self._get_template(learned)
            if pipeline is not None:
                return self._filter_to_available(pipeline)

        # 2. Heuristic template selection.
        template_name = self._heuristic_select(fingerprint, budget)
        pipeline = self._get_template(template_name)
        if pipeline is None:
            # Absolute fallback: exploration_first or first available template.
            pipeline = self._get_template("exploration_first") or next(
                iter(self._templates.values())
            )

        return self._filter_to_available(pipeline)

    def _best_learned_pipeline(self, fp_key: str) -> str | None:
        """Return the pipeline name with best learned score, or None."""
        candidates: list[tuple[str, float]] = []
        for (pname, fkey), rec in self._records.items():
            if fkey != fp_key:
                continue
            if rec.n_uses < 3:
                continue
            confidence = min(1.0, rec.n_uses / 5.0)
            win_rate = rec.win_count / rec.n_uses if rec.n_uses > 0 else 0.0
            score = win_rate * confidence - rec.avg_failure_rate
            candidates.append((pname, score))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _heuristic_select(
        self, fingerprint: ProblemFingerprint, budget: int
    ) -> str:
        """Rule-based template selection from fingerprint + budget."""
        # HIGH noise or FRAGMENTED region -> restart_on_stagnation
        if (
            fingerprint.noise_regime == NoiseRegime.HIGH
            or fingerprint.feasible_region == FeasibleRegion.FRAGMENTED
        ):
            return "restart_on_stagnation"

        # TINY data + LOW noise -> exploit_heavy
        if (
            fingerprint.data_scale == DataScale.TINY
            and fingerprint.noise_regime == NoiseRegime.LOW
        ):
            return "exploit_heavy"

        # Larger budget -> screening_then_optimize
        if budget > 30:
            return "screening_then_optimize"

        # Default
        return "exploration_first"

    def _get_template(self, name: str) -> ComposerPipeline | None:
        """Return a deep copy of a named template, or None."""
        tpl = self._templates.get(name)
        if tpl is None:
            return None
        return ComposerPipeline.from_dict(tpl.to_dict())

    def _filter_to_available(self, pipeline: ComposerPipeline) -> ComposerPipeline:
        """Substitute unavailable backends with the first available one."""
        if not self._available_backends:
            return pipeline

        fallback = self._available_backends[0]
        for stage in pipeline.stages:
            if stage.backend_name not in self._available_backends:
                stage.backend_name = fallback
        return pipeline

    # ── Stage Selection ───────────────────────────────

    def select_stage(
        self,
        pipeline: ComposerPipeline,
        iteration: int,
        budget: int,
        diagnostics: dict[str, float],
        snapshot: CampaignSnapshot | None,
    ) -> PipelineStage:
        """Determine the active pipeline stage for the current iteration.

        Checks whether the current stage's exit condition has been met.
        If so, transitions to the next stage (or loops back on stagnation).

        Parameters
        ----------
        pipeline:
            The active pipeline.
        iteration:
            Current global iteration (0-based).
        budget:
            Total iteration budget.
        diagnostics:
            Current diagnostic signals (keys are signal names).
        snapshot:
            Current campaign snapshot (may be None for cold start).

        Returns
        -------
        The :class:`PipelineStage` that should be active at this iteration.
        """
        if not pipeline.stages:
            raise ValueError("Pipeline has no stages.")

        # Clamp index to valid range.
        if self._active_stage_index >= pipeline.n_stages:
            self._active_stage_index = pipeline.n_stages - 1

        current_stage = pipeline.stages[self._active_stage_index]
        iterations_in_stage = iteration - self._stage_start_iteration

        # Respect min_iterations.
        if iterations_in_stage < current_stage.min_iterations:
            return current_stage

        # Check exit conditions.
        should_exit = self._check_exit(
            current_stage, iterations_in_stage, budget, diagnostics
        )

        # Respect max_iterations (0 = no limit).
        if (
            not should_exit
            and current_stage.max_iterations > 0
            and iterations_in_stage >= current_stage.max_iterations
        ):
            should_exit = True

        if not should_exit:
            return current_stage

        # Determine next stage.
        is_last = self._active_stage_index >= pipeline.n_stages - 1

        if is_last and pipeline.loop_on_stagnation and pipeline.restart_stage_id:
            # Loop back to restart stage.
            restart_idx = self._find_stage_index(pipeline, pipeline.restart_stage_id)
            next_stage = pipeline.stages[restart_idx]
            self._record_transition(
                current_stage.stage_id,
                next_stage.stage_id,
                iteration,
                f"loop_restart:{current_stage.exit_condition_type.value}",
                diagnostics,
            )
            self._active_stage_index = restart_idx
            self._stage_start_iteration = iteration
            return next_stage

        if is_last:
            # Stay on the last stage.
            return current_stage

        # Advance to next stage.
        next_idx = self._active_stage_index + 1
        next_stage = pipeline.stages[next_idx]
        self._record_transition(
            current_stage.stage_id,
            next_stage.stage_id,
            iteration,
            current_stage.exit_condition_type.value,
            diagnostics,
        )
        self._active_stage_index = next_idx
        self._stage_start_iteration = iteration
        return next_stage

    def _check_exit(
        self,
        stage: PipelineStage,
        iterations_in_stage: int,
        budget: int,
        diagnostics: dict[str, float],
    ) -> bool:
        """Evaluate whether a stage's exit condition is satisfied."""
        cond = stage.exit_condition_type

        if cond == StageExitCondition.ITERATION_FRACTION:
            target = budget * stage.iteration_fraction
            return iterations_in_stage >= target

        if cond == StageExitCondition.DIAGNOSTIC_THRESHOLD:
            # Any diagnostic signal that meets or exceeds its threshold.
            for signal_name, threshold in stage.exit_conditions.items():
                if signal_name in diagnostics and diagnostics[signal_name] >= threshold:
                    return True
            # Also respect iteration fraction as a hard upper bound.
            if stage.iteration_fraction > 0:
                target = budget * stage.iteration_fraction
                if iterations_in_stage >= target:
                    return True
            return False

        if cond == StageExitCondition.STAGNATION_DETECTED:
            # Same logic as DIAGNOSTIC_THRESHOLD for stagnation signals.
            for signal_name, threshold in stage.exit_conditions.items():
                if signal_name in diagnostics and diagnostics[signal_name] >= threshold:
                    return True
            if stage.iteration_fraction > 0:
                target = budget * stage.iteration_fraction
                if iterations_in_stage >= target:
                    return True
            return False

        if cond == StageExitCondition.IMPROVEMENT_BELOW:
            # Exit when any signal drops to or below its threshold.
            for signal_name, threshold in stage.exit_conditions.items():
                if signal_name in diagnostics and diagnostics[signal_name] <= threshold:
                    return True
            if stage.iteration_fraction > 0:
                target = budget * stage.iteration_fraction
                if iterations_in_stage >= target:
                    return True
            return False

        if cond == StageExitCondition.MANUAL:
            # Never auto-exit; only iteration_fraction as hard bound.
            if stage.iteration_fraction > 0:
                target = budget * stage.iteration_fraction
                if iterations_in_stage >= target:
                    return True
            return False

        return False

    def _find_stage_index(self, pipeline: ComposerPipeline, stage_id: str) -> int:
        """Return the index of a stage by its identifier."""
        for i, stage in enumerate(pipeline.stages):
            if stage.stage_id == stage_id:
                return i
        return 0  # Fallback to first stage.

    def _record_transition(
        self,
        from_id: str,
        to_id: str,
        iteration: int,
        trigger: str,
        diagnostics: dict[str, float],
    ) -> None:
        """Append a stage transition record."""
        self._transitions.append(
            StageTransition(
                from_stage_id=from_id,
                to_stage_id=to_id,
                iteration=iteration,
                trigger=trigger,
                diagnostics_at_transition=dict(diagnostics),
            )
        )

    # ── Outcome Recording ─────────────────────────────

    def record_outcome(
        self,
        pipeline: ComposerPipeline,
        fingerprint: ProblemFingerprint,
        outcome: PipelineOutcome,
    ) -> None:
        """Update learning records with a pipeline run outcome.

        Uses incremental mean updates (same pattern as AlgorithmPortfolio).

        Parameters
        ----------
        pipeline:
            The pipeline that was executed.
        fingerprint:
            The problem fingerprint for this campaign.
        outcome:
            The outcome of the pipeline run.
        """
        fp_key = _fingerprint_key(fingerprint)
        rec_key = (pipeline.name, fp_key)

        if rec_key not in self._records:
            self._records[rec_key] = PipelineRecord(
                pipeline_name=pipeline.name,
                fingerprint_key=fp_key,
            )

        rec = self._records[rec_key]
        n = rec.n_uses

        rec.avg_best_kpi = _incremental_mean(rec.avg_best_kpi, n, outcome.best_kpi)
        rec.avg_convergence_speed = _incremental_mean(
            rec.avg_convergence_speed, n, outcome.convergence_speed
        )
        rec.avg_failure_rate = _incremental_mean(
            rec.avg_failure_rate, n, outcome.failure_rate
        )

        rec.n_uses = n + 1
        if outcome.is_winner:
            rec.win_count += 1

    # ── Pipeline Ranking ──────────────────────────────

    def rank_pipelines(
        self, fingerprint: ProblemFingerprint
    ) -> list[tuple[str, float]]:
        """Rank known pipelines for a fingerprint by learned performance.

        Score = win_rate * confidence - avg_failure_rate
        confidence = min(1.0, n_uses / 5.0)

        Parameters
        ----------
        fingerprint:
            The problem fingerprint to rank pipelines for.

        Returns
        -------
        List of (pipeline_name, score) sorted descending by score.
        """
        fp_key = _fingerprint_key(fingerprint)
        scored: list[tuple[str, float]] = []

        for (pname, fkey), rec in self._records.items():
            if fkey != fp_key:
                continue
            confidence = min(1.0, rec.n_uses / 5.0)
            win_rate = rec.win_count / rec.n_uses if rec.n_uses > 0 else 0.0
            score = win_rate * confidence - rec.avg_failure_rate
            scored.append((pname, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ── State Management ──────────────────────────────

    def reset(self) -> None:
        """Clear active run state (stage index, transitions)."""
        self._active_stage_index = 0
        self._stage_start_iteration = 0
        self._transitions = []

    @property
    def transitions(self) -> list[StageTransition]:
        """Return recorded stage transitions for the current run."""
        return list(self._transitions)

    @property
    def active_stage_index(self) -> int:
        """Return the index of the currently active stage."""
        return self._active_stage_index

    # ── Serialization ─────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the composer state to a plain dict."""
        return {
            "available_backends": list(self._available_backends),
            "records": [rec.to_dict() for rec in self._records.values()],
            "active_stage_index": self._active_stage_index,
            "stage_start_iteration": self._stage_start_iteration,
            "transitions": [t.to_dict() for t in self._transitions],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        available_backends: list[str] | None = None,
        templates: dict[str, ComposerPipeline] | None = None,
    ) -> AlgorithmComposer:
        """Deserialize a composer from a plain dict.

        Parameters
        ----------
        data:
            Dictionary produced by :meth:`to_dict`.
        available_backends:
            Override backend list (defaults to value in data).
        templates:
            Override template dict (defaults to PIPELINE_TEMPLATES).
        """
        backends = available_backends or data.get("available_backends", cls.DEFAULT_BACKENDS)
        composer = cls(available_backends=backends, templates=templates)

        for rec_data in data.get("records", []):
            rec = PipelineRecord.from_dict(rec_data)
            key = (rec.pipeline_name, rec.fingerprint_key)
            composer._records[key] = rec

        composer._active_stage_index = data.get("active_stage_index", 0)
        composer._stage_start_iteration = data.get("stage_start_iteration", 0)
        composer._transitions = [
            StageTransition.from_dict(t) for t in data.get("transitions", [])
        ]

        return composer
