"""Curriculum Engine: builds and manages progressive difficulty stages for optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    VariableType,
)
from optimization_copilot.diagnostics.engine import DiagnosticsVector


# ---------------------------------------------------------------------------
# Module-local helpers
# ---------------------------------------------------------------------------


def _rank_parameters(
    specs: list[ParameterSpec],
    importance_scores: dict[str, float] | None,
) -> list[str]:
    """Return parameter names ordered by importance (descending).

    If *importance_scores* is ``None`` or empty, the original order from
    *specs* is preserved.  Scored parameters are sorted by score descending
    (ties broken alphabetically by name); unscored parameters are appended
    in their original order.
    """
    if not importance_scores:
        return [s.name for s in specs]

    scored: list[tuple[str, float]] = []
    unscored: list[str] = []
    for s in specs:
        if s.name in importance_scores:
            scored.append((s.name, importance_scores[s.name]))
        else:
            unscored.append(s.name)

    # Sort by score descending, then by name ascending for tie-breaking.
    scored.sort(key=lambda pair: (-pair[1], pair[0]))

    return [name for name, _ in scored] + unscored


def _widen_bounds(
    spec: ParameterSpec,
    factor: float,
) -> tuple[float, float]:
    """Widen the bounds of *spec* by *factor*, clamped to 2x original range.

    If ``spec.lower`` or ``spec.upper`` is ``None``, returns ``(0.0, 1.0)``.
    """
    if spec.lower is None or spec.upper is None:
        return (0.0, 1.0)

    midpoint = (spec.lower + spec.upper) / 2.0
    half_range = (spec.upper - spec.lower) / 2.0
    new_half = half_range * factor

    # Clamp: don't expand beyond 2x original range in either direction.
    original_range = spec.upper - spec.lower
    max_half = original_range  # 2x original range / 2
    new_half = min(new_half, max_half)

    new_lower = midpoint - new_half
    new_upper = midpoint + new_half
    return (new_lower, new_upper)


def _interpolate_bounds(
    spec: ParameterSpec,
    widened: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    """Linearly interpolate between *widened* bounds (t=0) and original (t=1).

    If ``spec.lower`` or ``spec.upper`` is ``None``, returns *widened*.
    """
    if spec.lower is None or spec.upper is None:
        return widened

    lower = widened[0] + t * (spec.lower - widened[0])
    upper = widened[1] + t * (spec.upper - widened[1])
    return (lower, upper)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CurriculumStage:
    """A single stage in a curriculum plan."""

    stage_id: int
    active_parameters: list[str]
    modified_bounds: dict[str, tuple[float, float]]
    constraints_enabled: list[str]
    difficulty_level: float  # 0.0 to 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_active_parameters(self) -> int:
        return len(self.active_parameters)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "stage_id": self.stage_id,
            "active_parameters": list(self.active_parameters),
            "modified_bounds": {
                k: list(v) for k, v in self.modified_bounds.items()
            },
            "constraints_enabled": list(self.constraints_enabled),
            "difficulty_level": self.difficulty_level,
            "metadata": dict(self.metadata),
        }
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CurriculumStage:
        data = data.copy()
        data["modified_bounds"] = {
            k: tuple(v) for k, v in data["modified_bounds"].items()
        }
        return cls(**data)


@dataclass
class CurriculumPolicy:
    """Configurable thresholds governing curriculum progression."""

    min_observations_per_stage: int = 10
    promotion_plateau_threshold: int = 3
    promotion_convergence_threshold: float = 0.2
    promotion_velocity_threshold: float = 0.0
    demotion_plateau_threshold: int = 8
    demotion_failure_rate_threshold: float = 0.4
    initial_parameter_fraction: float = 0.3
    parameter_increment_fraction: float = 0.2
    bounds_widening_factor: float = 1.5
    bounds_tightening_per_stage: float = 0.8  # reserved for forward compatibility
    constraint_introduction_stage: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_observations_per_stage": self.min_observations_per_stage,
            "promotion_plateau_threshold": self.promotion_plateau_threshold,
            "promotion_convergence_threshold": self.promotion_convergence_threshold,
            "promotion_velocity_threshold": self.promotion_velocity_threshold,
            "demotion_plateau_threshold": self.demotion_plateau_threshold,
            "demotion_failure_rate_threshold": self.demotion_failure_rate_threshold,
            "initial_parameter_fraction": self.initial_parameter_fraction,
            "parameter_increment_fraction": self.parameter_increment_fraction,
            "bounds_widening_factor": self.bounds_widening_factor,
            "bounds_tightening_per_stage": self.bounds_tightening_per_stage,
            "constraint_introduction_stage": self.constraint_introduction_stage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CurriculumPolicy:
        return cls(**data)


@dataclass
class CurriculumPlan:
    """Complete curriculum plan with stage history and current position."""

    stages: list[CurriculumStage]
    current_stage_index: int
    total_parameters: int
    total_constraints: int
    promotion_history: list[dict[str, Any]] = field(default_factory=list)
    demotion_history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.current_stage_index]

    @property
    def is_final_stage(self) -> bool:
        return self.current_stage_index >= len(self.stages) - 1

    @property
    def is_complete(self) -> bool:
        """True when current stage has all parameters AND all constraints."""
        stage = self.current_stage
        return (
            len(stage.active_parameters) >= self.total_parameters
            and len(stage.constraints_enabled) >= self.total_constraints
        )

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [s.to_dict() for s in self.stages],
            "current_stage_index": self.current_stage_index,
            "total_parameters": self.total_parameters,
            "total_constraints": self.total_constraints,
            "promotion_history": list(self.promotion_history),
            "demotion_history": list(self.demotion_history),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CurriculumPlan:
        data = data.copy()
        data["stages"] = [CurriculumStage.from_dict(s) for s in data["stages"]]
        return cls(**data)


# ---------------------------------------------------------------------------
# CurriculumEngine
# ---------------------------------------------------------------------------


class CurriculumEngine:
    """Builds and evaluates progressive curriculum plans for optimization campaigns.

    The engine creates a multi-stage plan that starts with a simplified version
    of the optimization problem (fewer parameters, wider bounds, no constraints)
    and progressively increases difficulty until the full problem is exposed.
    """

    def __init__(self, policy: CurriculumPolicy | None = None) -> None:
        self.policy = policy or CurriculumPolicy()

    # -- plan creation ------------------------------------------------------

    def create_plan(
        self,
        snapshot: CampaignSnapshot,
        importance_scores: dict[str, float] | None = None,
        seed: int = 42,  # noqa: ARG002 – reserved for future stochastic policies
    ) -> CurriculumPlan:
        """Create a curriculum plan from the current campaign state.

        Parameters
        ----------
        snapshot:
            Current campaign snapshot with parameter specs and constraints.
        importance_scores:
            Optional mapping of parameter name -> importance score used to
            decide which parameters to introduce first.
        seed:
            Random seed (reserved for future stochastic stage generation).

        Returns
        -------
        CurriculumPlan
        """
        specs = snapshot.parameter_specs
        n_params = len(specs)

        ranked_params = _rank_parameters(specs, importance_scores)
        spec_map: dict[str, ParameterSpec] = {s.name: s for s in specs}

        constraint_names: list[str] = [
            c.get("name", f"constraint_{i}")
            for i, c in enumerate(snapshot.constraints)
        ]
        n_constraints = len(constraint_names)

        # ---- trivial / single-param case -----------------------------------
        if n_params <= 1:
            stage = CurriculumStage(
                stage_id=0,
                active_parameters=list(ranked_params),
                modified_bounds={
                    name: (spec_map[name].lower or 0.0, spec_map[name].upper or 1.0)
                    for name in ranked_params
                    if spec_map[name].type != VariableType.CATEGORICAL
                },
                constraints_enabled=list(constraint_names),
                difficulty_level=1.0,
            )
            return CurriculumPlan(
                stages=[stage],
                current_stage_index=0,
                total_parameters=n_params,
                total_constraints=n_constraints,
            )

        # ---- multi-param: compute stage parameter counts -------------------
        policy = self.policy
        initial_count = max(1, math.ceil(n_params * policy.initial_parameter_fraction))
        increment = max(1, math.ceil(n_params * policy.parameter_increment_fraction))

        stage_param_counts: list[int] = []
        count = initial_count
        while count < n_params:
            stage_param_counts.append(count)
            count += increment
            if count > n_params:
                count = n_params
        # Always include the final stage with all parameters.
        stage_param_counts.append(n_params)

        n_stages = len(stage_param_counts)

        # ---- compute widened bounds (skip CATEGORICAL) ---------------------
        widened_bounds: dict[str, tuple[float, float]] = {}
        for name in ranked_params:
            spec = spec_map[name]
            if spec.type == VariableType.CATEGORICAL:
                continue
            widened_bounds[name] = _widen_bounds(spec, policy.bounds_widening_factor)

        # ---- constraint introduction scheduling ----------------------------
        intro_stage = policy.constraint_introduction_stage
        stages_with_constraints = max(1, n_stages - intro_stage)

        # ---- build stages --------------------------------------------------
        stages: list[CurriculumStage] = []
        for stage_idx, param_count in enumerate(stage_param_counts):
            active_params = ranked_params[:param_count]

            # Bounds interpolation: t goes from 0 (fully widened) to 1 (original).
            t = stage_idx / max(1, n_stages - 1)

            modified_bounds: dict[str, tuple[float, float]] = {}
            for name in active_params:
                spec = spec_map[name]
                if spec.type == VariableType.CATEGORICAL:
                    continue
                wb = widened_bounds[name]
                modified_bounds[name] = _interpolate_bounds(spec, wb, t)

            # Constraints: none before intro stage; progressively introduced after.
            if stage_idx < intro_stage or n_constraints == 0:
                enabled_constraints: list[str] = []
            else:
                constraint_progress = (stage_idx - intro_stage + 1) / max(
                    1, stages_with_constraints
                )
                n_enabled = max(1, math.ceil(constraint_progress * n_constraints))
                enabled_constraints = constraint_names[:n_enabled]

            difficulty = round(stage_idx / max(1, n_stages - 1), 4)

            stages.append(
                CurriculumStage(
                    stage_id=stage_idx,
                    active_parameters=list(active_params),
                    modified_bounds=modified_bounds,
                    constraints_enabled=enabled_constraints,
                    difficulty_level=difficulty,
                )
            )

        return CurriculumPlan(
            stages=stages,
            current_stage_index=0,
            total_parameters=n_params,
            total_constraints=n_constraints,
        )

    # -- plan evaluation (promotion / demotion) -----------------------------

    def evaluate(
        self,
        plan: CurriculumPlan,
        snapshot: CampaignSnapshot,
        diagnostics: DiagnosticsVector,
    ) -> CurriculumPlan:
        """Evaluate the current plan against diagnostics and return an updated plan.

        This method does **not** mutate the input *plan*; it returns a new
        ``CurriculumPlan`` with potentially updated ``current_stage_index`` and
        history entries.

        Returns
        -------
        CurriculumPlan
            A (possibly unchanged) copy of the plan.
        """
        policy = self.policy
        n_obs = snapshot.n_observations
        current_idx = plan.current_stage_index
        n_stages = len(plan.stages)

        # Copy mutable history lists so we don't mutate the original plan.
        promotion_history = list(plan.promotion_history)
        demotion_history = list(plan.demotion_history)
        new_idx = current_idx

        # Not enough data yet -- return unchanged copy.
        if n_obs < policy.min_observations_per_stage:
            return CurriculumPlan(
                stages=plan.stages,
                current_stage_index=new_idx,
                total_parameters=plan.total_parameters,
                total_constraints=plan.total_constraints,
                promotion_history=promotion_history,
                demotion_history=demotion_history,
                metadata=dict(plan.metadata),
            )

        plateau = diagnostics.kpi_plateau_length
        failure_rate = diagnostics.failure_rate
        convergence = diagnostics.convergence_trend
        velocity = diagnostics.improvement_velocity

        demoted = False

        # ---- demotion check (safety first) --------------------------------
        if current_idx > 0:
            if (
                plateau > policy.demotion_plateau_threshold
                or failure_rate > policy.demotion_failure_rate_threshold
            ):
                new_idx = current_idx - 1
                demoted = True
                demotion_history.append(
                    {
                        "from_stage": current_idx,
                        "to_stage": new_idx,
                        "reason": (
                            f"Demotion: plateau={plateau} "
                            f"(threshold={policy.demotion_plateau_threshold}), "
                            f"failure_rate={failure_rate:.3f} "
                            f"(threshold={policy.demotion_failure_rate_threshold})"
                        ),
                        "n_observations": n_obs,
                        "kpi_plateau_length": plateau,
                        "failure_rate": failure_rate,
                    }
                )

        # ---- promotion check ----------------------------------------------
        if not demoted and current_idx < n_stages - 1:
            if (
                plateau < policy.promotion_plateau_threshold
                and convergence > policy.promotion_convergence_threshold
                and velocity > policy.promotion_velocity_threshold
            ):
                new_idx = current_idx + 1
                promotion_history.append(
                    {
                        "from_stage": current_idx,
                        "to_stage": new_idx,
                        "reason": (
                            f"Promotion: plateau={plateau} "
                            f"(threshold={policy.promotion_plateau_threshold}), "
                            f"convergence={convergence:.3f} "
                            f"(threshold={policy.promotion_convergence_threshold}), "
                            f"velocity={velocity:.3f} "
                            f"(threshold={policy.promotion_velocity_threshold})"
                        ),
                        "n_observations": n_obs,
                        "kpi_plateau_length": plateau,
                        "convergence_trend": convergence,
                        "improvement_velocity": velocity,
                    }
                )

        return CurriculumPlan(
            stages=plan.stages,
            current_stage_index=new_idx,
            total_parameters=plan.total_parameters,
            total_constraints=plan.total_constraints,
            promotion_history=promotion_history,
            demotion_history=demotion_history,
            metadata=dict(plan.metadata),
        )

    # -- active snapshot generation -----------------------------------------

    @staticmethod
    def get_active_snapshot(
        plan: CurriculumPlan,
        snapshot: CampaignSnapshot,
    ) -> CampaignSnapshot:
        """Return a filtered snapshot matching the current curriculum stage.

        * Only active parameters are included (with modified bounds applied).
        * Only enabled constraints are included.
        * All observations are kept unchanged.
        * Curriculum metadata is added.
        """
        stage = plan.current_stage
        active_set = set(stage.active_parameters)

        # Filter and modify parameter specs.
        filtered_specs: list[ParameterSpec] = []
        for spec in snapshot.parameter_specs:
            if spec.name not in active_set:
                continue
            if spec.name in stage.modified_bounds:
                new_lower, new_upper = stage.modified_bounds[spec.name]
                filtered_specs.append(
                    ParameterSpec(
                        name=spec.name,
                        type=spec.type,
                        lower=new_lower,
                        upper=new_upper,
                        categories=spec.categories,
                    )
                )
            else:
                filtered_specs.append(spec)

        # Filter constraints to enabled set.
        enabled_set = set(stage.constraints_enabled)
        filtered_constraints: list[dict[str, Any]] = []
        for i, c in enumerate(snapshot.constraints):
            c_name = c.get("name", f"constraint_{i}")
            if c_name in enabled_set:
                filtered_constraints.append(c)

        # Build metadata.
        new_metadata = dict(snapshot.metadata)
        new_metadata["curriculum_stage_id"] = stage.stage_id
        new_metadata["curriculum_difficulty"] = stage.difficulty_level

        return CampaignSnapshot(
            campaign_id=snapshot.campaign_id,
            parameter_specs=filtered_specs,
            observations=list(snapshot.observations),
            objective_names=list(snapshot.objective_names),
            objective_directions=list(snapshot.objective_directions),
            constraints=filtered_constraints,
            current_iteration=snapshot.current_iteration,
            metadata=new_metadata,
        )
