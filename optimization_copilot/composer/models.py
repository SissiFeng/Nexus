"""Data models for algorithm composition and pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Enums ──────────────────────────────────────────────


class StageExitCondition(str, Enum):
    """How a pipeline stage decides to hand off to the next stage."""

    ITERATION_FRACTION = "iteration_fraction"
    DIAGNOSTIC_THRESHOLD = "diagnostic_threshold"
    STAGNATION_DETECTED = "stagnation_detected"
    IMPROVEMENT_BELOW = "improvement_below"
    MANUAL = "manual"


# ── Pipeline Stage ─────────────────────────────────────


@dataclass
class PipelineStage:
    """A single stage in a multi-algorithm pipeline."""

    stage_id: str
    backend_name: str
    iteration_fraction: float = 0.0
    min_iterations: int = 1
    max_iterations: int = 0  # 0 = no limit
    exit_conditions: dict[str, float] = field(default_factory=dict)
    exit_condition_type: StageExitCondition = StageExitCondition.ITERATION_FRACTION
    phase_trigger: str | None = None
    exploration_override: float | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "backend_name": self.backend_name,
            "iteration_fraction": self.iteration_fraction,
            "min_iterations": self.min_iterations,
            "max_iterations": self.max_iterations,
            "exit_conditions": dict(self.exit_conditions),
            "exit_condition_type": self.exit_condition_type.value,
            "phase_trigger": self.phase_trigger,
            "exploration_override": self.exploration_override,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineStage:
        data = data.copy()
        if "exit_condition_type" in data:
            data["exit_condition_type"] = StageExitCondition(data["exit_condition_type"])
        return cls(**data)


# ── Stage Transition ───────────────────────────────────


@dataclass
class StageTransition:
    """Record of a transition between two pipeline stages."""

    from_stage_id: str
    to_stage_id: str
    iteration: int
    trigger: str
    diagnostics_at_transition: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_stage_id": self.from_stage_id,
            "to_stage_id": self.to_stage_id,
            "iteration": self.iteration,
            "trigger": self.trigger,
            "diagnostics_at_transition": dict(self.diagnostics_at_transition),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StageTransition:
        return cls(**data)


# ── Composer Pipeline ──────────────────────────────────


@dataclass
class ComposerPipeline:
    """A complete multi-stage optimization pipeline."""

    name: str
    description: str
    stages: list[PipelineStage] = field(default_factory=list)
    loop_on_stagnation: bool = False
    restart_stage_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def stage_ids(self) -> list[str]:
        """Return ordered list of stage identifiers."""
        return [s.stage_id for s in self.stages]

    @property
    def n_stages(self) -> int:
        """Return number of stages in the pipeline."""
        return len(self.stages)

    def get_stage(self, stage_id: str) -> PipelineStage | None:
        """Look up a stage by its identifier."""
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "stages": [s.to_dict() for s in self.stages],
            "loop_on_stagnation": self.loop_on_stagnation,
            "restart_stage_id": self.restart_stage_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComposerPipeline:
        data = data.copy()
        data["stages"] = [PipelineStage.from_dict(s) for s in data.get("stages", [])]
        return cls(**data)


# ── Pipeline Outcome ───────────────────────────────────


@dataclass
class PipelineOutcome:
    """Result of running a pipeline on a specific problem."""

    pipeline_name: str
    fingerprint_key: str
    n_iterations: int
    best_kpi: float
    convergence_speed: float
    failure_rate: float
    transitions: list[StageTransition] = field(default_factory=list)
    is_winner: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "fingerprint_key": self.fingerprint_key,
            "n_iterations": self.n_iterations,
            "best_kpi": self.best_kpi,
            "convergence_speed": self.convergence_speed,
            "failure_rate": self.failure_rate,
            "transitions": [t.to_dict() for t in self.transitions],
            "is_winner": self.is_winner,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineOutcome:
        data = data.copy()
        data["transitions"] = [
            StageTransition.from_dict(t) for t in data.get("transitions", [])
        ]
        return cls(**data)


# ── Pipeline Record ────────────────────────────────────


@dataclass
class PipelineRecord:
    """Aggregated performance record for a pipeline on a fingerprint."""

    pipeline_name: str
    fingerprint_key: str
    n_uses: int = 0
    win_count: int = 0
    avg_best_kpi: float = 0.0
    avg_convergence_speed: float = 0.0
    avg_failure_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "fingerprint_key": self.fingerprint_key,
            "n_uses": self.n_uses,
            "win_count": self.win_count,
            "avg_best_kpi": self.avg_best_kpi,
            "avg_convergence_speed": self.avg_convergence_speed,
            "avg_failure_rate": self.avg_failure_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineRecord:
        return cls(**data)
