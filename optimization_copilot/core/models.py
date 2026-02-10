"""Core data models for the OptimizationAgent."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


# ── Enums ──────────────────────────────────────────────

class VariableType(str, Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    MIXED = "mixed"


class ObjectiveForm(str, Enum):
    SINGLE = "single"
    MULTI_OBJECTIVE = "multi_objective"
    CONSTRAINED = "constrained"


class NoiseRegime(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CostProfile(str, Enum):
    UNIFORM = "uniform"
    HETEROGENEOUS = "heterogeneous"


class FailureInformativeness(str, Enum):
    WEAK = "weak"
    STRONG = "strong"


class DataScale(str, Enum):
    TINY = "tiny"        # < 10 points
    SMALL = "small"      # 10-50
    MODERATE = "moderate" # 50+


class Dynamics(str, Enum):
    STATIC = "static"
    TIME_SERIES = "time_series"


class FeasibleRegion(str, Enum):
    WIDE = "wide"
    NARROW = "narrow"
    FRAGMENTED = "fragmented"


class RiskPosture(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class Phase(str, Enum):
    COLD_START = "cold_start"
    LEARNING = "learning"
    EXPLOITATION = "exploitation"
    STAGNATION = "stagnation"
    TERMINATION = "termination"


# ── Core Data Models ───────────────────────────────────

@dataclass
class ParameterSpec:
    """Specification for a single optimization parameter."""
    name: str
    type: VariableType
    lower: float | None = None
    upper: float | None = None
    categories: list[str] | None = None


@dataclass
class Observation:
    """A single experimental observation."""
    iteration: int
    parameters: dict[str, Any]
    kpi_values: dict[str, float]
    qc_passed: bool = True
    is_failure: bool = False
    failure_reason: str | None = None
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CampaignSnapshot:
    """Complete state of an optimization campaign at a point in time."""
    campaign_id: str
    parameter_specs: list[ParameterSpec]
    observations: list[Observation]
    objective_names: list[str]
    objective_directions: list[str]  # "minimize" or "maximize"
    constraints: list[dict[str, Any]] = field(default_factory=list)
    current_iteration: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_observations(self) -> int:
        return len(self.observations)

    @property
    def n_failures(self) -> int:
        return sum(1 for o in self.observations if o.is_failure)

    @property
    def failure_rate(self) -> float:
        if not self.observations:
            return 0.0
        return self.n_failures / self.n_observations

    @property
    def parameter_names(self) -> list[str]:
        return [p.name for p in self.parameter_specs]

    @property
    def successful_observations(self) -> list[Observation]:
        return [o for o in self.observations if not o.is_failure]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CampaignSnapshot:
        data = data.copy()
        data["parameter_specs"] = [ParameterSpec(**p) for p in data["parameter_specs"]]
        data["observations"] = [Observation(**o) for o in data["observations"]]
        return cls(**data)


@dataclass
class StabilizeSpec:
    """Data conditioning policies."""
    noise_smoothing_window: int = 3
    outlier_rejection_sigma: float = 3.0
    failure_handling: str = "penalize"  # penalize, exclude, impute
    censored_data_policy: str = "ignore"  # ignore, model, impute
    constraint_tightening_rate: float = 0.0
    reweighting_strategy: str = "none"  # none, recency, quality
    retry_normalization: bool = False


@dataclass
class StrategyDecision:
    """Output of the OptimizationAgent: which strategy to use and how."""
    backend_name: str
    stabilize_spec: StabilizeSpec
    exploration_strength: float  # 0.0 (pure exploit) to 1.0 (pure explore)
    batch_size: int = 1
    batch_control_hints: dict[str, Any] = field(default_factory=dict)
    risk_posture: RiskPosture = RiskPosture.MODERATE
    phase: Phase = Phase.COLD_START
    reason_codes: list[str] = field(default_factory=list)
    fallback_events: list[str] = field(default_factory=list)
    decision_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["risk_posture"] = self.risk_posture.value
        d["phase"] = self.phase.value
        return d


@dataclass
class ProblemFingerprint:
    """Classification of the optimization problem context."""
    variable_types: VariableType = VariableType.CONTINUOUS
    objective_form: ObjectiveForm = ObjectiveForm.SINGLE
    noise_regime: NoiseRegime = NoiseRegime.LOW
    cost_profile: CostProfile = CostProfile.UNIFORM
    failure_informativeness: FailureInformativeness = FailureInformativeness.WEAK
    data_scale: DataScale = DataScale.TINY
    dynamics: Dynamics = Dynamics.STATIC
    feasible_region: FeasibleRegion = FeasibleRegion.WIDE

    def to_dict(self) -> dict:
        return {k: v.value for k, v in asdict(self).items()}

    def to_tuple(self) -> tuple:
        return (
            self.variable_types.value,
            self.objective_form.value,
            self.noise_regime.value,
            self.cost_profile.value,
            self.failure_informativeness.value,
            self.data_scale.value,
            self.dynamics.value,
            self.feasible_region.value,
        )
