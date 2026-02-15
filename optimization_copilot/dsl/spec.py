"""Core dataclass definitions for the optimization DSL.

Provides a declarative specification language for defining optimization
campaigns, including parameters, objectives, budgets, and execution
preferences. All dataclasses support round-trip serialization via
to_dict / from_dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


# ── Enums ──────────────────────────────────────────────


class ParamType(str, Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"


class Direction(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class RiskPreference(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class DiversityStrategy(str, Enum):
    MAXIMIN = "maximin"
    COVERAGE = "coverage"
    HYBRID = "hybrid"


# ── Dataclasses ────────────────────────────────────────


@dataclass
class ConditionDef:
    """Conditional parameter dependency.

    Specifies that a parameter is only active when its parent parameter
    takes a specific value. For example, a 'learning_rate' parameter
    might only be active when 'optimizer' is set to 'adam'.
    """
    parent_name: str
    parent_value: Any

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ConditionDef:
        return cls(**data)


@dataclass
class ParameterDef:
    """Specification for a single optimization parameter.

    Supports continuous, discrete, and categorical parameter types
    with optional conditional activation, freezing, and metadata.
    """
    name: str
    type: ParamType
    lower: float | None = None
    upper: float | None = None
    categories: list[str] | None = None
    step_size: float | None = None
    condition: ConditionDef | None = None
    frozen: bool = False
    frozen_value: Any = None
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        # Represent condition as dict or None
        if self.condition is not None:
            d["condition"] = self.condition.to_dict()
        else:
            d["condition"] = None
        return d

    @classmethod
    def from_dict(cls, data: dict) -> ParameterDef:
        data = data.copy()
        data["type"] = ParamType(data["type"])
        if data.get("condition") is not None:
            data["condition"] = ConditionDef.from_dict(data["condition"])
        else:
            data["condition"] = None
        return cls(**data)


@dataclass
class ObjectiveDef:
    """Specification for an optimization objective.

    Objectives can be minimized or maximized, optionally subject to
    upper/lower constraints. Multi-objective campaigns use multiple
    ObjectiveDef instances with weights to express relative importance.
    """
    name: str
    direction: Direction
    constraint_lower: float | None = None
    constraint_upper: float | None = None
    is_primary: bool = True
    weight: float = 1.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["direction"] = self.direction.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> ObjectiveDef:
        data = data.copy()
        data["direction"] = Direction(data["direction"])
        return cls(**data)


@dataclass
class BudgetDef:
    """Resource budget constraints for the optimization campaign.

    All fields are optional; unset fields imply no constraint on that
    resource dimension.
    """
    max_samples: int | None = None
    max_time_seconds: float | None = None
    max_cost: float | None = None
    max_iterations: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> BudgetDef:
        return cls(**data)


@dataclass
class ParallelDef:
    """Parallel evaluation configuration.

    Controls how many candidates are evaluated simultaneously and
    how diversity among batch members is enforced.
    """
    batch_size: int = 1
    diversity_strategy: DiversityStrategy = DiversityStrategy.HYBRID

    def to_dict(self) -> dict:
        d = asdict(self)
        d["diversity_strategy"] = self.diversity_strategy.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> ParallelDef:
        data = data.copy()
        data["diversity_strategy"] = DiversityStrategy(data["diversity_strategy"])
        return cls(**data)


@dataclass
class OptimizationSpec:
    """Complete specification for an optimization campaign.

    The top-level container that brings together parameter definitions,
    objectives, budget constraints, risk preferences, and parallel
    evaluation settings into a single declarative specification.
    """
    campaign_id: str
    parameters: list[ParameterDef]
    objectives: list[ObjectiveDef]
    budget: BudgetDef
    risk_preference: RiskPreference = RiskPreference.MODERATE
    parallel: ParallelDef = field(default_factory=ParallelDef)
    name: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    seed: int = 42

    def to_dict(self) -> dict:
        return {
            "campaign_id": self.campaign_id,
            "parameters": [p.to_dict() for p in self.parameters],
            "objectives": [o.to_dict() for o in self.objectives],
            "budget": self.budget.to_dict(),
            "risk_preference": self.risk_preference.value,
            "parallel": self.parallel.to_dict(),
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> OptimizationSpec:
        data = data.copy()
        data["parameters"] = [ParameterDef.from_dict(p) for p in data["parameters"]]
        data["objectives"] = [ObjectiveDef.from_dict(o) for o in data["objectives"]]
        data["budget"] = BudgetDef.from_dict(data["budget"])
        data["risk_preference"] = RiskPreference(data["risk_preference"])
        data["parallel"] = ParallelDef.from_dict(data["parallel"])
        return cls(**data)
