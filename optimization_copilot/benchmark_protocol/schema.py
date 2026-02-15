"""JSON schema for benchmark definition."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import json


@dataclass
class ParameterDefinition:
    """Schema for a single parameter in the benchmark."""

    name: str
    type: str  # "continuous", "discrete", "categorical"
    lower: float | None = None
    upper: float | None = None
    categories: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary, omitting None values."""
        d: dict[str, Any] = {"name": self.name, "type": self.type}
        if self.lower is not None:
            d["lower"] = self.lower
        if self.upper is not None:
            d["upper"] = self.upper
        if self.categories is not None:
            d["categories"] = list(self.categories)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParameterDefinition:
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            type=data["type"],
            lower=data.get("lower"),
            upper=data.get("upper"),
            categories=data.get("categories"),
        )


@dataclass
class ObjectiveDefinition:
    """Schema for a single objective in the benchmark."""

    name: str
    direction: str  # "minimize" or "maximize"
    target: float | None = None  # optional known optimum

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary, omitting None values."""
        d: dict[str, Any] = {"name": self.name, "direction": self.direction}
        if self.target is not None:
            d["target"] = self.target
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectiveDefinition:
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            direction=data["direction"],
            target=data.get("target"),
        )


@dataclass
class BenchmarkSchema:
    """Complete schema for an SDL benchmark."""

    name: str
    version: str
    description: str
    domain: str  # e.g. "electrochemistry", "catalysis"
    parameters: list[ParameterDefinition]
    objectives: list[ObjectiveDefinition]
    constraints: list[dict[str, Any]] = field(default_factory=list)
    evaluation_budget: int = 100
    noise_level: float = 0.01
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "domain": self.domain,
            "parameters": [p.to_dict() for p in self.parameters],
            "objectives": [o.to_dict() for o in self.objectives],
            "constraints": list(self.constraints),
            "evaluation_budget": self.evaluation_budget,
            "noise_level": self.noise_level,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkSchema:
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            version=data["version"],
            description=data["description"],
            domain=data["domain"],
            parameters=[ParameterDefinition.from_dict(p) for p in data["parameters"]],
            objectives=[ObjectiveDefinition.from_dict(o) for o in data["objectives"]],
            constraints=data.get("constraints", []),
            evaluation_budget=data.get("evaluation_budget", 100),
            noise_level=data.get("noise_level", 0.01),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> BenchmarkSchema:
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors: list[str] = []

        if not self.name:
            errors.append("Benchmark name must not be empty")
        if not self.version:
            errors.append("Benchmark version must not be empty")
        if not self.parameters:
            errors.append("Benchmark must have at least one parameter")
        if not self.objectives:
            errors.append("Benchmark must have at least one objective")

        valid_param_types = {"continuous", "discrete", "categorical"}
        for p in self.parameters:
            if p.type not in valid_param_types:
                errors.append(
                    f"Parameter '{p.name}' has invalid type '{p.type}'. "
                    f"Must be one of: {valid_param_types}"
                )
            if p.type in ("continuous", "discrete"):
                if p.lower is None or p.upper is None:
                    errors.append(
                        f"Parameter '{p.name}' of type '{p.type}' must have "
                        f"lower and upper bounds"
                    )
                elif p.lower >= p.upper:
                    errors.append(
                        f"Parameter '{p.name}' lower bound ({p.lower}) must be "
                        f"less than upper bound ({p.upper})"
                    )
            if p.type == "categorical":
                if not p.categories or len(p.categories) == 0:
                    errors.append(
                        f"Parameter '{p.name}' of type 'categorical' must have "
                        f"at least one category"
                    )

        valid_directions = {"minimize", "maximize"}
        for o in self.objectives:
            if o.direction not in valid_directions:
                errors.append(
                    f"Objective '{o.name}' has invalid direction '{o.direction}'. "
                    f"Must be one of: {valid_directions}"
                )

        if self.evaluation_budget <= 0:
            errors.append("Evaluation budget must be positive")
        if self.noise_level < 0:
            errors.append("Noise level must be non-negative")

        return errors
