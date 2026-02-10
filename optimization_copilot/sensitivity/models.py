"""Data models for decision sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParameterSensitivity:
    """Sensitivity of the KPI to a single parameter."""

    parameter_name: str
    sensitivity_score: float
    correlation: float
    local_gradient: float
    rank: int
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "sensitivity_score": self.sensitivity_score,
            "correlation": self.correlation,
            "local_gradient": self.local_gradient,
            "rank": self.rank,
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParameterSensitivity:
        return cls(**data)


@dataclass
class DecisionStability:
    """Stability of the top-K ranking under perturbation."""

    top_k: int
    stable_count: int
    stability_score: float
    margin_to_next: float
    margin_relative: float
    swapped_pairs: list[tuple[int, int]] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "stable_count": self.stable_count,
            "stability_score": self.stability_score,
            "margin_to_next": self.margin_to_next,
            "margin_relative": self.margin_relative,
            "swapped_pairs": [list(p) for p in self.swapped_pairs],
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionStability:
        data = data.copy()
        data["swapped_pairs"] = [tuple(p) for p in data.get("swapped_pairs", [])]
        return cls(**data)


@dataclass
class SensitivityReport:
    """Complete sensitivity analysis report."""

    parameter_sensitivities: list[ParameterSensitivity]
    decision_stability: DecisionStability
    robustness_score: float
    most_sensitive_parameter: str
    least_sensitive_parameter: str
    recommendations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_sensitivities": [
                ps.to_dict() for ps in self.parameter_sensitivities
            ],
            "decision_stability": self.decision_stability.to_dict(),
            "robustness_score": self.robustness_score,
            "most_sensitive_parameter": self.most_sensitive_parameter,
            "least_sensitive_parameter": self.least_sensitive_parameter,
            "recommendations": list(self.recommendations),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SensitivityReport:
        data = data.copy()
        data["parameter_sensitivities"] = [
            ParameterSensitivity.from_dict(ps)
            for ps in data.get("parameter_sensitivities", [])
        ]
        data["decision_stability"] = DecisionStability.from_dict(
            data["decision_stability"]
        )
        return cls(**data)
