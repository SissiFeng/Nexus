"""Agent interface contracts for Layer 3 consumption (v6+ stubs).

These data classes define the contract between Layer 2 (Optimization Engine)
and Layer 3 (Scientific Reasoning Agents).  In v4, only the *types* are
provided — no agent logic is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.uncertainty.types import (
    MeasurementWithUncertainty,
    ObservationWithNoise,
)


@dataclass
class AgentContext:
    """Context package that Layer 3 Agents receive from Layer 2.

    v4 provides the fields below.  Agent implementations consume
    them starting in v6.

    Parameters
    ----------
    measurements : list[MeasurementWithUncertainty]
        All KPI measurements from the current trial.
    observation : ObservationWithNoise
        The propagated observation for the GP.
    gp_state : dict
        Snapshot of the GP model state (hyper-parameters, etc.).
    history : list[ObservationWithNoise]
        All previous propagated observations.
    domain_config : dict
        Domain-specific configuration (instrument specs, constraints).
    """

    measurements: list[MeasurementWithUncertainty]
    observation: ObservationWithNoise
    gp_state: dict[str, Any] = field(default_factory=dict)
    history: list[ObservationWithNoise] = field(default_factory=list)
    domain_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "measurements": [m.to_dict() for m in self.measurements],
            "observation": self.observation.to_dict(),
            "gp_state": self.gp_state,
            "history": [h.to_dict() for h in self.history],
            "domain_config": self.domain_config,
        }


@dataclass
class OptimizationFeedback:
    """Feedback from Layer 3 Agents back to Layer 2.

    In v4, only ``noise_override`` is expected to be used (by the
    Extractor-level self-check).  All other fields become active in v6+.

    Parameters
    ----------
    noise_override : float | None
        Override the propagated noise estimate for the current point.
    prior_adjustment : dict | None
        Adjust GP priors (v7b).
    constraint_update : dict | None
        Update search-space constraints (v6a).
    rerun_suggested : bool
        Whether the agent suggests rerunning the experiment.
    rerun_reason : str
        Human-readable reason for the rerun suggestion.
    """

    noise_override: float | None = None
    prior_adjustment: dict[str, Any] | None = None
    constraint_update: dict[str, Any] | None = None
    rerun_suggested: bool = False
    rerun_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "noise_override": self.noise_override,
            "prior_adjustment": self.prior_adjustment,
            "constraint_update": self.constraint_update,
            "rerun_suggested": self.rerun_suggested,
            "rerun_reason": self.rerun_reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OptimizationFeedback:
        return cls(
            noise_override=d.get("noise_override"),
            prior_adjustment=d.get("prior_adjustment"),
            constraint_update=d.get("constraint_update"),
            rerun_suggested=d.get("rerun_suggested", False),
            rerun_reason=d.get("rerun_reason", ""),
        )
