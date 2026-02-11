"""Data models for the meta-learning optimizer selector.

Provides cross-campaign outcome tracking, learned parameter containers,
and the MetaAdvice output that feeds back into the MetaController.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.models import Phase, ProblemFingerprint
from optimization_copilot.meta_controller.controller import SwitchingThresholds
from optimization_copilot.portfolio.scorer import ScoringWeights


# ── Per-backend performance in a campaign ─────────────────


@dataclass
class BackendPerformance:
    """Performance metrics for a single backend within one campaign."""

    backend_name: str
    convergence_iteration: int | None  # None if didn't converge
    final_best_kpi: float
    regret: float  # gap to oracle best
    sample_efficiency: float  # kpi_gain / n_iterations
    failure_rate: float  # fraction of failed trials
    drift_encountered: bool = False
    drift_score: float = 0.0  # 0.0 if no drift

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "convergence_iteration": self.convergence_iteration,
            "final_best_kpi": self.final_best_kpi,
            "regret": self.regret,
            "sample_efficiency": self.sample_efficiency,
            "failure_rate": self.failure_rate,
            "drift_encountered": self.drift_encountered,
            "drift_score": self.drift_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BackendPerformance:
        return cls(**data)


# ── Campaign outcome ──────────────────────────────────────


@dataclass
class CampaignOutcome:
    """Complete outcome of a finished optimization campaign."""

    campaign_id: str
    fingerprint: ProblemFingerprint
    phase_transitions: list[tuple[str, str, int]]  # (from_phase, to_phase, iteration)
    backend_performances: list[BackendPerformance]
    failure_type_counts: dict[str, int]  # FailureType.name → count
    stabilization_used: dict[str, str]  # backend → StabilizeSpec summary
    total_iterations: int
    best_kpi: float
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "fingerprint": self.fingerprint.to_dict(),
            "phase_transitions": [list(t) for t in self.phase_transitions],
            "backend_performances": [bp.to_dict() for bp in self.backend_performances],
            "failure_type_counts": dict(self.failure_type_counts),
            "stabilization_used": dict(self.stabilization_used),
            "total_iterations": self.total_iterations,
            "best_kpi": self.best_kpi,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignOutcome:
        d = data.copy()
        d["fingerprint"] = _fingerprint_from_dict(d["fingerprint"])
        d["phase_transitions"] = [tuple(t) for t in d["phase_transitions"]]
        d["backend_performances"] = [
            BackendPerformance.from_dict(bp) for bp in d["backend_performances"]
        ]
        return cls(**d)


# ── Experience record ─────────────────────────────────────


@dataclass
class ExperienceRecord:
    """A stored campaign outcome with its fingerprint key for indexing."""

    outcome: CampaignOutcome
    fingerprint_key: str  # str(outcome.fingerprint.to_tuple())

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.to_dict(),
            "fingerprint_key": self.fingerprint_key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperienceRecord:
        return cls(
            outcome=CampaignOutcome.from_dict(data["outcome"]),
            fingerprint_key=data["fingerprint_key"],
        )


# ── Configuration ─────────────────────────────────────────


@dataclass
class MetaLearningConfig:
    """Tunable parameters for meta-learning behaviour."""

    min_experiences_for_learning: int = 3  # cold-start threshold
    similarity_decay: float = 0.3  # cross-fingerprint weight decay
    weight_learning_rate: float = 0.1  # EMA rate for weight updates
    threshold_learning_rate: float = 0.05  # EMA rate for threshold updates
    recency_halflife: int = 20  # campaigns before half-weight


# ── Learned outputs ───────────────────────────────────────


@dataclass
class LearnedWeights:
    """Learned ScoringWeights for a problem fingerprint class."""

    fingerprint_key: str
    gain: float
    fail: float
    cost: float
    drift: float
    incompatibility: float
    n_campaigns: int = 0
    confidence: float = 0.0  # saturates at ~10 campaigns

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint_key": self.fingerprint_key,
            "gain": self.gain,
            "fail": self.fail,
            "cost": self.cost,
            "drift": self.drift,
            "incompatibility": self.incompatibility,
            "n_campaigns": self.n_campaigns,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedWeights:
        return cls(**data)


@dataclass
class LearnedThresholds:
    """Learned SwitchingThresholds for a problem fingerprint class."""

    fingerprint_key: str
    cold_start_min_observations: float = 10.0
    learning_plateau_length: float = 5.0
    exploitation_gain_threshold: float = -0.1
    n_campaigns: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint_key": self.fingerprint_key,
            "cold_start_min_observations": self.cold_start_min_observations,
            "learning_plateau_length": self.learning_plateau_length,
            "exploitation_gain_threshold": self.exploitation_gain_threshold,
            "n_campaigns": self.n_campaigns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedThresholds:
        return cls(**data)


@dataclass
class FailureStrategy:
    """Learned stabilization approach for a failure type."""

    failure_type: str  # FailureType.name
    best_stabilization: str  # e.g., "noise_smoothing_window=5"
    effectiveness_score: float = 0.0  # 0-1
    n_campaigns: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_type": self.failure_type,
            "best_stabilization": self.best_stabilization,
            "effectiveness_score": self.effectiveness_score,
            "n_campaigns": self.n_campaigns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailureStrategy:
        return cls(**data)


@dataclass
class DriftRobustness:
    """Learned drift resilience for a single backend."""

    backend_name: str
    drift_resilience_score: float = 0.0  # 0-1, higher = more robust
    n_drift_campaigns: int = 0
    avg_kpi_loss_under_drift: float = 0.0  # regret increase under drift

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "drift_resilience_score": self.drift_resilience_score,
            "n_drift_campaigns": self.n_drift_campaigns,
            "avg_kpi_loss_under_drift": self.avg_kpi_loss_under_drift,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DriftRobustness:
        return cls(**data)


@dataclass
class MetaAdvice:
    """Output of MetaLearningAdvisor — injected into MetaController."""

    recommended_backends: list[str] = field(default_factory=list)
    scoring_weights: ScoringWeights | None = None
    switching_thresholds: SwitchingThresholds | None = None
    failure_adjustments: dict[str, str] = field(default_factory=dict)
    drift_robust_backends: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reason_codes: list[str] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────


def _fingerprint_from_dict(data: dict[str, str]) -> ProblemFingerprint:
    """Reconstruct a ProblemFingerprint from its dict representation."""
    from optimization_copilot.core.models import (
        VariableType,
        ObjectiveForm,
        NoiseRegime,
        CostProfile,
        FailureInformativeness,
        DataScale,
        Dynamics,
        FeasibleRegion,
    )

    return ProblemFingerprint(
        variable_types=VariableType(data["variable_types"]),
        objective_form=ObjectiveForm(data["objective_form"]),
        noise_regime=NoiseRegime(data["noise_regime"]),
        cost_profile=CostProfile(data["cost_profile"]),
        failure_informativeness=FailureInformativeness(data["failure_informativeness"]),
        data_scale=DataScale(data["data_scale"]),
        dynamics=Dynamics(data["dynamics"]),
        feasible_region=FeasibleRegion(data["feasible_region"]),
    )
