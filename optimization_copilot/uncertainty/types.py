"""Core data types for measurement uncertainty propagation.

This module defines the shared type system between Layer 2 (Optimization
Engine) and Layer 3 (Scientific Reasoning Agents).  Every Extractor emits
``MeasurementWithUncertainty``; the propagation layer converts a set of
measurements into an ``ObservationWithNoise`` that carries per-point noise
variance for the heteroscedastic GP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Enums ──────────────────────────────────────────────────────────────


class UncertaintySource(str, Enum):
    """Origin category for a reported uncertainty."""

    INSTRUMENT = "instrument"    # hardware spec (e.g. potentiostat accuracy)
    MODEL_FIT = "model_fit"      # fitting residual (e.g. NLLS covariance)
    REPETITION = "repetition"    # repeat-experiment variance
    PROPAGATED = "propagated"    # error-propagated from upstream KPIs


class PropagationMethod(str, Enum):
    """How uncertainty is propagated from KPIs to the objective."""

    LINEAR = "linear"           # σ²_obj = Σ w_i² σ²_i
    DELTA = "delta"             # delta method: J^T Σ J
    MONTE_CARLO = "monte_carlo" # sampling-based


class ConfidenceLevel(str, Enum):
    """Qualitative reliability bucket for quick filtering."""

    HIGH = "high"       # confidence >= 0.8
    MEDIUM = "medium"   # 0.5 <= confidence < 0.8
    LOW = "low"         # confidence < 0.5


# ── Core Data Types ────────────────────────────────────────────────────


@dataclass
class MeasurementWithUncertainty:
    """A KPI value with associated uncertainty information.

    Standard output format for all uncertainty-aware extractors.
    Also the standard input format for Layer 3 Agents (v6+).

    Parameters
    ----------
    value : float
        Extracted KPI value (μ).
    variance : float
        Measurement variance (σ²).
    confidence : float
        Overall reliability score in [0, 1].
    source : str
        Label identifying the uncertainty origin
        (e.g. ``"EIS_Rct_ensemble"``).
    fit_residual : float | None
        Fitting residual, if applicable.
    n_points_used : int | None
        Number of raw data points consumed.
    method : str
        Extraction method identifier (e.g. ``"interpolation"``).
    metadata : dict
        Extensible context for Agent consumption.
    """

    # Core fields
    value: float
    variance: float
    confidence: float
    source: str

    # Diagnostic fields
    fit_residual: float | None = None
    n_points_used: int | None = None
    method: str = "direct"

    # Agent-interface extension
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.variance < 0:
            raise ValueError(f"variance must be >= 0, got {self.variance}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )

    # ── Derived properties ────────────────────────────────────────

    @property
    def std(self) -> float:
        """Standard deviation (σ)."""
        return self.variance ** 0.5

    @property
    def relative_uncertainty(self) -> float:
        """Coefficient of variation (CV = σ / |μ|)."""
        if abs(self.value) < 1e-12:
            return float("inf")
        return self.std / abs(self.value)

    @property
    def is_reliable(self) -> bool:
        """Quick reliability check (pragmatic threshold)."""
        return self.confidence >= 0.5 and self.relative_uncertainty < 0.5

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Qualitative reliability bucket."""
        if self.confidence >= 0.8:
            return ConfidenceLevel.HIGH
        if self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    # ── Serialisation ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict (JSON-safe except metadata)."""
        return {
            "value": self.value,
            "variance": self.variance,
            "confidence": self.confidence,
            "source": self.source,
            "fit_residual": self.fit_residual,
            "n_points_used": self.n_points_used,
            "method": self.method,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MeasurementWithUncertainty:
        """Reconstruct from a plain dict."""
        return cls(
            value=d["value"],
            variance=d["variance"],
            confidence=d["confidence"],
            source=d["source"],
            fit_residual=d.get("fit_residual"),
            n_points_used=d.get("n_points_used"),
            method=d.get("method", "direct"),
            metadata=d.get("metadata", {}),
        )


@dataclass
class UncertaintyBudget:
    """Breakdown of variance contributions by source.

    Parameters
    ----------
    contributions : dict[str, float]
        Mapping from source label to its σ² contribution.
    total_variance : float
        Sum of all contributions.
    dominant_source : str
        Label of the largest contributor.
    """

    contributions: dict[str, float]
    total_variance: float
    dominant_source: str

    @classmethod
    def from_contributions(
        cls, contributions: dict[str, float]
    ) -> UncertaintyBudget:
        """Build from a raw contribution dict."""
        total = sum(contributions.values())
        dominant = max(contributions, key=contributions.get) if contributions else ""
        return cls(
            contributions=contributions,
            total_variance=total,
            dominant_source=dominant,
        )

    def fraction(self, source: str) -> float:
        """Fractional contribution of *source* to total variance."""
        if self.total_variance < 1e-30:
            return 0.0
        return self.contributions.get(source, 0.0) / self.total_variance

    def to_dict(self) -> dict[str, Any]:
        return {
            "contributions": dict(self.contributions),
            "total_variance": self.total_variance,
            "dominant_source": self.dominant_source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UncertaintyBudget:
        return cls(
            contributions=d["contributions"],
            total_variance=d["total_variance"],
            dominant_source=d["dominant_source"],
        )


@dataclass
class ObservationWithNoise:
    """Observation augmented with per-point heteroscedastic noise.

    This is what the propagation layer outputs and the heteroscedastic
    GP consumes.

    Parameters
    ----------
    objective_value : float
        Scalar objective function value.
    noise_variance : float
        Noise variance (σ²_i) for this observation.
    kpi_contributions : list[dict] | None
        Per-KPI breakdown (value, weight, variance).
    uncertainty_budget : UncertaintyBudget | None
        Full variance decomposition.
    metadata : dict
        Extensible context for Orchestrator / Agent.
    """

    objective_value: float
    noise_variance: float

    kpi_contributions: list[dict[str, Any]] | None = None
    uncertainty_budget: UncertaintyBudget | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.noise_variance < 0:
            raise ValueError(
                f"noise_variance must be >= 0, got {self.noise_variance}"
            )

    @property
    def noise_std(self) -> float:
        return self.noise_variance ** 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_value": self.objective_value,
            "noise_variance": self.noise_variance,
            "kpi_contributions": self.kpi_contributions,
            "uncertainty_budget": (
                self.uncertainty_budget.to_dict()
                if self.uncertainty_budget
                else None
            ),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ObservationWithNoise:
        budget = d.get("uncertainty_budget")
        return cls(
            objective_value=d["objective_value"],
            noise_variance=d["noise_variance"],
            kpi_contributions=d.get("kpi_contributions"),
            uncertainty_budget=(
                UncertaintyBudget.from_dict(budget) if budget else None
            ),
            metadata=d.get("metadata", {}),
        )


@dataclass
class PropagationResult:
    """Result of propagating KPI uncertainties to the objective.

    Parameters
    ----------
    objective_value : float
        Computed objective function value.
    objective_variance : float
        Propagated variance on the objective.
    method : PropagationMethod
        Which propagation method was used.
    budget : UncertaintyBudget
        Per-source variance breakdown.
    kpi_details : list[dict]
        Per-KPI propagation detail.
    """

    objective_value: float
    objective_variance: float
    method: PropagationMethod
    budget: UncertaintyBudget
    kpi_details: list[dict[str, Any]] = field(default_factory=list)

    def to_observation_with_noise(self) -> ObservationWithNoise:
        """Convert to the GP-ready format."""
        return ObservationWithNoise(
            objective_value=self.objective_value,
            noise_variance=self.objective_variance,
            kpi_contributions=self.kpi_details,
            uncertainty_budget=self.budget,
            metadata={
                "propagation_method": self.method.value,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_value": self.objective_value,
            "objective_variance": self.objective_variance,
            "method": self.method.value,
            "budget": self.budget.to_dict(),
            "kpi_details": self.kpi_details,
        }
