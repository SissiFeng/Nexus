"""Data models for the Canonical Experiment Store.

Provides Artifact and Experiment dataclasses that unify parameters, KPIs,
metadata, and artifacts (curves, images, spectra) into a single structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Enums ──────────────────────────────────────────────


class ArtifactType(str, Enum):
    """Classification of artifact data types."""

    CURVE = "curve"
    BINARY = "binary"
    SPECTRAL = "spectral"
    METADATA = "metadata"
    RAW = "raw"


# ── Dataclasses ────────────────────────────────────────


@dataclass
class Artifact:
    """An artifact attached to an experiment.

    Attributes:
        artifact_id: Unique identifier for this artifact.
        artifact_type: Classification of the artifact data.
        name: Human-readable name (e.g., "eis_spectrum_run3").
        data: The artifact payload (JSON-serializable).
            - CURVE: dict with "x_values", "y_values", "metadata" keys
            - BINARY: str (base64-encoded)
            - SPECTRAL: dict with "wavelengths", "intensities" keys
            - METADATA: dict[str, Any]
            - RAW: Any JSON-serializable value
        metadata: Additional metadata about the artifact.
    """

    artifact_id: str
    artifact_type: ArtifactType
    name: str
    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type.value,
            "name": self.name,
            "data": self.data,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Artifact:
        data = data.copy()
        data["artifact_type"] = ArtifactType(data["artifact_type"])
        return cls(**data)


@dataclass
class Experiment:
    """A single experimental record in the store.

    Richer than core.models.Observation: includes artifacts and explicit
    column-role tracking.  Bridges to Observation for engine consumption.
    """

    experiment_id: str
    campaign_id: str
    iteration: int
    parameters: dict[str, Any]
    kpi_values: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    timestamp: float = 0.0
    qc_passed: bool = True
    is_failure: bool = False
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "campaign_id": self.campaign_id,
            "iteration": self.iteration,
            "parameters": dict(self.parameters),
            "kpi_values": dict(self.kpi_values),
            "metadata": dict(self.metadata),
            "artifacts": [a.to_dict() for a in self.artifacts],
            "timestamp": self.timestamp,
            "qc_passed": self.qc_passed,
            "is_failure": self.is_failure,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Experiment:
        data = data.copy()
        data["artifacts"] = [Artifact.from_dict(a) for a in data.get("artifacts", [])]
        return cls(**data)
