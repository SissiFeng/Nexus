"""Hypothesis Engine data models.

Provides core data structures for representing scientific hypotheses,
predictions, and evidence within the optimization copilot framework.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HypothesisStatus(str, Enum):
    """Lifecycle status of a hypothesis."""

    PROPOSED = "proposed"
    TESTING = "testing"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


@dataclass
class Prediction:
    """A quantitative prediction derived from a hypothesis.

    Parameters
    ----------
    hypothesis_id : str
        Identifier of the parent hypothesis.
    variable : str
        Name of the predicted variable.
    predicted_value : float
        Point prediction.
    confidence_interval : tuple[float, float]
        ``(lower, upper)`` bounds for the prediction.
    condition : dict[str, float]
        Input conditions under which the prediction is made.
    """

    hypothesis_id: str
    variable: str
    predicted_value: float
    confidence_interval: tuple[float, float]  # (lower, upper)
    condition: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "variable": self.variable,
            "predicted_value": self.predicted_value,
            "confidence_interval": list(self.confidence_interval),
            "condition": dict(self.condition),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Prediction:
        ci = d["confidence_interval"]
        return cls(
            hypothesis_id=d["hypothesis_id"],
            variable=d["variable"],
            predicted_value=d["predicted_value"],
            confidence_interval=(ci[0], ci[1]),
            condition=d.get("condition", {}),
        )


@dataclass
class Evidence:
    """An observation that supports or refutes a prediction.

    Parameters
    ----------
    prediction : Prediction
        The prediction being tested.
    observed_value : float
        The actual observed value.
    within_ci : bool
        Whether the observation falls within the confidence interval.
    residual : float
        ``observed_value - predicted_value``.
    timestamp : float
        Unix timestamp of when the evidence was recorded.
    """

    prediction: Prediction
    observed_value: float
    within_ci: bool = False
    residual: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        lo, hi = self.prediction.confidence_interval
        self.within_ci = lo <= self.observed_value <= hi
        self.residual = self.observed_value - self.prediction.predicted_value
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction.to_dict(),
            "observed_value": self.observed_value,
            "within_ci": self.within_ci,
            "residual": self.residual,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Evidence:
        pred = Prediction.from_dict(d["prediction"])
        ev = cls(
            prediction=pred,
            observed_value=d["observed_value"],
            timestamp=d.get("timestamp", 0.0),
        )
        # __post_init__ already computed within_ci and residual
        return ev


@dataclass
class Hypothesis:
    """A testable scientific hypothesis with lifecycle tracking.

    Parameters
    ----------
    id : str
        Unique identifier (e.g. ``"H0001"``).
    description : str
        Human-readable description.
    equation : str | None
        Symbolic expression from symbolic regression.
    causal_mechanism : str | None
        Causal path description from the causal graph.
    source : str
        Origin of the hypothesis: ``"symreg"``, ``"causal"``,
        ``"fanova"``, ``"correlation"``, or ``"manual"``.
    status : HypothesisStatus
        Current lifecycle status.
    evidence : list[Evidence]
        Collected evidence entries.
    predictions : list[Prediction]
        Predictions generated from this hypothesis.
    bic_score : float | None
        Bayesian Information Criterion score.
    support_count : int
        Number of supporting evidence entries.
    refute_count : int
        Number of refuting evidence entries.
    created_at : float
        Unix timestamp of creation.
    n_parameters : int
        Number of free parameters (complexity proxy).
    """

    id: str
    description: str
    equation: str | None = None
    causal_mechanism: str | None = None
    source: str = "manual"
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    evidence: list[Evidence] = field(default_factory=list)
    predictions: list[Prediction] = field(default_factory=list)
    bic_score: float | None = None
    support_count: int = 0
    refute_count: int = 0
    created_at: float = field(default_factory=time.time)
    n_parameters: int = 1

    def add_evidence(self, evidence: Evidence) -> None:
        """Add evidence and update support/refute counts."""
        self.evidence.append(evidence)
        if evidence.within_ci:
            self.support_count += 1
        else:
            self.refute_count += 1

    def evidence_ratio(self) -> float:
        """Return support / (support + refute), or 0.5 if no evidence."""
        total = self.support_count + self.refute_count
        return self.support_count / total if total > 0 else 0.5

    def to_dict(self) -> dict[str, Any]:
        """Serialize the hypothesis to a plain dict."""
        return {
            "id": self.id,
            "description": self.description,
            "equation": self.equation,
            "causal_mechanism": self.causal_mechanism,
            "source": self.source,
            "status": self.status.value,
            "evidence": [e.to_dict() for e in self.evidence],
            "predictions": [p.to_dict() for p in self.predictions],
            "bic_score": self.bic_score,
            "support_count": self.support_count,
            "refute_count": self.refute_count,
            "created_at": self.created_at,
            "n_parameters": self.n_parameters,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Hypothesis:
        """Deserialize a hypothesis from a plain dict."""
        evidence_list = [Evidence.from_dict(e) for e in d.get("evidence", [])]
        prediction_list = [Prediction.from_dict(p) for p in d.get("predictions", [])]
        return cls(
            id=d["id"],
            description=d["description"],
            equation=d.get("equation"),
            causal_mechanism=d.get("causal_mechanism"),
            source=d.get("source", "manual"),
            status=HypothesisStatus(d.get("status", "proposed")),
            evidence=evidence_list,
            predictions=prediction_list,
            bic_score=d.get("bic_score"),
            support_count=d.get("support_count", 0),
            refute_count=d.get("refute_count", 0),
            created_at=d.get("created_at", 0.0),
            n_parameters=d.get("n_parameters", 1),
        )
