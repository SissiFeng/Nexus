"""Three-layer campaign output structure.

Layer 1 — **Actionable Dashboard** (experimentalist sees daily):
    Ranked candidate table, stage gate protocol, batch recommendation.

Layer 2 — **Campaign Intelligence** (PI sees):
    Pareto front summary, model performance metrics, learning report.

Layer 3 — **Agent Reasoning Log** (debug):
    Diagnostics, fANOVA, mechanism analysis, execution traces.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.campaign.ranker import RankedTable
from optimization_copilot.campaign.stage_gate import ScreeningProtocol


# ------------------------------------------------------------------
# Layer 1: Actionable Dashboard
# ------------------------------------------------------------------


@dataclass
class Layer1Dashboard:
    """What the experimentalist sees — next batch and screening protocol.

    Parameters
    ----------
    ranked_table : RankedTable
        Candidates sorted by acquisition priority.
    batch_size : int
        How many to synthesize this round.
    screening_protocol : ScreeningProtocol | None
        Stage gate protocol (``None`` if no multi-fidelity setup).
    iteration : int
        Current campaign iteration number.
    """

    ranked_table: RankedTable
    batch_size: int
    screening_protocol: ScreeningProtocol | None
    iteration: int

    @property
    def next_batch(self) -> list[dict[str, Any]]:
        """Top-n candidates as dicts for the experimentalist."""
        return [c.to_dict() for c in self.ranked_table.top_n(self.batch_size)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "batch_size": self.batch_size,
            "next_batch": self.next_batch,
            "full_ranking": self.ranked_table.to_dict(),
            "screening_protocol": (
                self.screening_protocol.to_dict()
                if self.screening_protocol
                else None
            ),
        }


# ------------------------------------------------------------------
# Layer 2: Campaign Intelligence
# ------------------------------------------------------------------


@dataclass
class ModelMetrics:
    """Surrogate model performance summary.

    Parameters
    ----------
    objective_name : str
        Which objective this model targets.
    n_training_points : int
        Number of observations used for fitting.
    y_mean : float
        Mean of observed values.
    y_std : float
        Std-dev of observed values.
    fit_duration_ms : float
        How long fit() took.
    """

    objective_name: str
    n_training_points: int
    y_mean: float
    y_std: float
    fit_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_name": self.objective_name,
            "n_training_points": self.n_training_points,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "fit_duration_ms": self.fit_duration_ms,
        }


@dataclass
class LearningReport:
    """What the model learned from new experimental data.

    Generated when new results are ingested.  Compares the model's
    prior predictions against actual outcomes.

    Parameters
    ----------
    new_observations : list[dict]
        Summaries of new data points.
    prediction_errors : list[dict]
        Per-observation comparison: ``{name, objective, predicted, actual, error, pct_error}``.
    mean_absolute_error : float
        Average absolute prediction error.
    model_updated : bool
        Whether the model was refitted.
    summary : str
        Human-readable learning summary.
    """

    new_observations: list[dict[str, Any]]
    prediction_errors: list[dict[str, Any]]
    mean_absolute_error: float
    model_updated: bool
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "new_observations": self.new_observations,
            "prediction_errors": self.prediction_errors,
            "mean_absolute_error": self.mean_absolute_error,
            "model_updated": self.model_updated,
            "summary": self.summary,
        }


@dataclass
class Layer2Intelligence:
    """What the PI sees — Pareto front, model tracking, learnings.

    Parameters
    ----------
    pareto_summary : dict | None
        Pareto front analysis results.
    model_metrics : list[ModelMetrics]
        One per fitted objective.
    learning_report : LearningReport | None
        Set when new data was ingested (``None`` on first iteration).
    iteration_count : int
        Total iterations completed.
    """

    pareto_summary: dict[str, Any] | None
    model_metrics: list[ModelMetrics]
    learning_report: LearningReport | None
    iteration_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "pareto_summary": self.pareto_summary,
            "model_metrics": [m.to_dict() for m in self.model_metrics],
            "learning_report": (
                self.learning_report.to_dict()
                if self.learning_report
                else None
            ),
            "iteration_count": self.iteration_count,
        }


# ------------------------------------------------------------------
# Layer 3: Agent Reasoning Log
# ------------------------------------------------------------------


@dataclass
class Layer3Reasoning:
    """Debug-level reasoning log — diagnostics, fANOVA, traces.

    Parameters
    ----------
    diagnostic_summary : dict | None
        Output from DiagnosticEngine (14-signal summary).
    fanova_result : dict | None
        Feature importance analysis.
    execution_traces : list[dict]
        Execution trace log from TracedScientificAgent.
    additional : dict
        Any other debug information.
    """

    diagnostic_summary: dict[str, Any] | None = None
    fanova_result: dict[str, Any] | None = None
    execution_traces: list[dict[str, Any]] = field(default_factory=list)
    additional: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnostic_summary": self.diagnostic_summary,
            "fanova_result": self.fanova_result,
            "execution_traces": self.execution_traces,
            "additional": self.additional,
        }


# ------------------------------------------------------------------
# Top-level deliverable
# ------------------------------------------------------------------


@dataclass
class CampaignDeliverable:
    """Complete 3-layer output from one campaign iteration.

    Parameters
    ----------
    iteration : int
        Campaign iteration number.
    dashboard : Layer1Dashboard
        Actionable output for experimentalists.
    intelligence : Layer2Intelligence
        Strategic output for PIs.
    reasoning : Layer3Reasoning
        Debug output for developers.
    timestamp : float
        When this deliverable was generated.
    """

    iteration: int
    dashboard: Layer1Dashboard
    intelligence: Layer2Intelligence
    reasoning: Layer3Reasoning
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    @property
    def next_batch(self) -> list[dict[str, Any]]:
        """Shortcut to the top-n candidates for synthesis."""
        return self.dashboard.next_batch

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "dashboard": self.dashboard.to_dict(),
            "intelligence": self.intelligence.to_dict(),
            "reasoning": self.reasoning.to_dict(),
        }
