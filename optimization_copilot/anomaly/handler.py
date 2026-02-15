"""Strategy selection for detected anomalies.

Given an ``AnomalyReport`` (from the detector), this module decides what
action to take for each anomaly: flag it, downweight the affected
observations, exclude them, or request a repeat measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from optimization_copilot.anomaly.signal_checks import SignalAnomaly
from optimization_copilot.anomaly.kpi_validator import KPIAnomaly
from optimization_copilot.anomaly.gp_outlier import GPAnomaly
from optimization_copilot.anomaly.bocpd import ChangePoint


# ── Enums ──────────────────────────────────────────────────────────────


class AnomalyAction(str, Enum):
    """Action to take for a detected anomaly."""

    FLAG = "flag"
    DOWNWEIGHT = "downweight"
    EXCLUDE = "exclude"
    REPEAT = "repeat"


# ── Data types ─────────────────────────────────────────────────────────


@dataclass
class AnomalyHandlerConfig:
    """Configuration mapping anomaly types to default actions."""

    signal_error_action: AnomalyAction = AnomalyAction.EXCLUDE
    signal_warning_action: AnomalyAction = AnomalyAction.FLAG
    kpi_out_of_range_action: AnomalyAction = AnomalyAction.EXCLUDE
    gp_outlier_action: AnomalyAction = AnomalyAction.DOWNWEIGHT
    drift_action: AnomalyAction = AnomalyAction.FLAG


@dataclass
class AnomalyDecision:
    """A decision about how to handle a detected anomaly."""

    anomaly_type: str
    action: AnomalyAction
    affected_indices: list[int]
    reason: str


@dataclass
class AnomalyReport:
    """Complete anomaly detection report from the three layers + BOCPD.

    This dataclass is defined here (rather than in detector.py) to avoid
    a circular import, since the handler consumes it.
    """

    signal_anomalies: list[SignalAnomaly] = field(default_factory=list)
    kpi_anomalies: list[KPIAnomaly] = field(default_factory=list)
    gp_anomalies: list[GPAnomaly] = field(default_factory=list)
    change_points: list[ChangePoint] = field(default_factory=list)
    is_anomalous: bool = False
    severity: str = "none"  # "none", "warning", "error"
    summary: str = ""


# ── AnomalyHandler ────────────────────────────────────────────────────


class AnomalyHandler:
    """Map anomalies in a report to concrete actions.

    Parameters
    ----------
    config : AnomalyHandlerConfig | None
        Action configuration.  Uses defaults if not provided.
    """

    def __init__(self, config: AnomalyHandlerConfig | None = None) -> None:
        self.config = config or AnomalyHandlerConfig()

    def handle(self, report: AnomalyReport) -> list[AnomalyDecision]:
        """Determine actions for every anomaly in *report*.

        Returns one ``AnomalyDecision`` per anomaly found.
        """
        decisions: list[AnomalyDecision] = []

        # Layer 1: signal anomalies
        for sa in report.signal_anomalies:
            if sa.severity == "error":
                action = self.config.signal_error_action
            else:
                action = self.config.signal_warning_action
            decisions.append(AnomalyDecision(
                anomaly_type="signal",
                action=action,
                affected_indices=list(sa.affected_indices),
                reason=sa.message,
            ))

        # Layer 2: KPI anomalies
        for ka in report.kpi_anomalies:
            decisions.append(AnomalyDecision(
                anomaly_type="kpi",
                action=self.config.kpi_out_of_range_action,
                affected_indices=[],  # KPI anomalies don't have per-point indices
                reason=ka.message,
            ))

        # Layer 3: GP outliers
        for ga in report.gp_anomalies:
            decisions.append(AnomalyDecision(
                anomaly_type="gp_outlier",
                action=self.config.gp_outlier_action,
                affected_indices=[ga.index],
                reason=ga.message,
            ))

        # Drift: change points
        for cp in report.change_points:
            decisions.append(AnomalyDecision(
                anomaly_type="drift",
                action=self.config.drift_action,
                affected_indices=[cp.index],
                reason=(
                    f"Change point at index {cp.index} "
                    f"(probability={cp.probability:.3f})"
                ),
            ))

        return decisions

    @staticmethod
    def compute_observation_weights(
        decisions: list[AnomalyDecision],
        n_observations: int,
    ) -> list[float]:
        """Compute a weight vector from anomaly decisions.

        Weights:
        - ``FLAG``  -> 1.0 (keep as-is)
        - ``DOWNWEIGHT`` -> 0.5
        - ``EXCLUDE`` -> 0.0
        - ``REPEAT`` -> 1.0 (keep, but trigger repeat elsewhere)

        When multiple decisions affect the same index, the *minimum*
        weight wins.
        """
        weights = [1.0] * n_observations

        _action_weights = {
            AnomalyAction.FLAG: 1.0,
            AnomalyAction.DOWNWEIGHT: 0.5,
            AnomalyAction.EXCLUDE: 0.0,
            AnomalyAction.REPEAT: 1.0,
        }

        for decision in decisions:
            w = _action_weights.get(decision.action, 1.0)
            for idx in decision.affected_indices:
                if 0 <= idx < n_observations:
                    weights[idx] = min(weights[idx], w)

        return weights
