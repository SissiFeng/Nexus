"""Main anomaly detector -- orchestrates the three-layer detection pipeline.

This module ties together:
- Layer 1: ``SignalChecker``  (raw signal checks)
- Layer 2: ``KPIValidator``   (physical range validation)
- Layer 3: ``GPOutlierDetector`` (statistical outlier detection)
- Drift:   ``BOCPD``          (Bayesian change-point detection)

It returns a pure ``AnomalyReport`` -- it does **not** call agents.
The ScientificOrchestrator (or caller) dispatches to agents.
"""

from __future__ import annotations

from typing import Any

from optimization_copilot.anomaly.signal_checks import SignalChecker
from optimization_copilot.anomaly.kpi_validator import KPIValidator
from optimization_copilot.anomaly.gp_outlier import GPOutlierDetector
from optimization_copilot.anomaly.bocpd import BOCPD
from optimization_copilot.anomaly.handler import AnomalyReport
from optimization_copilot.domain_knowledge.loader import DomainConfig


# Re-export AnomalyReport so callers can do:
#   from optimization_copilot.anomaly.detector import AnomalyDetector, AnomalyReport
__all__ = ["AnomalyDetector", "AnomalyReport"]


class AnomalyDetector:
    """Three-layer anomaly detection with optional BOCPD drift detection.

    Parameters
    ----------
    domain_config : DomainConfig | None
        Optional domain configuration for KPI validation.
    gp_threshold : float
        Threshold (in sigma) for GP outlier detection (default 3.0).
    bocpd_hazard : float
        BOCPD hazard rate (expected run length before change, default 100).
    """

    def __init__(
        self,
        domain_config: DomainConfig | None = None,
        gp_threshold: float = 3.0,
        bocpd_hazard: float = 100.0,
    ) -> None:
        self.signal_checker = SignalChecker()
        self.kpi_validator = KPIValidator(domain_config=domain_config)
        self.gp_detector = GPOutlierDetector(threshold_sigma=gp_threshold)
        self.bocpd_hazard = bocpd_hazard

    def detect(
        self,
        x: list[list[float]] | None = None,
        y: list[float] | None = None,
        raw_data: dict[str, Any] | None = None,
        kpi_values: dict[str, float] | None = None,
    ) -> AnomalyReport:
        """Run the full three-layer detection pipeline.

        Parameters
        ----------
        x : list[list[float]] | None
            Input features for GP outlier detection.
        y : list[float] | None
            Target values for GP outlier detection and BOCPD.
        raw_data : dict | None
            Raw measurement data for Layer 1 signal checks.
        kpi_values : dict[str, float] | None
            Extracted KPI values for Layer 2 validation.

        Returns
        -------
        AnomalyReport
            Complete detection report across all layers.
        """
        report = AnomalyReport()

        # Layer 1: signal checks
        if raw_data is not None:
            report.signal_anomalies = self.signal_checker.check_all(raw_data)

        # Layer 2: KPI validation
        if kpi_values is not None:
            report.kpi_anomalies = self.kpi_validator.validate_all(kpi_values)

        # Layer 3: GP outlier detection (standardized residual)
        if x is not None and y is not None and len(y) >= 3:
            # For GP-based detection, we need predictions.
            # Use LOO-based detection which doesn't require pre-computed predictions.
            report.gp_anomalies = self.gp_detector.detect_loo_outlier(
                X=x, y=y, noise=0.01,
            )

        # Drift detection via BOCPD (needs > 10 points)
        if y is not None and len(y) > 10:
            bocpd = BOCPD(hazard_rate=self.bocpd_hazard)
            report.change_points = bocpd.detect(y, threshold=0.5)

        # Compute aggregate severity and summary
        report.is_anomalous = bool(
            report.signal_anomalies
            or report.kpi_anomalies
            or report.gp_anomalies
            or report.change_points
        )
        report.severity = self._compute_severity(report)
        report.summary = self._build_summary(report)

        return report

    @staticmethod
    def _compute_severity(report: AnomalyReport) -> str:
        """Determine the overall severity of the report."""
        severities: list[str] = []

        for sa in report.signal_anomalies:
            severities.append(sa.severity)
        for ka in report.kpi_anomalies:
            severities.append(ka.severity)

        # GP anomalies and change points are always at least warnings
        if report.gp_anomalies:
            severities.append("warning")
        if report.change_points:
            severities.append("warning")

        if "error" in severities:
            return "error"
        if "warning" in severities:
            return "warning"
        return "none"

    @staticmethod
    def _build_summary(report: AnomalyReport) -> str:
        """Build a human-readable summary string."""
        parts: list[str] = []

        n_signal = len(report.signal_anomalies)
        n_kpi = len(report.kpi_anomalies)
        n_gp = len(report.gp_anomalies)
        n_cp = len(report.change_points)

        if n_signal:
            parts.append(f"{n_signal} signal anomaly(ies)")
        if n_kpi:
            parts.append(f"{n_kpi} KPI violation(s)")
        if n_gp:
            parts.append(f"{n_gp} GP outlier(s)")
        if n_cp:
            parts.append(f"{n_cp} change point(s)")

        if not parts:
            return "No anomalies detected."
        return "Detected: " + ", ".join(parts) + "."
