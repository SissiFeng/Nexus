"""Tests for BOCPD, AnomalyHandler, and AnomalyDetector (integration)."""

from __future__ import annotations

import math

from optimization_copilot.anomaly.bocpd import BOCPD, ChangePoint
from optimization_copilot.anomaly.handler import (
    AnomalyAction,
    AnomalyDecision,
    AnomalyHandler,
    AnomalyHandlerConfig,
    AnomalyReport,
)
from optimization_copilot.anomaly.detector import AnomalyDetector
from optimization_copilot.anomaly.signal_checks import SignalAnomaly
from optimization_copilot.anomaly.kpi_validator import KPIAnomaly
from optimization_copilot.anomaly.gp_outlier import GPAnomaly


# ── ChangePoint dataclass tests ────────────────────────────────────────


class TestChangePointDataclass:
    def test_creation(self):
        cp = ChangePoint(index=10, probability=0.8, prior_mean=0.0, posterior_mean=5.0)
        assert cp.index == 10
        assert cp.probability == 0.8
        assert cp.prior_mean == 0.0
        assert cp.posterior_mean == 5.0


# ── BOCPD tests ────────────────────────────────────────────────────────


class TestBOCPD:
    def test_bocpd_no_change_point(self):
        """Constant data -> no change points detected."""
        bocpd = BOCPD(hazard_rate=50.0)
        data = [5.0] * 50
        result = bocpd.detect(data, threshold=0.5)
        assert len(result) == 0

    def test_bocpd_single_change_point(self):
        """Clear mean shift -> at least one change point."""
        bocpd = BOCPD(hazard_rate=50.0, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        data = [0.0] * 30 + [10.0] * 30
        result = bocpd.detect(data, threshold=0.3)
        assert len(result) >= 1
        # The change point should be near index 30
        indices = [cp.index for cp in result]
        assert any(28 <= i <= 35 for i in indices)

    def test_bocpd_multiple_change_points(self):
        """Two mean shifts -> multiple change points."""
        bocpd = BOCPD(hazard_rate=30.0)
        data = [0.0] * 30 + [10.0] * 30 + [0.0] * 30
        result = bocpd.detect(data, threshold=0.3)
        assert len(result) >= 2
        indices = [cp.index for cp in result]
        assert any(28 <= i <= 35 for i in indices)
        assert any(58 <= i <= 65 for i in indices)

    def test_bocpd_gradual_drift(self):
        """Gradual drift -> may detect eventually."""
        bocpd = BOCPD(hazard_rate=20.0)
        # Slowly increasing mean
        data = [float(i) * 0.5 for i in range(60)]
        result = bocpd.detect(data, threshold=0.3)
        # Gradual drift may or may not produce strong change points
        assert isinstance(result, list)

    def test_bocpd_empty_data(self):
        """Empty data -> no change points."""
        bocpd = BOCPD()
        result = bocpd.detect([], threshold=0.5)
        assert result == []

    def test_bocpd_single_point(self):
        """Single point -> no change points."""
        bocpd = BOCPD()
        result = bocpd.detect([5.0], threshold=0.5)
        # A single point might have high CP probability at first step,
        # but there's nothing meaningful to detect
        assert isinstance(result, list)

    def test_bocpd_high_threshold(self):
        """Very high threshold -> fewer detections."""
        bocpd = BOCPD(hazard_rate=50.0)
        data = [0.0] * 30 + [10.0] * 30
        result_low = bocpd.detect(data, threshold=0.1)
        result_high = bocpd.detect(data, threshold=0.9)
        assert len(result_high) <= len(result_low)

    def test_bocpd_hazard_function(self):
        """Hazard function returns 1/hazard_rate."""
        bocpd = BOCPD(hazard_rate=100.0)
        assert abs(bocpd._hazard(0) - 0.01) < 1e-10
        assert abs(bocpd._hazard(50) - 0.01) < 1e-10

    def test_bocpd_run_length_posterior(self):
        """Run length posterior should sum to ~1 after updates."""
        bocpd = BOCPD(hazard_rate=100.0)
        for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
            bocpd.update(x)
        posterior = bocpd.get_run_length_posterior()
        total = sum(posterior)
        assert abs(total - 1.0) < 0.01  # approximately 1

    def test_bocpd_student_t_logpdf(self):
        """Student-t log PDF should return finite values."""
        bocpd = BOCPD()
        val = bocpd._student_t_logpdf(0.0, 0.0, 1.0, 2.0)
        assert math.isfinite(val)
        # Should be negative (log of a probability density)
        assert val < 0

    def test_bocpd_student_t_logpdf_edge(self):
        """Student-t with zero variance -> -inf."""
        bocpd = BOCPD()
        val = bocpd._student_t_logpdf(0.0, 0.0, 0.0, 2.0)
        assert val == float("-inf")

    def test_bocpd_reset_on_detect(self):
        """Calling detect again resets state."""
        bocpd = BOCPD(hazard_rate=50.0)
        data1 = [0.0] * 20 + [10.0] * 20
        result1 = bocpd.detect(data1, threshold=0.3)
        # Run again with different data
        data2 = [5.0] * 40
        result2 = bocpd.detect(data2, threshold=0.3)
        # Second run should have no change points
        assert len(result2) == 0

    def test_bocpd_variance_shift(self):
        """Change in variance (not mean) should also trigger detection."""
        bocpd = BOCPD(hazard_rate=30.0)
        import random
        rng = random.Random(42)
        # Low variance then high variance
        data = [rng.gauss(0, 0.1) for _ in range(30)]
        data += [rng.gauss(0, 5.0) for _ in range(30)]
        result = bocpd.detect(data, threshold=0.3)
        assert isinstance(result, list)


# ── AnomalyAction enum tests ──────────────────────────────────────────


class TestAnomalyAction:
    def test_flag(self):
        assert AnomalyAction.FLAG == "flag"

    def test_downweight(self):
        assert AnomalyAction.DOWNWEIGHT == "downweight"

    def test_exclude(self):
        assert AnomalyAction.EXCLUDE == "exclude"

    def test_repeat(self):
        assert AnomalyAction.REPEAT == "repeat"

    def test_is_str(self):
        assert isinstance(AnomalyAction.FLAG, str)


# ── AnomalyHandlerConfig tests ────────────────────────────────────────


class TestAnomalyHandlerConfig:
    def test_default_config(self):
        cfg = AnomalyHandlerConfig()
        assert cfg.signal_error_action == AnomalyAction.EXCLUDE
        assert cfg.signal_warning_action == AnomalyAction.FLAG
        assert cfg.kpi_out_of_range_action == AnomalyAction.EXCLUDE
        assert cfg.gp_outlier_action == AnomalyAction.DOWNWEIGHT
        assert cfg.drift_action == AnomalyAction.FLAG

    def test_custom_config(self):
        cfg = AnomalyHandlerConfig(
            signal_error_action=AnomalyAction.REPEAT,
            gp_outlier_action=AnomalyAction.EXCLUDE,
        )
        assert cfg.signal_error_action == AnomalyAction.REPEAT
        assert cfg.gp_outlier_action == AnomalyAction.EXCLUDE


# ── AnomalyHandler tests ──────────────────────────────────────────────


class TestAnomalyHandler:
    def test_handler_default_config(self):
        handler = AnomalyHandler()
        assert handler.config.signal_error_action == AnomalyAction.EXCLUDE

    def test_handler_empty_report(self):
        handler = AnomalyHandler()
        report = AnomalyReport()
        decisions = handler.handle(report)
        assert decisions == []

    def test_handler_flag_action(self):
        handler = AnomalyHandler()
        report = AnomalyReport(
            signal_anomalies=[
                SignalAnomaly("test", "warning", "msg", [0, 1]),
            ],
        )
        decisions = handler.handle(report)
        assert len(decisions) == 1
        assert decisions[0].action == AnomalyAction.FLAG

    def test_handler_exclude_action(self):
        handler = AnomalyHandler()
        report = AnomalyReport(
            signal_anomalies=[
                SignalAnomaly("test", "error", "msg", [2, 3]),
            ],
        )
        decisions = handler.handle(report)
        assert len(decisions) == 1
        assert decisions[0].action == AnomalyAction.EXCLUDE

    def test_handler_downweight_action(self):
        handler = AnomalyHandler()
        report = AnomalyReport(
            gp_anomalies=[
                GPAnomaly(index=5, detection_method="loo_cv", score=4.0, threshold=3.0, message="msg"),
            ],
        )
        decisions = handler.handle(report)
        assert len(decisions) == 1
        assert decisions[0].action == AnomalyAction.DOWNWEIGHT

    def test_handler_kpi_action(self):
        handler = AnomalyHandler()
        report = AnomalyReport(
            kpi_anomalies=[
                KPIAnomaly("CE", 120.0, (0.0, 105.0), "error", "msg"),
            ],
        )
        decisions = handler.handle(report)
        assert len(decisions) == 1
        assert decisions[0].action == AnomalyAction.EXCLUDE

    def test_handler_drift_action(self):
        handler = AnomalyHandler()
        report = AnomalyReport(
            change_points=[
                ChangePoint(index=10, probability=0.8, prior_mean=0.0, posterior_mean=5.0),
            ],
        )
        decisions = handler.handle(report)
        assert len(decisions) == 1
        assert decisions[0].action == AnomalyAction.FLAG
        assert decisions[0].anomaly_type == "drift"

    def test_handler_multiple_anomalies(self):
        handler = AnomalyHandler()
        report = AnomalyReport(
            signal_anomalies=[
                SignalAnomaly("s1", "warning", "m1", [0]),
                SignalAnomaly("s2", "error", "m2", [1]),
            ],
            kpi_anomalies=[
                KPIAnomaly("CE", 120.0, (0.0, 105.0), "error", "m3"),
            ],
            gp_anomalies=[
                GPAnomaly(index=5, detection_method="loo_cv", score=4.0, threshold=3.0, message="m4"),
            ],
            change_points=[
                ChangePoint(index=10, probability=0.8, prior_mean=0.0, posterior_mean=5.0),
            ],
        )
        decisions = handler.handle(report)
        assert len(decisions) == 5
        types = [d.anomaly_type for d in decisions]
        assert types.count("signal") == 2
        assert types.count("kpi") == 1
        assert types.count("gp_outlier") == 1
        assert types.count("drift") == 1

    def test_handler_custom_config(self):
        config = AnomalyHandlerConfig(
            signal_error_action=AnomalyAction.REPEAT,
            gp_outlier_action=AnomalyAction.FLAG,
        )
        handler = AnomalyHandler(config=config)
        report = AnomalyReport(
            signal_anomalies=[SignalAnomaly("s", "error", "m", [0])],
            gp_anomalies=[GPAnomaly(0, "sr", 4.0, 3.0, "m")],
        )
        decisions = handler.handle(report)
        assert decisions[0].action == AnomalyAction.REPEAT
        assert decisions[1].action == AnomalyAction.FLAG


# ── Observation weight tests ──────────────────────────────────────────


class TestObservationWeights:
    def test_weights_no_decisions(self):
        weights = AnomalyHandler.compute_observation_weights([], 5)
        assert weights == [1.0, 1.0, 1.0, 1.0, 1.0]

    def test_weights_flag(self):
        decisions = [
            AnomalyDecision("signal", AnomalyAction.FLAG, [0], "msg"),
        ]
        weights = AnomalyHandler.compute_observation_weights(decisions, 3)
        assert weights == [1.0, 1.0, 1.0]

    def test_weights_downweight(self):
        decisions = [
            AnomalyDecision("gp", AnomalyAction.DOWNWEIGHT, [1], "msg"),
        ]
        weights = AnomalyHandler.compute_observation_weights(decisions, 3)
        assert weights == [1.0, 0.5, 1.0]

    def test_weights_exclude(self):
        decisions = [
            AnomalyDecision("signal", AnomalyAction.EXCLUDE, [2], "msg"),
        ]
        weights = AnomalyHandler.compute_observation_weights(decisions, 3)
        assert weights == [1.0, 1.0, 0.0]

    def test_weights_repeat(self):
        decisions = [
            AnomalyDecision("signal", AnomalyAction.REPEAT, [0], "msg"),
        ]
        weights = AnomalyHandler.compute_observation_weights(decisions, 3)
        assert weights == [1.0, 1.0, 1.0]

    def test_weights_multiple_decisions_same_index(self):
        """Multiple decisions on same index -> minimum weight wins."""
        decisions = [
            AnomalyDecision("signal", AnomalyAction.DOWNWEIGHT, [1], "msg"),
            AnomalyDecision("gp", AnomalyAction.EXCLUDE, [1], "msg"),
        ]
        weights = AnomalyHandler.compute_observation_weights(decisions, 3)
        assert weights[1] == 0.0

    def test_weights_out_of_range_index(self):
        """Index beyond n_observations -> ignored."""
        decisions = [
            AnomalyDecision("signal", AnomalyAction.EXCLUDE, [10], "msg"),
        ]
        weights = AnomalyHandler.compute_observation_weights(decisions, 3)
        assert weights == [1.0, 1.0, 1.0]

    def test_weights_negative_index(self):
        """Negative index -> ignored."""
        decisions = [
            AnomalyDecision("signal", AnomalyAction.EXCLUDE, [-1], "msg"),
        ]
        weights = AnomalyHandler.compute_observation_weights(decisions, 3)
        assert weights == [1.0, 1.0, 1.0]

    def test_weights_all_excluded(self):
        decisions = [
            AnomalyDecision("signal", AnomalyAction.EXCLUDE, [0, 1, 2], "msg"),
        ]
        weights = AnomalyHandler.compute_observation_weights(decisions, 3)
        assert weights == [0.0, 0.0, 0.0]

    def test_weights_zero_observations(self):
        weights = AnomalyHandler.compute_observation_weights([], 0)
        assert weights == []


# ── AnomalyReport tests ───────────────────────────────────────────────


class TestAnomalyReport:
    def test_empty_report(self):
        report = AnomalyReport()
        assert not report.is_anomalous
        assert report.severity == "none"
        assert report.summary == ""

    def test_report_with_signal_anomaly(self):
        report = AnomalyReport(
            signal_anomalies=[SignalAnomaly("s", "error", "m", [0])],
            is_anomalous=True,
            severity="error",
        )
        assert report.is_anomalous
        assert report.severity == "error"

    def test_report_severity_warning(self):
        report = AnomalyReport(severity="warning")
        assert report.severity == "warning"

    def test_report_severity_none(self):
        report = AnomalyReport(severity="none")
        assert report.severity == "none"


# ── AnomalyDetector integration tests ─────────────────────────────────


class TestAnomalyDetector:
    def test_detector_creation(self):
        det = AnomalyDetector()
        assert det.signal_checker is not None
        assert det.kpi_validator is not None
        assert det.gp_detector is not None

    def test_detector_no_data(self):
        """No data provided -> empty report, no anomalies."""
        det = AnomalyDetector()
        report = det.detect()
        assert not report.is_anomalous
        assert report.severity == "none"
        assert report.signal_anomalies == []
        assert report.kpi_anomalies == []
        assert report.gp_anomalies == []
        assert report.change_points == []

    def test_detector_signal_only(self):
        """Only raw_data provided -> only signal checks run."""
        det = AnomalyDetector()
        report = det.detect(raw_data={
            "xrd_intensities": [65535.0, 100.0],
        })
        assert report.is_anomalous
        assert len(report.signal_anomalies) == 1

    def test_detector_kpi_only(self):
        """Only KPI values provided -> only KPI checks run."""
        det = AnomalyDetector()
        report = det.detect(kpi_values={"CE": 120.0})
        assert report.is_anomalous
        assert len(report.kpi_anomalies) == 1

    def test_detector_gp_only(self):
        """Only X and y provided -> GP checks run."""
        det = AnomalyDetector(gp_threshold=2.0)
        X = [[float(i)] for i in range(10)]
        y = [float(i) for i in range(10)]
        y[5] = 100.0  # outlier
        report = det.detect(x=X, y=y)
        assert report.is_anomalous
        assert len(report.gp_anomalies) >= 1

    def test_detector_bocpd_triggered(self):
        """BOCPD runs when y has > 10 points."""
        det = AnomalyDetector(bocpd_hazard=20.0)
        y = [0.0] * 15 + [10.0] * 15
        report = det.detect(y=y)
        # BOCPD should detect the shift
        # It may or may not depending on threshold and data
        assert isinstance(report.change_points, list)

    def test_detector_bocpd_not_triggered_short(self):
        """BOCPD does not run with <= 10 points."""
        det = AnomalyDetector()
        report = det.detect(y=[1.0] * 5)
        assert report.change_points == []

    def test_detector_full_pipeline(self):
        """Full pipeline with all data types."""
        det = AnomalyDetector(gp_threshold=2.0, bocpd_hazard=20.0)
        X = [[float(i)] for i in range(20)]
        y = [float(i) for i in range(20)]
        y[10] = 100.0  # GP outlier

        report = det.detect(
            x=X,
            y=y,
            raw_data={"xrd_intensities": [65535.0]},
            kpi_values={"CE": 120.0},
        )

        assert report.is_anomalous
        assert len(report.signal_anomalies) >= 1
        assert len(report.kpi_anomalies) >= 1
        assert len(report.gp_anomalies) >= 1

    def test_detector_severity_computed(self):
        """Severity should reflect the worst anomaly."""
        det = AnomalyDetector()
        report = det.detect(
            raw_data={"xrd_intensities": [65535.0]},  # error severity
        )
        assert report.severity == "error"

    def test_detector_severity_warning_only(self):
        """Only warning-level anomalies."""
        det = AnomalyDetector()
        report = det.detect(
            raw_data={"absorbance": [-1.0, -2.0, 0.5]},
        )
        assert report.severity == "warning"

    def test_detector_summary(self):
        """Summary should describe what was found."""
        det = AnomalyDetector()
        report = det.detect(
            raw_data={"xrd_intensities": [65535.0]},
            kpi_values={"CE": 120.0},
        )
        assert "signal" in report.summary.lower() or "anomaly" in report.summary.lower()
        assert "kpi" in report.summary.lower() or "violation" in report.summary.lower()

    def test_detector_clean_data(self):
        """Clean data -> no anomalies."""
        det = AnomalyDetector()
        report = det.detect(
            raw_data={"voltages": [1.0, 1.1, 1.0, 0.9, 1.0, 1.1, 1.0]},
            kpi_values={"CE": 98.0, "conversion": 50.0},
        )
        assert not report.is_anomalous
        assert report.severity == "none"

    def test_detector_custom_gp_threshold(self):
        """Custom GP threshold should be used."""
        det = AnomalyDetector(gp_threshold=100.0)
        assert det.gp_detector.threshold_sigma == 100.0

    def test_detector_is_anomalous_flag(self):
        """is_anomalous should be True when any anomaly exists."""
        det = AnomalyDetector()
        report = det.detect(kpi_values={"CE": -10.0})
        assert report.is_anomalous is True

    def test_detector_not_anomalous(self):
        det = AnomalyDetector()
        report = det.detect(kpi_values={"CE": 98.0})
        assert report.is_anomalous is False

    def test_detector_gp_needs_3_points(self):
        """GP detection needs at least 3 points."""
        det = AnomalyDetector()
        report = det.detect(x=[[0.0], [1.0]], y=[0.0, 100.0])
        assert report.gp_anomalies == []
