"""Tests for the concept drift detection module."""

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    Phase,
    VariableType,
)
from optimization_copilot.drift.detector import DriftDetector, DriftReport


# ── Helpers ───────────────────────────────────────────────


def _make_specs() -> list[ParameterSpec]:
    return [
        ParameterSpec(name="x1", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
        ParameterSpec(name="x2", type=VariableType.CONTINUOUS, lower=0.0, upper=10.0),
    ]


def _make_snapshot(
    kpi_values: list[float],
    x1_values: list[float] | None = None,
    x2_values: list[float] | None = None,
) -> CampaignSnapshot:
    """Build a snapshot with controlled KPI and parameter sequences."""
    n = len(kpi_values)
    if x1_values is None:
        x1_values = [float(i) for i in range(n)]
    if x2_values is None:
        x2_values = [float(i) * 0.5 for i in range(n)]
    obs = []
    for i in range(n):
        obs.append(
            Observation(
                iteration=i,
                parameters={"x1": x1_values[i], "x2": x2_values[i]},
                kpi_values={"y": kpi_values[i]},
                timestamp=float(i),
            )
        )
    return CampaignSnapshot(
        campaign_id="drift-test",
        parameter_specs=_make_specs(),
        observations=obs,
        objective_names=["y"],
        objective_directions=["maximize"],
        current_iteration=n,
    )


# ── Test: Stable data (no drift) ─────────────────────────


class TestNoDrift:
    def test_stable_data_no_drift(self):
        """Constant-ish KPI across both windows should produce no drift."""
        kpi = [5.0 + 0.1 * (i % 3) for i in range(30)]
        snap = _make_snapshot(kpi)
        det = DriftDetector(reference_window=10, test_window=10)
        report = det.detect(snap)

        assert report.drift_detected is False
        assert report.drift_score < 0.3
        assert report.drift_type == "none"
        assert report.recommended_action == "continue"

    def test_mildly_noisy_stable(self):
        """Small noise on a stable mean should not trigger drift."""
        import math

        kpi = [10.0 + 0.5 * math.sin(i) for i in range(30)]
        snap = _make_snapshot(kpi)
        det = DriftDetector(reference_window=10, test_window=10)
        report = det.detect(snap)

        assert report.drift_detected is False
        assert report.drift_type == "none"


# ── Test: Sudden drift ────────────────────────────────────


class TestSuddenDrift:
    def test_abrupt_kpi_shift(self):
        """KPI jumps abruptly at the midpoint -- should detect drift."""
        n = 30
        kpi = [1.0] * 15 + [10.0] * 15
        snap = _make_snapshot(kpi)
        det = DriftDetector(reference_window=10, test_window=10)
        report = det.detect(snap)

        assert report.drift_detected is True
        assert report.drift_score >= 0.5
        assert report.recommended_action != "continue"

    def test_large_step_change(self):
        """Very large step change produces high drift score."""
        kpi = [0.0] * 15 + [100.0] * 15
        snap = _make_snapshot(kpi)
        det = DriftDetector(reference_window=10, test_window=10)
        report = det.detect(snap)

        assert report.drift_detected is True
        assert report.drift_score >= 0.7


# ── Test: Gradual drift ──────────────────────────────────


class TestGradualDrift:
    def test_slow_trend_change(self):
        """KPI slowly drifts upward -- should eventually be detected."""
        n = 40
        kpi = [float(i) * 0.5 for i in range(n)]
        # Reference window (indices 20-29) mean ~12.25
        # Test window (indices 30-39) mean ~17.25
        snap = _make_snapshot(kpi)
        det = DriftDetector(reference_window=10, test_window=10)
        report = det.detect(snap)

        assert report.drift_detected is True
        assert report.drift_score > 0.0

    def test_classify_gradual_pattern(self):
        """Rolling scores that rise gradually classify as 'gradual'."""
        det = DriftDetector()
        scores = [0.05, 0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
        result = det.classify_drift_type(scores)
        assert result == "gradual"


# ── Test: Affected parameters identification ──────────────


class TestAffectedParameters:
    def test_parameter_correlation_flip(self):
        """When a parameter's relationship with KPI flips, flag it."""
        n = 30
        x1 = [float(i) for i in range(n)]
        # In the first half, x1 positively correlates with y.
        # In the second half, x1 negatively correlates with y.
        kpi = [float(i) for i in range(15)] + [15.0 - float(i - 15) for i in range(15, 30)]
        x2 = [0.5] * n  # x2 stays constant -- not affected.

        snap = _make_snapshot(kpi, x1_values=x1, x2_values=x2)
        det = DriftDetector(reference_window=10, test_window=10)
        report = det.detect(snap)

        # x1 should be flagged because its correlation with y flipped.
        assert "x1" in report.affected_parameters

    def test_unaffected_parameters_not_flagged(self):
        """Parameters with stable relationships should not be flagged."""
        n = 30
        kpi = [5.0 + 0.1 * (i % 3) for i in range(n)]
        snap = _make_snapshot(kpi)
        det = DriftDetector(reference_window=10, test_window=10)
        report = det.detect(snap)

        assert len(report.affected_parameters) == 0


# ── Test: Action recommendations ──────────────────────────


class TestActionRecommendations:
    def test_no_drift_continue(self):
        report = DriftReport(
            drift_detected=False,
            drift_score=0.1,
            drift_type="none",
            affected_parameters=[],
            recommended_action="continue",
        )
        det = DriftDetector()
        action = det.recommend_action(report, Phase.LEARNING)
        assert action == "continue"

    def test_mild_drift_reweight(self):
        report = DriftReport(
            drift_detected=True,
            drift_score=0.4,
            drift_type="gradual",
            affected_parameters=["x1"],
            recommended_action="",
        )
        det = DriftDetector()
        action = det.recommend_action(report, Phase.LEARNING)
        assert action == "reweight"

    def test_mild_drift_exploitation_reweight(self):
        report = DriftReport(
            drift_detected=True,
            drift_score=0.4,
            drift_type="gradual",
            affected_parameters=["x1"],
            recommended_action="",
        )
        det = DriftDetector()
        action = det.recommend_action(report, Phase.EXPLOITATION)
        assert action == "reweight"

    def test_moderate_drift_re_screen(self):
        report = DriftReport(
            drift_detected=True,
            drift_score=0.6,
            drift_type="sudden",
            affected_parameters=["x1", "x2"],
            recommended_action="",
        )
        det = DriftDetector()
        action = det.recommend_action(report, Phase.LEARNING)
        assert action == "re_screen"

    def test_severe_drift_learning_re_learn(self):
        report = DriftReport(
            drift_detected=True,
            drift_score=0.8,
            drift_type="sudden",
            affected_parameters=["x1"],
            recommended_action="",
        )
        det = DriftDetector()
        action = det.recommend_action(report, Phase.LEARNING)
        assert action == "re_learn"

    def test_severe_drift_exploitation_restart(self):
        report = DriftReport(
            drift_detected=True,
            drift_score=0.8,
            drift_type="sudden",
            affected_parameters=["x1", "x2"],
            recommended_action="",
        )
        det = DriftDetector()
        action = det.recommend_action(report, Phase.EXPLOITATION)
        assert action == "restart"

    def test_cold_start_no_reweight(self):
        """During cold start, mild drift should just continue."""
        report = DriftReport(
            drift_detected=True,
            drift_score=0.35,
            drift_type="gradual",
            affected_parameters=["x1"],
            recommended_action="",
        )
        det = DriftDetector()
        action = det.recommend_action(report, Phase.COLD_START)
        assert action == "continue"


# ── Test: Drift-aware window sizing ───────────────────────


class TestDriftAwareWindow:
    def test_no_drift_returns_full_history(self):
        """Without drift, the full observation count should be returned."""
        kpi = [5.0 + 0.1 * (i % 3) for i in range(30)]
        snap = _make_snapshot(kpi)
        det = DriftDetector(reference_window=10, test_window=10)
        window = det.compute_drift_aware_window(snap)
        assert window == 30

    def test_drift_shrinks_window(self):
        """With drift, the returned window should be smaller."""
        kpi = [1.0] * 15 + [20.0] * 15
        snap = _make_snapshot(kpi)
        det = DriftDetector(reference_window=10, test_window=10)
        window = det.compute_drift_aware_window(snap)
        assert window < 30

    def test_severe_drift_small_window(self):
        """Severe drift should produce a notably small window."""
        kpi = [0.0] * 15 + [100.0] * 15
        snap = _make_snapshot(kpi)
        det = DriftDetector(reference_window=10, test_window=10)
        window = det.compute_drift_aware_window(snap)
        # Should keep at most ~25% of data at extreme drift.
        assert window <= 15

    def test_window_at_least_test_window(self):
        """Even under severe drift, the window is at least test_window."""
        kpi = [0.0] * 15 + [1000.0] * 15
        snap = _make_snapshot(kpi)
        det = DriftDetector(reference_window=10, test_window=10)
        window = det.compute_drift_aware_window(snap)
        assert window >= 10


# ── Test: Edge cases ──────────────────────────────────────


class TestEdgeCases:
    def test_too_few_observations(self):
        """With fewer observations than needed, report no drift gracefully."""
        kpi = [1.0, 2.0, 3.0]
        snap = _make_snapshot(kpi)
        det = DriftDetector(reference_window=10, test_window=10)
        report = det.detect(snap)

        assert report.drift_detected is False
        assert report.drift_score == 0.0
        assert report.drift_type == "none"
        assert report.recommended_action == "continue"
        assert report.window_stats.get("reason") == "insufficient_data"

    def test_all_same_kpi(self):
        """All identical KPI values should produce no drift."""
        kpi = [7.0] * 30
        snap = _make_snapshot(kpi)
        det = DriftDetector(reference_window=10, test_window=10)
        report = det.detect(snap)

        assert report.drift_detected is False
        assert report.drift_score == 0.0
        assert report.drift_type == "none"

    def test_exactly_minimum_observations(self):
        """Exactly reference_window + test_window observations should work."""
        kpi = [1.0] * 10 + [1.0] * 10
        snap = _make_snapshot(kpi)
        det = DriftDetector(reference_window=10, test_window=10)
        report = det.detect(snap)

        assert report.drift_detected is False
        assert report.drift_type == "none"

    def test_single_observation(self):
        """A single observation should not crash."""
        snap = _make_snapshot([42.0])
        det = DriftDetector(reference_window=10, test_window=10)
        report = det.detect(snap)

        assert report.drift_detected is False

    def test_empty_observations(self):
        """No observations at all should not crash."""
        snap = CampaignSnapshot(
            campaign_id="empty",
            parameter_specs=_make_specs(),
            observations=[],
            objective_names=["y"],
            objective_directions=["maximize"],
        )
        det = DriftDetector(reference_window=10, test_window=10)
        report = det.detect(snap)

        assert report.drift_detected is False
        assert report.drift_score == 0.0

    def test_observations_with_failures(self):
        """Observations marked as failures should be excluded."""
        n = 30
        kpi_vals = [5.0] * n
        obs = []
        for i in range(n):
            is_fail = i % 10 == 0  # Every 10th is a failure.
            obs.append(
                Observation(
                    iteration=i,
                    parameters={"x1": float(i), "x2": float(i) * 0.5},
                    kpi_values={"y": kpi_vals[i]},
                    is_failure=is_fail,
                    timestamp=float(i),
                )
            )
        snap = CampaignSnapshot(
            campaign_id="fail-test",
            parameter_specs=_make_specs(),
            observations=obs,
            objective_names=["y"],
            objective_directions=["maximize"],
            current_iteration=n,
        )
        det = DriftDetector(reference_window=10, test_window=10)
        report = det.detect(snap)
        # Stable KPI among successful observations -- no drift.
        assert report.drift_detected is False


# ── Test: classify_drift_type standalone ──────────────────


class TestClassifyDriftType:
    def test_all_low_scores(self):
        det = DriftDetector()
        assert det.classify_drift_type([0.01, 0.02, 0.05, 0.03]) == "none"

    def test_sudden_spike(self):
        det = DriftDetector()
        scores = [0.02, 0.03, 0.01, 0.02, 0.01, 0.7, 0.8, 0.9]
        result = det.classify_drift_type(scores)
        assert result == "sudden"

    def test_recurring_pattern(self):
        det = DriftDetector()
        scores = [0.1, 0.5, 0.1, 0.6, 0.1, 0.5, 0.1, 0.6]
        result = det.classify_drift_type(scores)
        assert result == "recurring"

    def test_empty_scores(self):
        det = DriftDetector()
        assert det.classify_drift_type([]) == "none"

    def test_single_score(self):
        det = DriftDetector()
        assert det.classify_drift_type([0.5]) == "gradual"

    def test_gradual_ramp(self):
        det = DriftDetector()
        scores = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        result = det.classify_drift_type(scores)
        assert result == "gradual"
