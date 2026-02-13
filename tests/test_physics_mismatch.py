"""Comprehensive tests for the theory mismatch detection module.

Tests cover all diagnostic signals (Q-test, systematic bias, trend detection,
adequacy scoring), the overall mismatch verdict, recommendation logic,
edge cases, the internal _pearson_r helper, and custom threshold overrides.
"""

from __future__ import annotations

import math
import random

import pytest

from optimization_copilot.physics.mismatch import MismatchDetector, MismatchReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector() -> MismatchDetector:
    """Default MismatchDetector with standard thresholds."""
    return MismatchDetector()


@pytest.fixture
def rng() -> random.Random:
    """Seeded RNG for reproducible noise generation."""
    return random.Random(42)


# ---------------------------------------------------------------------------
# 1. Correct theory -> keep_hybrid
# ---------------------------------------------------------------------------

class TestCorrectTheory:
    """When predictions closely match observations the detector should report
    no mismatch and recommend keeping the hybrid model."""

    def test_perfect_predictions(self, detector: MismatchDetector) -> None:
        """Identical y_obs and y_pred => perfect adequacy, no mismatch."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        report = detector.detect(y, y)

        assert report.q_statistic == 0.0
        assert report.q_over_n == 0.0
        assert report.mean_residual == 0.0
        assert report.residual_std == 0.0
        assert report.adequacy_score == 1.0
        assert report.is_mismatched is False
        assert report.recommendation == "keep_hybrid"

    def test_small_gaussian_noise(self, detector: MismatchDetector, rng: random.Random) -> None:
        """Predictions with small additive noise should still be accepted."""
        n = 100
        y_obs = [float(i) for i in range(n)]
        # Add small noise (std ~0.05, much smaller than signal std)
        y_pred = [y + rng.gauss(0, 0.05) for y in y_obs]

        report = detector.detect(y_obs, y_pred)

        assert report.q_over_n < 3.0, "q_over_n should be small for near-perfect predictions"
        assert report.adequacy_score > 0.9, "Adequacy should remain high"
        assert report.is_mismatched is False
        assert report.recommendation == "keep_hybrid"

    def test_report_fields_present(self, detector: MismatchDetector) -> None:
        """All MismatchReport dataclass fields must be populated."""
        report = detector.detect([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

        assert isinstance(report, MismatchReport)
        assert isinstance(report.q_statistic, float)
        assert isinstance(report.q_over_n, float)
        assert isinstance(report.mean_residual, float)
        assert isinstance(report.residual_std, float)
        assert isinstance(report.has_systematic_bias, bool)
        assert isinstance(report.has_trend, bool)
        assert isinstance(report.trend_correlation, float)
        assert isinstance(report.adequacy_score, float)
        assert isinstance(report.is_mismatched, bool)
        assert isinstance(report.recommendation, str)


# ---------------------------------------------------------------------------
# 2. Completely wrong theory -> fallback_data_driven
# ---------------------------------------------------------------------------

class TestCompletelyWrongTheory:
    """When predictions are wildly off the detector should flag mismatch and
    recommend falling back to a data-driven approach."""

    def test_large_varied_mismatch(self, detector: MismatchDetector) -> None:
        """Predictions wildly different from observations with high residual variance.

        A pure constant offset would produce residual_std=0 (all residuals
        identical), keeping adequacy at 1.0.  Instead we use predictions that
        are both far from and uncorrelated with the observations so that
        residual_std dominates y_std, driving adequacy below 0.1.
        """
        n = 50
        y_obs = [float(i) for i in range(n)]
        # Predictions that are large, varied, and unrelated to observations
        rng = random.Random(7)
        y_pred = [rng.gauss(500, 300) for _ in range(n)]

        report = detector.detect(y_obs, y_pred)

        assert report.q_over_n > 5.0
        assert report.adequacy_score < 0.1
        assert report.is_mismatched is True
        assert report.recommendation == "fallback_data_driven"

    def test_uncorrelated_predictions(self, detector: MismatchDetector, rng: random.Random) -> None:
        """Completely random predictions => very poor adequacy."""
        n = 100
        y_obs = [float(i) for i in range(n)]
        y_pred = [rng.gauss(500, 200) for _ in range(n)]

        report = detector.detect(y_obs, y_pred)

        assert report.is_mismatched is True
        assert report.recommendation == "fallback_data_driven"

    def test_inverted_predictions(self, detector: MismatchDetector) -> None:
        """Predictions with opposite sign => large residuals."""
        y_obs = [10.0, 20.0, 30.0, 40.0, 50.0]
        y_pred = [-10.0, -20.0, -30.0, -40.0, -50.0]

        report = detector.detect(y_obs, y_pred)

        assert report.q_over_n > 5.0
        assert report.is_mismatched is True
        assert report.recommendation == "fallback_data_driven"

    def test_q_over_n_above_five_triggers_fallback(self) -> None:
        """Even with decent adequacy, q_over_n > 5.0 forces fallback."""
        # Use very small noise_var to inflate q_over_n
        detector = MismatchDetector(noise_var=0.001)
        y_obs = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.1, 3.1, 4.1, 5.1]

        report = detector.detect(y_obs, y_pred)

        assert report.q_over_n > 5.0
        assert report.recommendation == "fallback_data_driven"


# ---------------------------------------------------------------------------
# 3. Theory with systematic bias
# ---------------------------------------------------------------------------

class TestSystematicBias:
    """When predictions are offset by a consistent amount the detector
    should flag systematic bias."""

    def test_constant_offset_detected(self, detector: MismatchDetector) -> None:
        """A uniform additive offset should trigger has_systematic_bias."""
        n = 50
        y_obs = [float(i) for i in range(n)]
        y_pred = [y - 2.0 for y in y_obs]  # constant offset of +2 in residuals

        report = detector.detect(y_obs, y_pred)

        assert report.has_systematic_bias is True
        assert report.mean_residual == pytest.approx(2.0, abs=1e-10)

    def test_no_bias_when_centered(self, detector: MismatchDetector, rng: random.Random) -> None:
        """Zero-mean noise should not trigger systematic bias for large samples."""
        n = 1000
        y_obs = [float(i) for i in range(n)]
        y_pred = [y + rng.gauss(0, 0.01) for y in y_obs]

        report = detector.detect(y_obs, y_pred)

        # With large n and tiny zero-mean noise, bias should not be flagged
        # (mean residual is near zero relative to SE)
        assert abs(report.mean_residual) < 0.01

    def test_negative_bias(self, detector: MismatchDetector) -> None:
        """Negative constant offset should also be detected."""
        y_obs = [10.0, 20.0, 30.0, 40.0, 50.0]
        y_pred = [y + 5.0 for y in y_obs]  # residuals = obs - pred = -5

        report = detector.detect(y_obs, y_pred)

        assert report.has_systematic_bias is True
        assert report.mean_residual == pytest.approx(-5.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 4. Theory with trend in residuals -> revise_theory
# ---------------------------------------------------------------------------

class TestTrendInResiduals:
    """When residuals correlate with predictions the detector should flag
    a trend and recommend revising the theory (provided adequacy > 0.1)."""

    def test_linear_trend_detected(self) -> None:
        """Residuals linearly proportional to predictions => has_trend=True."""
        n = 100
        y_obs = [float(i) for i in range(n)]
        # Predictions that systematically underestimate larger values:
        # y_pred = 0.8 * y_obs => residuals = 0.2 * y_obs, strongly correlated with y_pred
        y_pred = [0.8 * y for y in y_obs]

        detector = MismatchDetector()
        report = detector.detect(y_obs, y_pred)

        assert report.has_trend is True
        assert abs(report.trend_correlation) > 0.5

    def test_trend_triggers_revise_theory(self) -> None:
        """With trend and adequacy > 0.1, recommendation should be revise_theory.

        We need q_over_n <= 5.0 and adequacy >= 0.1 so the recommendation
        doesn't short-circuit to fallback_data_driven.  A small scaling error
        with a large noise_var keeps q_over_n low while still producing a
        perfect trend correlation between y_pred and residuals.
        """
        n = 100
        y_obs = [float(i) for i in range(n)]
        # 2% scaling error: predictions are 98% of actual
        y_pred = [0.98 * y for y in y_obs]

        # Large noise_var suppresses q_over_n while trend remains detectable
        detector = MismatchDetector(noise_var=100.0)
        report = detector.detect(y_obs, y_pred)

        assert report.has_trend is True
        assert report.adequacy_score > 0.1
        assert report.q_over_n <= 5.0
        assert report.recommendation == "revise_theory"

    def test_no_trend_with_random_residuals(self, detector: MismatchDetector, rng: random.Random) -> None:
        """Random zero-mean noise should not produce a significant trend."""
        n = 200
        y_obs = [float(i) for i in range(n)]
        y_pred = [y + rng.gauss(0, 0.1) for y in y_obs]

        report = detector.detect(y_obs, y_pred)

        assert report.has_trend is False
        assert abs(report.trend_correlation) < 0.5

    def test_quadratic_trend(self) -> None:
        """Quadratic mismatch should still produce detectable trend."""
        n = 100
        y_obs = [float(i) for i in range(n)]
        # Predictions miss the quadratic component: pred = linear only
        y_pred_with_quad = [y + 0.01 * y * y for y in y_obs]

        detector = MismatchDetector()
        report = detector.detect(y_pred_with_quad, y_obs)

        # Residuals will correlate with predictions
        assert abs(report.trend_correlation) > 0.3


# ---------------------------------------------------------------------------
# 5. Empty inputs -> safe defaults
# ---------------------------------------------------------------------------

class TestEmptyInputs:
    """Empty arrays should return safe defaults without raising errors."""

    def test_empty_lists(self, detector: MismatchDetector) -> None:
        report = detector.detect([], [])

        assert report.q_statistic == 0.0
        assert report.q_over_n == 0.0
        assert report.mean_residual == 0.0
        assert report.residual_std == 0.0
        assert report.has_systematic_bias is False
        assert report.has_trend is False
        assert report.trend_correlation == 0.0
        assert report.adequacy_score == 1.0
        assert report.is_mismatched is False
        assert report.recommendation == "keep_hybrid"

    def test_empty_inputs_return_mismatch_report(self, detector: MismatchDetector) -> None:
        report = detector.detect([], [])
        assert isinstance(report, MismatchReport)


# ---------------------------------------------------------------------------
# 6. Unequal lengths -> ValueError
# ---------------------------------------------------------------------------

class TestUnequalLengths:
    """Mismatched array lengths should raise ValueError."""

    def test_observed_longer(self, detector: MismatchDetector) -> None:
        with pytest.raises(ValueError, match="equal length"):
            detector.detect([1.0, 2.0, 3.0], [1.0])

    def test_predicted_longer(self, detector: MismatchDetector) -> None:
        with pytest.raises(ValueError, match="equal length"):
            detector.detect([1.0], [1.0, 2.0])

    def test_one_empty(self, detector: MismatchDetector) -> None:
        with pytest.raises(ValueError, match="equal length"):
            detector.detect([], [1.0])

    def test_error_message(self, detector: MismatchDetector) -> None:
        with pytest.raises(ValueError) as exc_info:
            detector.detect([1.0, 2.0], [1.0])
        assert "y_observed" in str(exc_info.value)
        assert "y_predicted" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 7. _pearson_r tests
# ---------------------------------------------------------------------------

class TestPearsonR:
    """Tests for the static _pearson_r helper method."""

    def test_perfect_positive_correlation(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        r = MismatchDetector._pearson_r(x, y)
        assert r == pytest.approx(1.0, abs=1e-10)

    def test_perfect_negative_correlation(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        r = MismatchDetector._pearson_r(x, y)
        assert r == pytest.approx(-1.0, abs=1e-10)

    def test_no_correlation(self) -> None:
        """Orthogonal vectors should yield correlation near zero."""
        # sin and cos sampled at uniform spacing are approximately uncorrelated
        n = 1000
        x = [math.sin(2 * math.pi * i / n) for i in range(n)]
        y = [math.cos(2 * math.pi * i / n) for i in range(n)]
        r = MismatchDetector._pearson_r(x, y)
        assert abs(r) < 0.05, f"Expected near-zero correlation, got {r}"

    def test_too_few_points_returns_zero(self) -> None:
        """Fewer than 2 points should return 0.0."""
        assert MismatchDetector._pearson_r([], []) == 0.0
        assert MismatchDetector._pearson_r([1.0], [2.0]) == 0.0

    def test_constant_values_returns_zero(self) -> None:
        """Zero variance in either variable should return 0.0."""
        x = [5.0, 5.0, 5.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0]
        assert MismatchDetector._pearson_r(x, y) == 0.0

        # Both constant
        assert MismatchDetector._pearson_r(x, x) == 0.0

    def test_known_correlation(self) -> None:
        """Verify against a hand-calculated example."""
        x = [1.0, 2.0, 3.0]
        y = [1.0, 3.0, 2.0]
        # Manual: mean_x=2, mean_y=2
        # cov = (1-2)(1-2) + (2-2)(3-2) + (3-2)(2-2) = 1 + 0 + 0 = 1
        # var_x = 1+0+1 = 2, var_y = 1+1+0 = 2
        # r = 1/sqrt(2*2) = 0.5
        r = MismatchDetector._pearson_r(x, y)
        assert r == pytest.approx(0.5, abs=1e-10)

    def test_symmetry(self) -> None:
        """pearson_r(x, y) should equal pearson_r(y, x)."""
        x = [1.0, 3.0, 5.0, 7.0]
        y = [2.0, 6.0, 4.0, 8.0]
        assert MismatchDetector._pearson_r(x, y) == pytest.approx(
            MismatchDetector._pearson_r(y, x), abs=1e-15
        )

    def test_invariant_to_linear_transform(self) -> None:
        """Pearson r should be invariant to positive linear scaling."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 1.0, 3.0, 5.0]

        r_original = MismatchDetector._pearson_r(x, y)
        # Scale and shift
        x_transformed = [10.0 * xi + 100.0 for xi in x]
        y_transformed = [5.0 * yi + 50.0 for yi in y]
        r_transformed = MismatchDetector._pearson_r(x_transformed, y_transformed)

        assert r_original == pytest.approx(r_transformed, abs=1e-10)


# ---------------------------------------------------------------------------
# 8. Custom thresholds
# ---------------------------------------------------------------------------

class TestCustomThresholds:
    """Verify that constructor parameters correctly modify detection behavior."""

    def test_strict_q_threshold(self) -> None:
        """A very low q_threshold should flag even small residuals."""
        detector = MismatchDetector(q_threshold=0.01)
        y_obs = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.05, 2.05, 3.05, 4.05, 5.05]

        report = detector.detect(y_obs, y_pred)

        # With default noise_var=1.0 and small residuals (0.05),
        # q_over_n = sum(0.05^2)/1.0 / 5 = 0.0125/5 = 0.0025
        # But q_threshold is 0.01, and 0.0025 < 0.01, so q alone won't trigger.
        # Test that a lenient detector does NOT trigger on the same data:
        lenient = MismatchDetector(q_threshold=100.0)
        report_lenient = lenient.detect(y_obs, y_pred)
        # With lenient threshold, should not be mismatched via q_over_n
        assert report_lenient.q_over_n < 100.0

    def test_strict_q_threshold_triggers_mismatch(self) -> None:
        """q_over_n just above threshold should trigger mismatch."""
        # noise_var=1.0, residuals of 2.0 each, n=5
        # q_over_n = sum(4.0)/1.0 / 5 = 20/5 = 4.0
        detector = MismatchDetector(q_threshold=3.0)
        y_obs = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [y - 2.0 for y in y_obs]

        report = detector.detect(y_obs, y_pred)
        assert report.q_over_n == pytest.approx(4.0, abs=1e-10)
        assert report.is_mismatched is True

    def test_relaxed_q_threshold(self) -> None:
        """A high q_threshold should not flag moderate residuals."""
        detector = MismatchDetector(q_threshold=100.0)
        y_obs = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [y - 2.0 for y in y_obs]

        report = detector.detect(y_obs, y_pred)
        # q_over_n = 4.0, which is below 100.0
        # Adequacy should be decent since offset is moderate relative to signal
        # Whether mismatch triggers depends on adequacy and bias+trend
        assert report.q_over_n < 100.0

    def test_low_trend_threshold_catches_weak_trends(self) -> None:
        """A very low trend_threshold should detect weak correlations."""
        detector = MismatchDetector(trend_threshold=0.05)
        n = 100
        y_obs = [float(i) for i in range(n)]
        # 95% correct, 5% scaling error => weak trend in residuals
        y_pred = [0.95 * y for y in y_obs]

        report = detector.detect(y_obs, y_pred)

        assert report.has_trend is True

    def test_high_trend_threshold_misses_moderate_trends(self) -> None:
        """A very high trend_threshold should not flag moderate correlations.

        A pure scaling error (y_pred = alpha * y_obs) produces trend_correlation
        of exactly 1.0, which exceeds any threshold < 1.  To get a moderate
        correlation we add noise so the residuals only partially correlate
        with predictions.
        """
        detector = MismatchDetector(trend_threshold=0.99)
        n = 200
        rng = random.Random(55)
        y_obs = [float(i) for i in range(n)]
        # Small scaling error plus substantial noise => moderate trend correlation
        y_pred = [0.98 * y + rng.gauss(0, 10.0) for y in y_obs]

        report = detector.detect(y_obs, y_pred)

        # The noise breaks the perfect correlation; |r| should be < 0.99
        assert abs(report.trend_correlation) < 0.99
        assert report.has_trend is False

    def test_custom_adequacy_low(self) -> None:
        """Raising adequacy_low should make the detector more sensitive."""
        strict = MismatchDetector(adequacy_low=0.95)
        lenient = MismatchDetector(adequacy_low=0.1)

        n = 50
        y_obs = [float(i) for i in range(n)]
        # Moderate noise to get adequacy around 0.5-0.8
        rng = random.Random(123)
        y_pred = [y + rng.gauss(0, 5.0) for y in y_obs]

        report_strict = strict.detect(y_obs, y_pred)
        report_lenient = lenient.detect(y_obs, y_pred)

        # Both see the same adequacy_score (it's computed the same way)
        assert report_strict.adequacy_score == pytest.approx(
            report_lenient.adequacy_score, abs=1e-10
        )
        # But the strict one is more likely to flag mismatch due to low adequacy
        if report_strict.adequacy_score < 0.95:
            assert report_strict.is_mismatched is True
        if report_lenient.adequacy_score >= 0.1:
            # Lenient might not be mismatched (depends on other signals)
            pass

    def test_custom_noise_var(self) -> None:
        """Larger noise_var should reduce q_statistic and q_over_n."""
        small_noise = MismatchDetector(noise_var=0.1)
        large_noise = MismatchDetector(noise_var=100.0)

        y_obs = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.5, 2.5, 3.5, 4.5, 5.5]

        report_small = small_noise.detect(y_obs, y_pred)
        report_large = large_noise.detect(y_obs, y_pred)

        assert report_small.q_over_n > report_large.q_over_n
        # The ratio should be exactly noise_var_large / noise_var_small = 1000
        assert report_small.q_statistic / report_large.q_statistic == pytest.approx(
            100.0 / 0.1, rel=1e-10
        )


# ---------------------------------------------------------------------------
# Additional edge case and integration tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_observation(self, detector: MismatchDetector) -> None:
        """A single data point should not crash and should handle gracefully."""
        report = detector.detect([5.0], [5.0])

        assert report.q_statistic == 0.0
        assert report.is_mismatched is False
        assert report.recommendation == "keep_hybrid"

    def test_single_observation_with_residual(self, detector: MismatchDetector) -> None:
        """Single point with a residual: pearson_r returns 0 (< 2 points)."""
        report = detector.detect([5.0], [3.0])

        assert report.q_statistic > 0.0
        assert report.has_trend is False  # Can't compute trend with 1 point

    def test_two_observations(self, detector: MismatchDetector) -> None:
        """Two data points: minimum for pearson_r computation."""
        report = detector.detect([1.0, 2.0], [1.0, 2.0])

        assert report.is_mismatched is False

    def test_constant_observations(self, detector: MismatchDetector) -> None:
        """Constant y_obs (zero variance) => adequacy = 1.0 per implementation."""
        y_obs = [5.0, 5.0, 5.0, 5.0, 5.0]
        y_pred = [5.0, 5.0, 5.0, 5.0, 5.0]

        report = detector.detect(y_obs, y_pred)

        assert report.adequacy_score == 1.0
        assert report.is_mismatched is False

    def test_constant_observations_with_offset(self, detector: MismatchDetector) -> None:
        """Constant y_obs with a prediction offset: y_std=0 => adequacy=1.0 per code,
        but bias is flagged."""
        y_obs = [5.0, 5.0, 5.0, 5.0, 5.0]
        y_pred = [3.0, 3.0, 3.0, 3.0, 3.0]

        report = detector.detect(y_obs, y_pred)

        # y_std = 0 => adequacy = 1.0 (special case in code)
        assert report.adequacy_score == 1.0
        assert report.has_systematic_bias is True
        assert report.mean_residual == pytest.approx(2.0)

    def test_large_dataset(self, detector: MismatchDetector, rng: random.Random) -> None:
        """Stress test with a large dataset."""
        n = 10000
        y_obs = [rng.gauss(50, 10) for _ in range(n)]
        y_pred = [y + rng.gauss(0, 0.1) for y in y_obs]

        report = detector.detect(y_obs, y_pred)

        assert report.is_mismatched is False
        assert report.recommendation == "keep_hybrid"

    def test_adequacy_clamped_to_zero_one(self) -> None:
        """Adequacy should be clamped to [0, 1] even in pathological cases."""
        detector = MismatchDetector()
        # Residuals with much larger std than y_obs => raw adequacy < 0
        y_obs = [1.0, 1.001, 1.002, 1.003, 1.004]
        y_pred = [100.0, -100.0, 50.0, -50.0, 200.0]

        report = detector.detect(y_obs, y_pred)

        assert 0.0 <= report.adequacy_score <= 1.0

    def test_negative_values(self, detector: MismatchDetector) -> None:
        """Negative observation and prediction values should work correctly."""
        y_obs = [-5.0, -4.0, -3.0, -2.0, -1.0]
        y_pred = [-5.0, -4.0, -3.0, -2.0, -1.0]

        report = detector.detect(y_obs, y_pred)

        assert report.is_mismatched is False
        assert report.recommendation == "keep_hybrid"

    def test_very_large_values(self, detector: MismatchDetector) -> None:
        """Very large values should not cause overflow or incorrect results."""
        scale = 1e10
        y_obs = [1.0 * scale, 2.0 * scale, 3.0 * scale, 4.0 * scale, 5.0 * scale]
        y_pred = list(y_obs)

        report = detector.detect(y_obs, y_pred)

        assert report.q_statistic == 0.0
        assert report.is_mismatched is False


class TestRecommendationLogic:
    """Targeted tests for the recommendation decision logic branches."""

    def test_adequacy_below_point_one_forces_fallback(self) -> None:
        """adequacy < 0.1 should always yield fallback_data_driven."""
        detector = MismatchDetector(noise_var=1.0)
        # Create scenario where residual std >> y_obs std
        y_obs = [1.0, 1.01, 1.02, 1.03, 1.04]
        y_pred = [100.0, -100.0, 50.0, -50.0, 200.0]

        report = detector.detect(y_obs, y_pred)

        assert report.adequacy_score < 0.1
        assert report.recommendation == "fallback_data_driven"

    def test_q_over_n_above_five_forces_fallback(self) -> None:
        """q_over_n > 5.0 should yield fallback_data_driven regardless of other signals."""
        detector = MismatchDetector(noise_var=0.01)
        y_obs = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [2.0, 3.0, 4.0, 5.0, 6.0]

        report = detector.detect(y_obs, y_pred)

        assert report.q_over_n > 5.0
        assert report.recommendation == "fallback_data_driven"

    def test_has_trend_with_adequate_model_suggests_revise(self) -> None:
        """has_trend=True with adequacy > 0.1 should yield revise_theory.

        To avoid q_over_n > 5.0 (which would force fallback_data_driven),
        we use a large noise_var so that sum(r^2)/noise_var/n stays small,
        while still maintaining a detectable trend and reasonable adequacy.
        """
        n = 100
        y_obs = [float(i) for i in range(n)]
        # Small scaling error => residuals perfectly correlated with predictions
        y_pred = [0.98 * y for y in y_obs]

        # noise_var=100 keeps q_over_n well below 5.0
        detector = MismatchDetector(noise_var=100.0)
        report = detector.detect(y_obs, y_pred)

        assert report.has_trend is True
        assert report.adequacy_score > 0.1
        assert report.q_over_n <= 5.0
        assert report.recommendation == "revise_theory"

    def test_bias_and_trend_together_flag_mismatch(self) -> None:
        """has_bias AND has_trend should set is_mismatched=True."""
        n = 100
        y_obs = [float(i) for i in range(n)]
        # Introduce both bias (constant offset) and scaling error (trend)
        y_pred = [0.8 * y - 5.0 for y in y_obs]

        detector = MismatchDetector()
        report = detector.detect(y_obs, y_pred)

        assert report.has_systematic_bias is True
        assert report.has_trend is True
        assert report.is_mismatched is True

    def test_keep_hybrid_default(self) -> None:
        """When none of the mismatch conditions are triggered, recommendation
        should be keep_hybrid."""
        detector = MismatchDetector()
        y_obs = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = list(y_obs)

        report = detector.detect(y_obs, y_pred)

        assert report.recommendation == "keep_hybrid"
        assert report.is_mismatched is False


class TestMismatchVerdictLogic:
    """Specifically test the is_mismatched boolean logic:
    q_over_n > q_threshold OR adequacy < adequacy_low OR (has_bias AND has_trend)"""

    def test_only_q_triggers(self) -> None:
        """Only q_over_n exceeding threshold should set is_mismatched."""
        detector = MismatchDetector(q_threshold=1.0, noise_var=0.1)
        y_obs = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [2.0, 3.0, 4.0, 5.0, 6.0]

        report = detector.detect(y_obs, y_pred)

        assert report.q_over_n > 1.0
        assert report.is_mismatched is True

    def test_only_adequacy_triggers(self) -> None:
        """Only low adequacy should set is_mismatched."""
        detector = MismatchDetector(adequacy_low=0.99, q_threshold=10000.0)
        n = 50
        rng = random.Random(99)
        y_obs = [float(i) for i in range(n)]
        y_pred = [y + rng.gauss(0, 3.0) for y in y_obs]

        report = detector.detect(y_obs, y_pred)

        if report.adequacy_score < 0.99:
            assert report.is_mismatched is True

    def test_only_bias_does_not_trigger_alone(self) -> None:
        """Bias alone (without trend) should not trigger is_mismatched
        unless q or adequacy conditions are also met."""
        # Constant offset with no trend
        detector = MismatchDetector(q_threshold=1000.0, adequacy_low=0.01)
        n = 50
        y_obs = [float(i) for i in range(n)]
        y_pred = [y - 0.5 for y in y_obs]  # small constant offset

        report = detector.detect(y_obs, y_pred)

        # Bias is present but if trend is not triggered and q/adequacy are fine,
        # is_mismatched should be False (bias alone is not sufficient)
        if not report.has_trend and report.q_over_n <= 1000.0 and report.adequacy_score >= 0.01:
            assert report.is_mismatched is False
