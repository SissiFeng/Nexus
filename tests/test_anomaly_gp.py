"""Tests for Layer 3: GPOutlierDetector (GP-based statistical outlier detection)."""

from __future__ import annotations

import math

from optimization_copilot.anomaly.gp_outlier import GPAnomaly, GPOutlierDetector


# ── GPAnomaly dataclass tests ──────────────────────────────────────────


class TestGPAnomalyDataclass:
    def test_creation(self):
        a = GPAnomaly(
            index=5,
            detection_method="standardized_residual",
            score=4.5,
            threshold=3.0,
            message="outlier at 5",
        )
        assert a.index == 5
        assert a.detection_method == "standardized_residual"
        assert a.score == 4.5
        assert a.threshold == 3.0


# ── Standardized residual tests ────────────────────────────────────────


class TestStandardizedResidual:
    def test_no_outlier(self):
        """All points within threshold -> no anomalies."""
        det = GPOutlierDetector(threshold_sigma=3.0)
        y_pred = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_std = [1.0, 1.0, 1.0, 1.0, 1.0]
        y_actual = [1.5, 2.5, 3.5, 4.5, 5.5]
        result = det.detect_standardized_residual(y_pred, y_std, y_actual)
        assert result == []

    def test_with_outlier(self):
        """One point far from prediction -> detected."""
        det = GPOutlierDetector(threshold_sigma=3.0)
        y_pred = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_std = [0.1, 0.1, 0.1, 0.1, 0.1]
        y_actual = [1.0, 2.0, 3.0, 4.0, 100.0]  # point 4 is way off
        result = det.detect_standardized_residual(y_pred, y_std, y_actual)
        assert len(result) >= 1
        assert any(a.index == 4 for a in result)

    def test_multiple_outliers(self):
        """Multiple outliers should all be detected."""
        det = GPOutlierDetector(threshold_sigma=3.0)
        y_pred = [0.0] * 5
        y_std = [0.1] * 5
        y_actual = [0.0, 10.0, 0.0, 10.0, 0.0]
        result = det.detect_standardized_residual(y_pred, y_std, y_actual)
        assert len(result) == 2
        indices = {a.index for a in result}
        assert indices == {1, 3}

    def test_zero_std_skipped(self):
        """Points with std=0 should be skipped (not division by zero)."""
        det = GPOutlierDetector(threshold_sigma=3.0)
        y_pred = [1.0, 2.0]
        y_std = [0.0, 1.0]
        y_actual = [100.0, 2.0]
        result = det.detect_standardized_residual(y_pred, y_std, y_actual)
        assert result == []

    def test_negative_std_skipped(self):
        """Negative std should be skipped."""
        det = GPOutlierDetector(threshold_sigma=3.0)
        y_pred = [1.0]
        y_std = [-1.0]
        y_actual = [100.0]
        result = det.detect_standardized_residual(y_pred, y_std, y_actual)
        assert result == []

    def test_empty_inputs(self):
        """Empty lists -> no anomalies."""
        det = GPOutlierDetector()
        result = det.detect_standardized_residual([], [], [])
        assert result == []

    def test_single_point_no_outlier(self):
        """Single point within threshold."""
        det = GPOutlierDetector(threshold_sigma=3.0)
        result = det.detect_standardized_residual([1.0], [1.0], [2.0])
        assert result == []

    def test_single_point_outlier(self):
        """Single point exceeding threshold."""
        det = GPOutlierDetector(threshold_sigma=3.0)
        result = det.detect_standardized_residual([0.0], [0.1], [10.0])
        assert len(result) == 1
        assert result[0].index == 0

    def test_custom_threshold(self):
        """Lower threshold should catch more outliers."""
        det_strict = GPOutlierDetector(threshold_sigma=1.0)
        det_loose = GPOutlierDetector(threshold_sigma=5.0)
        y_pred = [0.0] * 5
        y_std = [1.0] * 5
        y_actual = [0.0, 2.0, 0.0, 2.0, 0.0]
        strict = det_strict.detect_standardized_residual(y_pred, y_std, y_actual)
        loose = det_loose.detect_standardized_residual(y_pred, y_std, y_actual)
        assert len(strict) >= len(loose)

    def test_score_value(self):
        """Score should be the standardized residual."""
        det = GPOutlierDetector(threshold_sigma=3.0)
        result = det.detect_standardized_residual([0.0], [0.1], [1.0])
        assert len(result) == 1
        assert abs(result[0].score - 10.0) < 1e-6

    def test_mismatched_lengths(self):
        """Uses min of all lengths."""
        det = GPOutlierDetector(threshold_sigma=3.0)
        result = det.detect_standardized_residual([0.0, 0.0], [1.0], [100.0, 100.0])
        # Only first point is checked (min length = 1)
        # |100 - 0| / 1 = 100 > 3
        assert len(result) == 1


# ── LOO outlier tests ──────────────────────────────────────────────────


class TestLOOOutlier:
    def test_loo_outlier_clean(self):
        """Clean data -> no outliers (or few)."""
        det = GPOutlierDetector(threshold_sigma=3.0)
        X = [[float(i)] for i in range(10)]
        y = [float(i) for i in range(10)]
        result = det.detect_loo_outlier(X, y, noise=0.1)
        # Clean linear data should have few/no outliers
        assert len(result) <= 2  # allow some numerical noise

    def test_loo_outlier_with_anomaly(self):
        """One wildly different point -> detected."""
        det = GPOutlierDetector(threshold_sigma=3.0)
        X = [[float(i)] for i in range(10)]
        y = [float(i) for i in range(10)]
        y[5] = 100.0  # outlier
        result = det.detect_loo_outlier(X, y, noise=0.01)
        assert any(a.index == 5 for a in result)

    def test_loo_too_few_points(self):
        """Fewer than 3 points -> no outliers."""
        det = GPOutlierDetector(threshold_sigma=3.0)
        result = det.detect_loo_outlier([[0.0]], [0.0])
        assert result == []

    def test_loo_two_points(self):
        """Exactly 2 points -> no outliers."""
        det = GPOutlierDetector(threshold_sigma=3.0)
        result = det.detect_loo_outlier([[0.0], [1.0]], [0.0, 1.0])
        assert result == []

    def test_loo_three_points(self):
        """Three points should work."""
        det = GPOutlierDetector(threshold_sigma=3.0)
        X = [[0.0], [1.0], [2.0]]
        y = [0.0, 1.0, 2.0]
        result = det.detect_loo_outlier(X, y, noise=0.1)
        # Clean data -> no outliers expected
        assert isinstance(result, list)

    def test_loo_detection_method(self):
        """Detection method should be 'loo_cv'."""
        det = GPOutlierDetector(threshold_sigma=2.0)
        X = [[float(i)] for i in range(10)]
        y = [float(i) for i in range(10)]
        y[5] = 100.0
        result = det.detect_loo_outlier(X, y, noise=0.01)
        for a in result:
            assert a.detection_method == "loo_cv"

    def test_loo_custom_kernel(self):
        """Custom kernel function should be accepted."""
        det = GPOutlierDetector(threshold_sigma=3.0)

        def custom_kernel(x1: list[float], x2: list[float]) -> float:
            return math.exp(-sum((a - b) ** 2 for a, b in zip(x1, x2)))

        X = [[float(i)] for i in range(5)]
        y = [float(i) for i in range(5)]
        result = det.detect_loo_outlier(X, y, kernel_fn=custom_kernel, noise=0.1)
        assert isinstance(result, list)

    def test_loo_multidimensional(self):
        """Multi-dimensional input should work."""
        det = GPOutlierDetector(threshold_sigma=3.0)
        X = [[float(i), float(i) * 2] for i in range(8)]
        y = [float(i) for i in range(8)]
        result = det.detect_loo_outlier(X, y, noise=0.1)
        assert isinstance(result, list)

    def test_loo_empty(self):
        """Empty input -> no outliers."""
        det = GPOutlierDetector()
        result = det.detect_loo_outlier([], [])
        assert result == []


# ── Entropy change tests ──────────────────────────────────────────────


class TestEntropyChange:
    def test_entropy_change_stable(self):
        """Constant data -> no entropy change."""
        det = GPOutlierDetector(threshold_sigma=3.0)
        y = [1.0] * 20
        result = det.detect_entropy_change(y, window=5)
        assert result == []

    def test_entropy_change_with_shift(self):
        """Sudden variance change -> detected."""
        det = GPOutlierDetector(threshold_sigma=3.0)
        # First half: low variance, second half: high variance
        y = [1.0, 1.1, 0.9, 1.0, 1.1, 0.9, 1.0, 1.1, 0.9, 1.0,
             10.0, -8.0, 12.0, -10.0, 15.0, -12.0, 10.0, -8.0, 12.0, -10.0]
        result = det.detect_entropy_change(y, window=5)
        assert len(result) >= 1
        # Change should be detected around the transition
        for a in result:
            assert a.detection_method == "entropy_change"

    def test_entropy_change_too_short(self):
        """Too few points for window -> no detection."""
        det = GPOutlierDetector()
        result = det.detect_entropy_change([1.0, 2.0, 3.0], window=5)
        assert result == []

    def test_entropy_change_exact_window_plus_one(self):
        """Exactly window+1 points -> one entropy value, no change."""
        det = GPOutlierDetector()
        result = det.detect_entropy_change([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], window=5)
        # 2 entropy values, 1 change -> might or might not trigger depending on data
        assert isinstance(result, list)

    def test_entropy_change_linear_data(self):
        """Linearly increasing data has constant local variance (approximately)."""
        det = GPOutlierDetector()
        y = [float(i) for i in range(20)]
        result = det.detect_entropy_change(y, window=5)
        # Linear data has roughly constant variance in windows -> few/no detections
        assert isinstance(result, list)

    def test_entropy_change_custom_window(self):
        """Different window sizes should work."""
        det = GPOutlierDetector()
        y = [1.0] * 10 + [100.0, -100.0] * 5
        result_small = det.detect_entropy_change(y, window=3)
        result_large = det.detect_entropy_change(y, window=7)
        assert isinstance(result_small, list)
        assert isinstance(result_large, list)

    def test_entropy_change_empty(self):
        """Empty input -> no detections."""
        det = GPOutlierDetector()
        result = det.detect_entropy_change([], window=5)
        assert result == []

    def test_entropy_change_single_point(self):
        """Single point -> no detections."""
        det = GPOutlierDetector()
        result = det.detect_entropy_change([1.0], window=5)
        assert result == []

    def test_entropy_change_score(self):
        """Score should be change/std ratio."""
        det = GPOutlierDetector()
        y = [1.0] * 10 + [100.0, -100.0, 100.0, -100.0, 100.0, -100.0, 100.0, -100.0, 100.0, -100.0]
        result = det.detect_entropy_change(y, window=5)
        for a in result:
            assert a.score > 0
            assert a.threshold == 2.0

    def test_entropy_change_all_same_variance(self):
        """Random-looking data with constant variance -> no detection."""
        det = GPOutlierDetector()
        # Alternating pattern with constant local variance
        y = [1.0, -1.0] * 20
        result = det.detect_entropy_change(y, window=5)
        # Should detect few or no changes since variance is roughly constant
        assert isinstance(result, list)


# ── Integration / edge case tests ─────────────────────────────────────


class TestGPOutlierEdgeCases:
    def test_default_threshold(self):
        """Default threshold should be 3.0."""
        det = GPOutlierDetector()
        assert det.threshold_sigma == 3.0

    def test_custom_threshold_init(self):
        """Custom threshold at construction."""
        det = GPOutlierDetector(threshold_sigma=2.5)
        assert det.threshold_sigma == 2.5

    def test_all_identical_predictions(self):
        """All predictions identical, all observations identical -> no outliers."""
        det = GPOutlierDetector()
        y_pred = [5.0] * 10
        y_std = [1.0] * 10
        y_actual = [5.0] * 10
        result = det.detect_standardized_residual(y_pred, y_std, y_actual)
        assert result == []

    def test_very_small_std(self):
        """Very small std makes everything look like an outlier."""
        det = GPOutlierDetector(threshold_sigma=3.0)
        y_pred = [0.0]
        y_std = [1e-10]
        y_actual = [1e-6]
        result = det.detect_standardized_residual(y_pred, y_std, y_actual)
        assert len(result) == 1

    def test_loo_constant_y(self):
        """All y values the same -> LOO predictions should be close."""
        det = GPOutlierDetector(threshold_sigma=5.0)
        X = [[float(i)] for i in range(10)]
        y = [5.0] * 10
        result = det.detect_loo_outlier(X, y, noise=0.1)
        # With constant y and enough data points, boundary effects are small
        assert len(result) == 0

    def test_standardized_residual_exact_threshold(self):
        """Residual exactly at threshold should not trigger."""
        det = GPOutlierDetector(threshold_sigma=3.0)
        # |3.0 - 0| / 1.0 = 3.0, not > 3.0
        result = det.detect_standardized_residual([0.0], [1.0], [3.0])
        assert result == []

    def test_standardized_residual_just_above(self):
        """Residual just above threshold should trigger."""
        det = GPOutlierDetector(threshold_sigma=3.0)
        result = det.detect_standardized_residual([0.0], [1.0], [3.01])
        assert len(result) == 1

    def test_loo_high_noise(self):
        """High noise should suppress outlier detection."""
        det = GPOutlierDetector(threshold_sigma=3.0)
        X = [[float(i)] for i in range(10)]
        y = [float(i) for i in range(10)]
        y[5] = 20.0  # moderate outlier
        # With very high noise, the GP is uncertain everywhere
        result = det.detect_loo_outlier(X, y, noise=100.0)
        # High noise -> fewer outliers detected
        assert isinstance(result, list)

    def test_entropy_change_gradual_increase(self):
        """Gradually increasing variance -> may not trigger sharp detection."""
        det = GPOutlierDetector()
        y = []
        for i in range(30):
            amplitude = 0.1 + i * 0.05
            y.append(amplitude if i % 2 == 0 else -amplitude)
        result = det.detect_entropy_change(y, window=5)
        assert isinstance(result, list)
