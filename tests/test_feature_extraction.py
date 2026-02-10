"""Tests for feature_extraction.extractors module."""

import math

import pytest

from optimization_copilot.feature_extraction.extractors import (
    BasicCurveExtractor,
    CurveData,
    ExtractedFeatures,
    FeatureExtractor,
    FeatureExtractorRegistry,
    ThresholdExtractor,
    curve_stability_signal,
)


# ── Helpers ───────────────────────────────────────────────


def _simple_curve() -> CurveData:
    """A monotonically increasing curve: y = 2*x for x in [0..4]."""
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    ys = [0.0, 2.0, 4.0, 6.0, 8.0]
    return CurveData(x_values=xs, y_values=ys, metadata={"type": "test"})


def _peaked_curve() -> CurveData:
    """A curve with a peak in the middle."""
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    ys = [1.0, 3.0, 5.0, 3.0, 1.0]
    return CurveData(x_values=xs, y_values=ys, metadata={"type": "peaked"})


def _empty_curve() -> CurveData:
    """An empty curve with no data points."""
    return CurveData(x_values=[], y_values=[], metadata={"type": "empty"})


# ── CurveData Tests ───────────────────────────────────────


class TestCurveData:
    def test_creation(self):
        c = _simple_curve()
        assert len(c.x_values) == 5
        assert len(c.y_values) == 5
        assert c.metadata["type"] == "test"

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="x_values length"):
            CurveData(x_values=[1.0, 2.0], y_values=[1.0])

    def test_empty_curve(self):
        c = _empty_curve()
        assert len(c.x_values) == 0
        assert len(c.y_values) == 0

    def test_default_metadata(self):
        c = CurveData(x_values=[1.0], y_values=[2.0])
        assert c.metadata == {}


# ── BasicCurveExtractor Tests ─────────────────────────────


class TestBasicCurveExtractor:
    def setup_method(self):
        self.extractor = BasicCurveExtractor()

    def test_name_and_version(self):
        assert self.extractor.name() == "basic_curve"
        assert self.extractor.version() == "1.0.0"

    def test_feature_names(self):
        names = self.extractor.feature_names()
        assert "peak_value" in names
        assert "peak_position" in names
        assert "area_under_curve" in names
        assert "mean_slope" in names
        assert "start_value" in names
        assert "end_value" in names
        assert "range" in names
        assert "std_dev" in names
        assert len(names) == 8

    def test_simple_curve(self):
        curve = _simple_curve()
        result = self.extractor.extract(curve)

        assert isinstance(result, ExtractedFeatures)
        assert result.extractor_name == "basic_curve"
        assert result.extractor_version == "1.0.0"

        f = result.features
        assert f["peak_value"] == 8.0
        assert f["peak_position"] == 4.0
        assert f["start_value"] == 0.0
        assert f["end_value"] == 8.0
        assert f["range"] == 8.0
        # mean slope: y=2x -> slope is always 2.0
        assert abs(f["mean_slope"] - 2.0) < 1e-9

    def test_peaked_curve(self):
        curve = _peaked_curve()
        result = self.extractor.extract(curve)
        f = result.features

        assert f["peak_value"] == 5.0
        assert f["peak_position"] == 2.0
        assert f["start_value"] == 1.0
        assert f["end_value"] == 1.0
        assert f["range"] == 4.0  # max=5, min=1

    def test_area_under_curve(self):
        # y = 2*x from 0 to 4: area = 0.5 * 4 * 8 = 16.0 (triangle)
        curve = _simple_curve()
        result = self.extractor.extract(curve)
        assert abs(result.features["area_under_curve"] - 16.0) < 1e-9

    def test_std_dev(self):
        # y = [0, 2, 4, 6, 8], mean = 4.0
        # variance = (16+4+0+4+16)/5 = 8.0, std = sqrt(8) ~ 2.828
        curve = _simple_curve()
        result = self.extractor.extract(curve)
        expected_std = math.sqrt(8.0)
        assert abs(result.features["std_dev"] - expected_std) < 1e-9

    def test_single_point_curve(self):
        curve = CurveData(x_values=[3.0], y_values=[7.0])
        result = self.extractor.extract(curve)
        f = result.features
        assert f["peak_value"] == 7.0
        assert f["peak_position"] == 3.0
        assert f["area_under_curve"] == 0.0
        assert f["mean_slope"] == 0.0
        assert f["start_value"] == 7.0
        assert f["end_value"] == 7.0
        assert f["range"] == 0.0
        assert f["std_dev"] == 0.0

    def test_empty_curve(self):
        curve = _empty_curve()
        result = self.extractor.extract(curve)
        # All features should be 0.0 for empty curves
        for name in self.extractor.feature_names():
            assert result.features[name] == 0.0


# ── ThresholdExtractor Tests ─────────────────────────────


class TestThresholdExtractor:
    def test_name_and_version(self):
        ext = ThresholdExtractor(threshold=5.0)
        assert ext.name() == "threshold"
        assert ext.version() == "1.0.0"

    def test_feature_names(self):
        ext = ThresholdExtractor()
        names = ext.feature_names()
        assert "time_to_threshold" in names
        assert "fraction_above_threshold" in names
        assert len(names) == 2

    def test_threshold_crossed(self):
        # y = [0, 2, 4, 6, 8], threshold = 5.0
        # First crossing at x=3 (y=6 >= 5)
        # Points above: y=6, y=8 -> 2/5 = 0.4
        ext = ThresholdExtractor(threshold=5.0)
        result = ext.extract(_simple_curve())
        f = result.features
        assert f["time_to_threshold"] == 3.0
        assert abs(f["fraction_above_threshold"] - 0.4) < 1e-9

    def test_threshold_never_reached(self):
        ext = ThresholdExtractor(threshold=100.0)
        result = ext.extract(_simple_curve())
        f = result.features
        assert f["time_to_threshold"] == float("inf")
        assert f["fraction_above_threshold"] == 0.0

    def test_threshold_at_start(self):
        ext = ThresholdExtractor(threshold=0.0)
        result = ext.extract(_simple_curve())
        f = result.features
        assert f["time_to_threshold"] == 0.0  # first point is at x=0, y=0 >= 0
        assert f["fraction_above_threshold"] == 1.0  # all >= 0

    def test_threshold_exact_match(self):
        # y = [1, 3, 5, 3, 1], threshold = 5.0
        ext = ThresholdExtractor(threshold=5.0)
        result = ext.extract(_peaked_curve())
        f = result.features
        assert f["time_to_threshold"] == 2.0
        assert abs(f["fraction_above_threshold"] - 0.2) < 1e-9  # 1/5

    def test_empty_curve(self):
        ext = ThresholdExtractor(threshold=1.0)
        result = ext.extract(_empty_curve())
        f = result.features
        assert f["time_to_threshold"] == float("inf")
        assert f["fraction_above_threshold"] == 0.0


# ── Registry Tests ────────────────────────────────────────


class TestFeatureExtractorRegistry:
    def test_register_and_get(self):
        reg = FeatureExtractorRegistry()
        reg.register(BasicCurveExtractor)
        ext = reg.get("basic_curve")
        assert isinstance(ext, BasicCurveExtractor)

    def test_register_with_kwargs(self):
        reg = FeatureExtractorRegistry()
        reg.register(ThresholdExtractor, threshold=3.0)
        ext = reg.get("threshold")
        assert isinstance(ext, ThresholdExtractor)

    def test_list_extractors(self):
        reg = FeatureExtractorRegistry()
        reg.register(BasicCurveExtractor)
        reg.register(ThresholdExtractor, threshold=1.0)
        names = reg.list_extractors()
        assert names == ["basic_curve", "threshold"]

    def test_list_empty_registry(self):
        reg = FeatureExtractorRegistry()
        assert reg.list_extractors() == []

    def test_get_unknown_raises(self):
        reg = FeatureExtractorRegistry()
        with pytest.raises(KeyError, match="Unknown extractor"):
            reg.get("nonexistent")

    def test_duplicate_register_raises(self):
        reg = FeatureExtractorRegistry()
        reg.register(BasicCurveExtractor)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(BasicCurveExtractor)

    def test_register_non_class_raises(self):
        reg = FeatureExtractorRegistry()
        with pytest.raises(TypeError, match="Expected a subclass"):
            reg.register("not_a_class")  # type: ignore[arg-type]

    def test_register_wrong_class_raises(self):
        reg = FeatureExtractorRegistry()
        with pytest.raises(TypeError, match="Expected a subclass"):
            reg.register(str)  # type: ignore[arg-type]


# ── extract_all Tests ─────────────────────────────────────


class TestExtractAll:
    def test_extract_all_multiple(self):
        reg = FeatureExtractorRegistry()
        reg.register(BasicCurveExtractor)
        reg.register(ThresholdExtractor, threshold=5.0)

        curve = _simple_curve()
        results = reg.extract_all(curve)

        assert "basic_curve" in results
        assert "threshold" in results
        assert results["basic_curve"].extractor_name == "basic_curve"
        assert results["threshold"].extractor_name == "threshold"

    def test_extract_all_empty_registry(self):
        reg = FeatureExtractorRegistry()
        results = reg.extract_all(_simple_curve())
        assert results == {}

    def test_extract_all_results_valid(self):
        reg = FeatureExtractorRegistry()
        reg.register(BasicCurveExtractor)

        curve = _peaked_curve()
        results = reg.extract_all(curve)
        basic = results["basic_curve"]

        assert basic.features["peak_value"] == 5.0
        assert len(basic.feature_hash) == 16


# ── Feature Hash Tests ────────────────────────────────────


class TestFeatureHash:
    def test_hash_is_deterministic(self):
        ext = BasicCurveExtractor()
        curve = _simple_curve()
        r1 = ext.extract(curve)
        r2 = ext.extract(curve)
        assert r1.feature_hash == r2.feature_hash

    def test_hash_length(self):
        ext = BasicCurveExtractor()
        result = ext.extract(_simple_curve())
        assert len(result.feature_hash) == 16

    def test_different_curves_different_hash(self):
        ext = BasicCurveExtractor()
        h1 = ext.extract(_simple_curve()).feature_hash
        h2 = ext.extract(_peaked_curve()).feature_hash
        assert h1 != h2

    def test_hash_same_across_extractors(self):
        """Same curve should produce same hash regardless of extractor."""
        curve = _simple_curve()
        h_basic = BasicCurveExtractor().extract(curve).feature_hash
        h_threshold = ThresholdExtractor(threshold=1.0).extract(curve).feature_hash
        assert h_basic == h_threshold

    def test_metadata_affects_hash(self):
        """Different metadata should produce different hashes."""
        c1 = CurveData(x_values=[1.0], y_values=[2.0], metadata={"type": "a"})
        c2 = CurveData(x_values=[1.0], y_values=[2.0], metadata={"type": "b"})
        ext = BasicCurveExtractor()
        assert ext.extract(c1).feature_hash != ext.extract(c2).feature_hash


# ── Curve Stability Signal Tests ──────────────────────────


class TestCurveStabilitySignal:
    def test_identical_curves_zero_cv(self):
        """Identical curves should have zero CV (no drift)."""
        curve = _simple_curve()
        ext = BasicCurveExtractor()
        signals = curve_stability_signal([curve, curve, curve], ext)

        for fname, cv in signals.items():
            assert cv == 0.0, f"Expected 0.0 CV for {fname}, got {cv}"

    def test_varying_curves_nonzero_cv(self):
        """Curves with variation should have non-zero CV for varying features."""
        c1 = CurveData(x_values=[0.0, 1.0], y_values=[1.0, 2.0])
        c2 = CurveData(x_values=[0.0, 1.0], y_values=[1.0, 4.0])
        c3 = CurveData(x_values=[0.0, 1.0], y_values=[1.0, 6.0])

        ext = BasicCurveExtractor()
        signals = curve_stability_signal([c1, c2, c3], ext)

        # peak_value varies across curves (2, 4, 6) -> CV should be > 0
        assert signals["peak_value"] > 0.0
        # start_value is always 1.0 -> CV should be 0.0
        assert signals["start_value"] == 0.0

    def test_empty_curve_list(self):
        """Empty list of curves should return 0.0 for all features."""
        ext = BasicCurveExtractor()
        signals = curve_stability_signal([], ext)
        assert all(v == 0.0 for v in signals.values())
        assert set(signals.keys()) == set(ext.feature_names())

    def test_single_curve_zero_cv(self):
        """A single curve has no variability -> zero CV."""
        ext = BasicCurveExtractor()
        signals = curve_stability_signal([_simple_curve()], ext)
        assert all(v == 0.0 for v in signals.values())

    def test_with_threshold_extractor(self):
        """Stability signal works with any FeatureExtractor."""
        c1 = CurveData(x_values=[0.0, 1.0, 2.0], y_values=[0.0, 3.0, 6.0])
        c2 = CurveData(x_values=[0.0, 1.0, 2.0], y_values=[0.0, 1.0, 2.0])

        ext = ThresholdExtractor(threshold=2.0)
        signals = curve_stability_signal([c1, c2], ext)

        # Both features should be present
        assert "time_to_threshold" in signals
        assert "fraction_above_threshold" in signals

    def test_cv_calculation_accuracy(self):
        """Verify the CV calculation against known values."""
        # peak values: 10, 20, 30 -> mean=20, std=sqrt(200/3)~8.165
        # CV = std/mean ~ 0.4082
        c1 = CurveData(x_values=[0.0], y_values=[10.0])
        c2 = CurveData(x_values=[0.0], y_values=[20.0])
        c3 = CurveData(x_values=[0.0], y_values=[30.0])

        ext = BasicCurveExtractor()
        signals = curve_stability_signal([c1, c2, c3], ext)

        mean = 20.0
        std = math.sqrt(((10 - 20) ** 2 + (20 - 20) ** 2 + (30 - 20) ** 2) / 3)
        expected_cv = std / abs(mean)
        assert abs(signals["peak_value"] - expected_cv) < 1e-9
