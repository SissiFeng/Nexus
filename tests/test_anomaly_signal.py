"""Tests for Layer 1: SignalChecker (raw signal-level anomaly checks)."""

from __future__ import annotations

import math

from optimization_copilot.anomaly.signal_checks import SignalAnomaly, SignalChecker


# ── Helpers ────────────────────────────────────────────────────────────


def _make_checker() -> SignalChecker:
    return SignalChecker()


# ── SignalAnomaly dataclass tests ──────────────────────────────────────


class TestSignalAnomalyDataclass:
    def test_creation(self):
        a = SignalAnomaly(
            check_name="test",
            severity="warning",
            message="test message",
            affected_indices=[0, 1],
        )
        assert a.check_name == "test"
        assert a.severity == "warning"
        assert a.message == "test message"
        assert a.affected_indices == [0, 1]
        assert a.metadata == {}

    def test_creation_with_metadata(self):
        a = SignalAnomaly(
            check_name="test",
            severity="error",
            message="msg",
            affected_indices=[],
            metadata={"key": "value"},
        )
        assert a.metadata == {"key": "value"}

    def test_severity_warning(self):
        a = SignalAnomaly("c", "warning", "m", [])
        assert a.severity == "warning"

    def test_severity_error(self):
        a = SignalAnomaly("c", "error", "m", [])
        assert a.severity == "error"


# ── EIS consistency tests ──────────────────────────────────────────────


class TestEISConsistency:
    def test_eis_consistency_pass_monotonic(self):
        """Monotonically increasing magnitudes -> no anomaly."""
        z_real = [10.0, 20.0, 30.0, 40.0, 50.0]
        z_imag = [5.0, 10.0, 15.0, 20.0, 25.0]
        result = SignalChecker.check_eis_consistency(z_real, z_imag)
        assert result is None

    def test_eis_consistency_fail_non_monotonic(self):
        """Heavily non-monotonic -> anomaly detected."""
        # Magnitudes: 10, 5, 3, 2, 1 -> all decreasing = all violations
        z_real = [10.0, 5.0, 3.0, 2.0, 1.0]
        z_imag = [0.0, 0.0, 0.0, 0.0, 0.0]
        result = SignalChecker.check_eis_consistency(z_real, z_imag)
        assert result is not None
        assert result.check_name == "eis_consistency"
        assert result.severity == "warning"
        assert len(result.affected_indices) > 0

    def test_eis_consistency_borderline_20_percent(self):
        """Exactly 20% violations should NOT trigger (> 20% required)."""
        # 10 points = 9 pairs; 20% = 1.8 pairs; we need > 1.8 = 2 violations
        # With 1 violation out of 9 = 11% -> no anomaly
        magnitudes = [1, 2, 3, 4, 3.5, 5, 6, 7, 8, 9]  # violation at index 4
        z_real = list(magnitudes)
        z_imag = [0.0] * 10
        result = SignalChecker.check_eis_consistency(z_real, z_imag)
        # 1/9 = 11% < 20% -> None
        assert result is None

    def test_eis_consistency_just_above_20_percent(self):
        """Just above 20% violations -> anomaly."""
        # 5 points = 4 pairs; need > 0.8 = 1 violation
        # magnitudes: 10, 5, 20, 3, 40 -> violations at indices 1, 3
        z_real = [10.0, 5.0, 20.0, 3.0, 40.0]
        z_imag = [0.0] * 5
        result = SignalChecker.check_eis_consistency(z_real, z_imag)
        # 2/4 = 50% > 20% -> anomaly
        assert result is not None

    def test_eis_consistency_single_point(self):
        """Single point -> no anomaly (not enough data)."""
        result = SignalChecker.check_eis_consistency([1.0], [1.0])
        assert result is None

    def test_eis_consistency_two_points_increasing(self):
        """Two points, increasing -> no anomaly."""
        result = SignalChecker.check_eis_consistency([1.0, 2.0], [0.0, 0.0])
        assert result is None

    def test_eis_consistency_two_points_decreasing(self):
        """Two points, decreasing -> 100% violations -> anomaly."""
        result = SignalChecker.check_eis_consistency([10.0, 1.0], [0.0, 0.0])
        assert result is not None

    def test_eis_consistency_empty(self):
        """Empty arrays -> no anomaly."""
        result = SignalChecker.check_eis_consistency([], [])
        assert result is None

    def test_eis_consistency_with_imaginary(self):
        """Complex impedances with proper imaginary parts."""
        # mag = sqrt(r^2 + i^2): 1.41, 2.24, 3.16, 4.12, 5.10 -> monotonic
        z_real = [1.0, 2.0, 3.0, 4.0, 5.0]
        z_imag = [-1.0, -1.0, -1.0, -1.0, -1.0]
        result = SignalChecker.check_eis_consistency(z_real, z_imag)
        assert result is None

    def test_eis_consistency_metadata(self):
        """Check that metadata contains violation fraction."""
        z_real = [10.0, 5.0, 3.0, 2.0, 1.0]
        z_imag = [0.0] * 5
        result = SignalChecker.check_eis_consistency(z_real, z_imag)
        assert result is not None
        assert "violation_fraction" in result.metadata


# ── Voltage spike tests ────────────────────────────────────────────────


class TestVoltageSpikeDetection:
    def test_voltage_spike_detection(self):
        """Large spike should be detected."""
        voltages = [1.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        result = SignalChecker.check_voltage_spike(voltages)
        assert result is not None
        assert result.check_name == "voltage_spike"
        assert result.severity == "error"
        assert 4 in result.affected_indices

    def test_voltage_no_spike(self):
        """Smooth data should not trigger."""
        voltages = [1.0, 1.1, 1.0, 0.9, 1.0, 1.1, 1.0, 0.9, 1.0, 1.1]
        result = SignalChecker.check_voltage_spike(voltages)
        assert result is None

    def test_voltage_spike_custom_threshold(self):
        """Lower threshold should be more sensitive."""
        voltages = [1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # With threshold_sigma=4, may not trigger
        result_high = SignalChecker.check_voltage_spike(voltages, threshold_sigma=10.0)
        # With threshold_sigma=1.5, should trigger
        result_low = SignalChecker.check_voltage_spike(voltages, threshold_sigma=1.5)
        assert result_low is not None

    def test_voltage_spike_empty(self):
        """Empty voltages -> no anomaly."""
        result = SignalChecker.check_voltage_spike([])
        assert result is None

    def test_voltage_spike_single_value(self):
        """Single value -> no anomaly."""
        result = SignalChecker.check_voltage_spike([1.0])
        assert result is None

    def test_voltage_spike_constant(self):
        """Constant signal -> no spike (std=0, skip division)."""
        result = SignalChecker.check_voltage_spike([5.0] * 20)
        assert result is None

    def test_voltage_spike_two_values(self):
        """Two values -> not enough data (need >= 3)."""
        result = SignalChecker.check_voltage_spike([0.0, 1000.0])
        assert result is None

    def test_voltage_multiple_spikes(self):
        """Multiple spikes should all be detected."""
        voltages = [1.0] * 5 + [100.0] + [1.0] * 5 + [100.0] + [1.0] * 5
        result = SignalChecker.check_voltage_spike(voltages)
        assert result is not None
        assert len(result.affected_indices) >= 2

    def test_voltage_spike_metadata(self):
        """Metadata should contain threshold_sigma."""
        voltages = [1.0] * 10 + [1000.0] + [1.0] * 10
        result = SignalChecker.check_voltage_spike(voltages)
        assert result is not None
        assert result.metadata["threshold_sigma"] == 4.0


# ── Negative absorbance tests ─────────────────────────────────────────


class TestNegativeAbsorbance:
    def test_negative_absorbance_detected(self):
        """More than 5% negative values -> anomaly."""
        absorbance = [-0.1, -0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        # 2/10 = 20% negative
        result = SignalChecker.check_negative_absorbance(absorbance)
        assert result is not None
        assert result.check_name == "negative_absorbance"
        assert result.severity == "warning"
        assert 0 in result.affected_indices
        assert 1 in result.affected_indices

    def test_negative_absorbance_clean(self):
        """All positive -> no anomaly."""
        absorbance = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = SignalChecker.check_negative_absorbance(absorbance)
        assert result is None

    def test_negative_absorbance_borderline(self):
        """Exactly 5% negative -> no anomaly (need >5%)."""
        # 1/20 = 5% exactly -> should NOT trigger
        absorbance = [-0.1] + [1.0] * 19
        result = SignalChecker.check_negative_absorbance(absorbance)
        assert result is None

    def test_negative_absorbance_all_negative(self):
        """All negative -> anomaly."""
        result = SignalChecker.check_negative_absorbance([-1.0, -2.0, -3.0])
        assert result is not None
        assert len(result.affected_indices) == 3

    def test_negative_absorbance_empty(self):
        """Empty list -> no anomaly."""
        result = SignalChecker.check_negative_absorbance([])
        assert result is None

    def test_negative_absorbance_single_negative(self):
        """Single negative value (100%) -> anomaly."""
        result = SignalChecker.check_negative_absorbance([-0.5])
        assert result is not None

    def test_negative_absorbance_single_positive(self):
        """Single positive value -> no anomaly."""
        result = SignalChecker.check_negative_absorbance([0.5])
        assert result is None

    def test_negative_absorbance_zeros(self):
        """All zeros -> no anomaly (zeros are not negative)."""
        result = SignalChecker.check_negative_absorbance([0.0, 0.0, 0.0])
        assert result is None

    def test_negative_absorbance_metadata(self):
        """Metadata should contain negative_fraction."""
        result = SignalChecker.check_negative_absorbance([-1.0, -2.0, 3.0])
        assert result is not None
        assert "negative_fraction" in result.metadata
        assert abs(result.metadata["negative_fraction"] - 2.0 / 3.0) < 1e-9


# ── XRD saturation tests ──────────────────────────────────────────────


class TestXRDSaturation:
    def test_xrd_saturation_detected(self):
        """Value near max_counts -> anomaly."""
        intensities = [100.0, 200.0, 65000.0, 300.0]
        result = SignalChecker.check_xrd_peak_saturation(intensities)
        assert result is not None
        assert result.check_name == "xrd_peak_saturation"
        assert result.severity == "error"
        assert 2 in result.affected_indices

    def test_xrd_saturation_clean(self):
        """All values well below max -> no anomaly."""
        intensities = [100.0, 200.0, 300.0, 400.0]
        result = SignalChecker.check_xrd_peak_saturation(intensities)
        assert result is None

    def test_xrd_saturation_at_exact_threshold(self):
        """Value at exactly 99% of max -> saturated."""
        max_counts = 65535.0
        threshold = max_counts * 0.99
        intensities = [threshold]
        result = SignalChecker.check_xrd_peak_saturation(intensities)
        assert result is not None

    def test_xrd_saturation_just_below_threshold(self):
        """Value just below 99% -> no anomaly."""
        max_counts = 65535.0
        threshold = max_counts * 0.99
        intensities = [threshold - 1.0]
        result = SignalChecker.check_xrd_peak_saturation(intensities)
        assert result is None

    def test_xrd_saturation_custom_max(self):
        """Custom max_counts should work."""
        result = SignalChecker.check_xrd_peak_saturation([990.0], max_counts=1000.0)
        assert result is not None

    def test_xrd_saturation_empty(self):
        """Empty list -> no anomaly."""
        result = SignalChecker.check_xrd_peak_saturation([])
        assert result is None

    def test_xrd_saturation_at_max(self):
        """Value equal to max_counts -> saturated."""
        result = SignalChecker.check_xrd_peak_saturation([65535.0])
        assert result is not None

    def test_xrd_saturation_multiple(self):
        """Multiple saturated values -> all reported."""
        intensities = [65000.0, 100.0, 65535.0, 200.0, 65100.0]
        result = SignalChecker.check_xrd_peak_saturation(intensities)
        assert result is not None
        assert len(result.affected_indices) == 3

    def test_xrd_saturation_metadata(self):
        """Metadata should include max_counts and threshold."""
        result = SignalChecker.check_xrd_peak_saturation([65535.0])
        assert result is not None
        assert result.metadata["max_counts"] == 65535.0
        assert "threshold" in result.metadata


# ── check_all tests ────────────────────────────────────────────────────


class TestCheckAll:
    def test_check_all_empty_data(self):
        """Empty dict -> no anomalies."""
        checker = _make_checker()
        result = checker.check_all({})
        assert result == []

    def test_check_all_with_eis_data_clean(self):
        """Clean EIS data -> no anomalies."""
        checker = _make_checker()
        result = checker.check_all({
            "z_real": [1.0, 2.0, 3.0, 4.0, 5.0],
            "z_imag": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        assert result == []

    def test_check_all_with_eis_data_bad(self):
        """Bad EIS data -> anomaly."""
        checker = _make_checker()
        result = checker.check_all({
            "z_real": [50.0, 10.0, 5.0, 2.0, 1.0],
            "z_imag": [0.0, 0.0, 0.0, 0.0, 0.0],
        })
        assert len(result) == 1
        assert result[0].check_name == "eis_consistency"

    def test_check_all_with_voltage_data(self):
        """Voltage spike -> anomaly."""
        checker = _make_checker()
        result = checker.check_all({
            "voltages": [1.0] * 10 + [1000.0] + [1.0] * 10,
        })
        assert len(result) >= 1
        assert any(a.check_name == "voltage_spike" for a in result)

    def test_check_all_with_absorbance(self):
        """Negative absorbance -> anomaly."""
        checker = _make_checker()
        result = checker.check_all({
            "absorbance": [-1.0, -2.0, 0.5],
        })
        assert len(result) == 1
        assert result[0].check_name == "negative_absorbance"

    def test_check_all_with_xrd(self):
        """XRD saturation -> anomaly."""
        checker = _make_checker()
        result = checker.check_all({
            "xrd_intensities": [65535.0, 100.0],
        })
        assert len(result) == 1
        assert result[0].check_name == "xrd_peak_saturation"

    def test_check_all_multiple_issues(self):
        """Multiple data types with issues -> multiple anomalies."""
        checker = _make_checker()
        result = checker.check_all({
            "absorbance": [-1.0, -2.0, 0.5],
            "xrd_intensities": [65535.0],
        })
        assert len(result) == 2
        names = {a.check_name for a in result}
        assert "negative_absorbance" in names
        assert "xrd_peak_saturation" in names

    def test_check_all_irrelevant_keys_ignored(self):
        """Keys that don't match any check -> no anomalies."""
        checker = _make_checker()
        result = checker.check_all({"temperature": [25.0, 26.0]})
        assert result == []

    def test_check_all_partial_eis_keys(self):
        """Only z_real without z_imag -> EIS check skipped."""
        checker = _make_checker()
        result = checker.check_all({"z_real": [1.0, 2.0]})
        assert result == []


# ── Edge case tests ────────────────────────────────────────────────────


class TestEdgeCases:
    def test_very_large_arrays(self):
        """Large array should not crash."""
        checker = _make_checker()
        voltages = [1.0] * 10000
        result = checker.check_all({"voltages": voltages})
        assert result == []

    def test_nan_in_voltage(self):
        """NaN values should not crash (but may not detect as spikes)."""
        # NaN comparisons return False, so they won't be flagged as spikes
        voltages = [1.0, 1.0, float("nan"), 1.0, 1.0]
        result = SignalChecker.check_voltage_spike(voltages)
        # Should not crash; result may vary
        assert result is None or isinstance(result, SignalAnomaly)

    def test_inf_in_absorbance(self):
        """Inf values should be handled."""
        absorbance = [0.5, float("inf"), 0.5]
        # inf is not negative -> no anomaly
        result = SignalChecker.check_negative_absorbance(absorbance)
        assert result is None

    def test_negative_inf_absorbance(self):
        """-Inf should count as negative."""
        absorbance = [float("-inf"), 0.5, 0.5]
        # 1/3 = 33% > 5%
        result = SignalChecker.check_negative_absorbance(absorbance)
        assert result is not None

    def test_all_same_eis_magnitudes(self):
        """All same magnitudes -> no violations (not decreasing)."""
        z_real = [5.0, 5.0, 5.0, 5.0, 5.0]
        z_imag = [0.0, 0.0, 0.0, 0.0, 0.0]
        result = SignalChecker.check_eis_consistency(z_real, z_imag)
        assert result is None

    def test_mismatched_eis_lengths(self):
        """z_real and z_imag of different lengths -> uses min."""
        z_real = [1.0, 2.0, 3.0]
        z_imag = [0.0, 0.0]
        # Uses min(3, 2) = 2 points -> 1 pair -> no anomaly possible if monotonic
        result = SignalChecker.check_eis_consistency(z_real, z_imag)
        assert result is None
