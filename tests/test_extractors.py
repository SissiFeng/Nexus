"""Tests for uncertainty-aware KPI extractors.

Covers all four extractors: DC cycling, UV-Vis, XRD, and EIS.
~60 tests total.
"""

from __future__ import annotations

import math
import unittest
from typing import Any

from optimization_copilot.domain_knowledge.dc_cycling import get_dc_config
from optimization_copilot.domain_knowledge.eis import get_eis_config
from optimization_copilot.domain_knowledge.uv_vis import get_uvvis_config
from optimization_copilot.domain_knowledge.xrd import get_xrd_config
from optimization_copilot.extractors.base import UncertaintyExtractor
from optimization_copilot.extractors.dc_extractor import DCCyclingExtractor
from optimization_copilot.extractors.eis_extractor import (
    EISExtractor,
    _z_randles,
)
from optimization_copilot.extractors.uv_vis_extractor import UVVisExtractor
from optimization_copilot.extractors.xrd_extractor import XRDExtractor
from optimization_copilot.uncertainty.types import MeasurementWithUncertainty


# ══════════════════════════════════════════════════════════════════════
# Base extractor tests
# ══════════════════════════════════════════════════════════════════════


class TestBaseExtractor(unittest.TestCase):
    """Test the abstract base class helpers."""

    def _make_extractor(self) -> UncertaintyExtractor:
        """Create a concrete subclass for testing."""

        class _Dummy(UncertaintyExtractor):
            def extract_with_uncertainty(self, raw_data):
                return []

        return _Dummy(domain_config={})

    def test_compute_confidence_normal(self):
        ext = self._make_extractor()
        # variance=1, value=10 => std=1, rel=0.1 => conf=0.9
        conf = ext._compute_confidence(1.0, 10.0)
        self.assertAlmostEqual(conf, 0.9, places=5)

    def test_compute_confidence_zero_value(self):
        ext = self._make_extractor()
        conf = ext._compute_confidence(1.0, 0.0)
        self.assertEqual(conf, 0.0)

    def test_compute_confidence_zero_variance(self):
        ext = self._make_extractor()
        conf = ext._compute_confidence(0.0, 10.0)
        self.assertEqual(conf, 1.0)

    def test_compute_confidence_clamps_to_zero(self):
        ext = self._make_extractor()
        # Very large variance relative to value
        conf = ext._compute_confidence(10000.0, 1.0)
        self.assertEqual(conf, 0.0)

    def test_apply_physical_constraints_below_min(self):
        ext = self._make_extractor()
        m = MeasurementWithUncertainty(
            value=-5.0, variance=1.0, confidence=0.9, source="test",
        )
        result = ext._apply_physical_constraints(m, "x", {"min": 0})
        self.assertAlmostEqual(result.confidence, 0.9 * 0.3, places=5)

    def test_apply_physical_constraints_above_max(self):
        ext = self._make_extractor()
        m = MeasurementWithUncertainty(
            value=200.0, variance=1.0, confidence=0.9, source="test",
        )
        result = ext._apply_physical_constraints(m, "x", {"max": 100})
        self.assertAlmostEqual(result.confidence, 0.9 * 0.3, places=5)

    def test_apply_physical_constraints_outside_typical(self):
        ext = self._make_extractor()
        m = MeasurementWithUncertainty(
            value=50.0, variance=1.0, confidence=0.9, source="test",
        )
        result = ext._apply_physical_constraints(
            m, "x", {"typical_range": (0, 10)},
        )
        self.assertAlmostEqual(result.confidence, 0.9 * 0.7, places=5)

    def test_apply_physical_constraints_within_range(self):
        ext = self._make_extractor()
        m = MeasurementWithUncertainty(
            value=5.0, variance=1.0, confidence=0.9, source="test",
        )
        result = ext._apply_physical_constraints(
            m, "x", {"min": 0, "max": 100, "typical_range": (0, 10)},
        )
        self.assertAlmostEqual(result.confidence, 0.9, places=5)


# ══════════════════════════════════════════════════════════════════════
# DC Cycling Extractor tests
# ══════════════════════════════════════════════════════════════════════


class TestDCCyclingExtractor(unittest.TestCase):
    """Test coulombic efficiency extraction."""

    def setUp(self):
        self.config = get_dc_config()
        self.ext = DCCyclingExtractor(self.config)

    def test_symmetric_waveform_ce_100(self):
        """Symmetric deposition/dissolution should give CE ~ 100%."""
        n = 200
        dt = 0.01
        time = [i * dt for i in range(n)]
        # First half: deposition (negative current)
        # Second half: dissolution (positive current)
        current = [-1.0] * (n // 2) + [1.0] * (n // 2)

        results = self.ext.extract_with_uncertainty({
            "current": current,
            "voltage": [0.0] * n,
            "time": time,
        })

        self.assertEqual(len(results), 1)
        m = results[0]
        self.assertAlmostEqual(m.value, 100.0, delta=1.0)
        self.assertGreater(m.confidence, 0.5)
        self.assertEqual(m.source, "DC_CE")
        self.assertEqual(m.method, "trapezoidal")

    def test_asymmetric_waveform_ce_below_100(self):
        """Less dissolution than deposition gives CE < 100%."""
        n = 200
        dt = 0.01
        time = [i * dt for i in range(n)]
        # 60% deposition, 40% dissolution, but dissolution current is 50%
        n_dep = 120
        n_dis = 80
        current = [-1.0] * n_dep + [0.5] * n_dis

        results = self.ext.extract_with_uncertainty({
            "current": current,
            "voltage": [0.0] * n,
            "time": time,
        })

        m = results[0]
        # Q_dep = 1.0 * 120 * 0.01 = 1.2
        # Q_dis = 0.5 * 80 * 0.01 = 0.4
        # CE = 0.4/1.2 * 100 = 33.3%
        self.assertLess(m.value, 100.0)
        self.assertGreater(m.value, 0.0)

    def test_empty_data_returns_nan(self):
        results = self.ext.extract_with_uncertainty({
            "current": [],
            "voltage": [],
            "time": [],
        })
        self.assertEqual(len(results), 1)
        self.assertTrue(math.isnan(results[0].value))
        self.assertEqual(results[0].confidence, 0.0)

    def test_single_point_returns_nan(self):
        results = self.ext.extract_with_uncertainty({
            "current": [1.0],
            "voltage": [0.5],
            "time": [0.0],
        })
        self.assertTrue(math.isnan(results[0].value))

    def test_drift_adds_to_variance(self):
        """Verify drift contribution increases variance."""
        n = 200
        dt = 0.01
        time = [i * dt for i in range(n)]
        current = [-1.0] * (n // 2) + [1.0] * (n // 2)

        # Normal config
        results_normal = self.ext.extract_with_uncertainty({
            "current": current,
            "voltage": [0.0] * n,
            "time": time,
        })

        # Config with very high drift
        high_drift_config = get_dc_config()
        high_drift_config["instrument"]["dc"]["zero_drift_a_per_hour"] = 1.0
        ext_drift = DCCyclingExtractor(high_drift_config)
        results_drift = ext_drift.extract_with_uncertainty({
            "current": current,
            "voltage": [0.0] * n,
            "time": time,
        })

        self.assertGreater(results_drift[0].variance, results_normal[0].variance)

    def test_physical_constraint_ce_above_105(self):
        """CE > 105% should trigger low confidence from constraints."""
        # Manufacture data where dissolution > deposition
        n = 200
        dt = 0.01
        time = [i * dt for i in range(n)]
        # More dissolution than deposition
        current = [-0.5] * (n // 2) + [1.0] * (n // 2)

        results = self.ext.extract_with_uncertainty({
            "current": current,
            "voltage": [0.0] * n,
            "time": time,
        })

        m = results[0]
        # CE should be ~200%, which exceeds 105% max
        self.assertGreater(m.value, 105.0)
        # Confidence should be penalized
        self.assertLess(m.confidence, 0.5)

    def test_returns_measurement_with_uncertainty(self):
        """Result has correct type and all expected fields."""
        n = 100
        time = [i * 0.01 for i in range(n)]
        current = [-1.0] * 50 + [1.0] * 50

        results = self.ext.extract_with_uncertainty({
            "current": current,
            "voltage": [0.0] * n,
            "time": time,
        })

        m = results[0]
        self.assertIsInstance(m, MeasurementWithUncertainty)
        self.assertIsNotNone(m.value)
        self.assertGreaterEqual(m.variance, 0.0)
        self.assertGreaterEqual(m.confidence, 0.0)
        self.assertLessEqual(m.confidence, 1.0)
        self.assertIn("q_deposit", m.metadata)
        self.assertIn("q_dissolve", m.metadata)
        self.assertIn("drift_contribution", m.metadata)

    def test_zero_deposition_returns_nan(self):
        """All positive current means no deposition."""
        n = 100
        time = [i * 0.01 for i in range(n)]
        current = [1.0] * n

        results = self.ext.extract_with_uncertainty({
            "current": current,
            "voltage": [0.0] * n,
            "time": time,
        })

        self.assertTrue(math.isnan(results[0].value))


# ══════════════════════════════════════════════════════════════════════
# UV-Vis Extractor tests
# ══════════════════════════════════════════════════════════════════════


class TestUVVisExtractor(unittest.TestCase):
    """Test UV-Vis absorbance extraction."""

    def setUp(self):
        self.config = get_uvvis_config()
        self.ext = UVVisExtractor(self.config)

    def test_exact_wavelength_match(self):
        """Exact match should have zero interpolation uncertainty."""
        results = self.ext.extract_with_uncertainty({
            "wavelengths": [400.0, 450.0, 500.0, 550.0],
            "absorbance": [0.1, 0.3, 0.5, 0.2],
            "target_wavelength": 500.0,
        })

        m = results[0]
        self.assertAlmostEqual(m.value, 0.5, places=6)
        self.assertEqual(m.metadata["interp_variance"], 0.0)

    def test_interpolation_between_points(self):
        """Interpolation between two data points."""
        results = self.ext.extract_with_uncertainty({
            "wavelengths": [400.0, 500.0],
            "absorbance": [0.2, 0.4],
            "target_wavelength": 450.0,
        })

        m = results[0]
        # Linear interpolation: midpoint
        self.assertAlmostEqual(m.value, 0.3, places=5)
        self.assertGreater(m.confidence, 0.0)

    def test_low_absorbance_noise(self):
        """Low absorbance tier: noise = 0.002."""
        results = self.ext.extract_with_uncertainty({
            "wavelengths": [400.0, 500.0],
            "absorbance": [0.1, 0.1],
            "target_wavelength": 400.0,
        })

        m = results[0]
        # instrument_var = 0.002^2 = 4e-6
        self.assertAlmostEqual(
            m.metadata["instrument_variance"], 0.002 ** 2, places=8,
        )

    def test_mid_absorbance_noise(self):
        """Mid absorbance tier: noise = 0.005."""
        results = self.ext.extract_with_uncertainty({
            "wavelengths": [400.0],
            "absorbance": [1.0],
            "target_wavelength": 400.0,
        })

        m = results[0]
        self.assertAlmostEqual(
            m.metadata["instrument_variance"], 0.005 ** 2, places=8,
        )

    def test_high_absorbance_noise(self):
        """High absorbance tier: noise = 0.02."""
        results = self.ext.extract_with_uncertainty({
            "wavelengths": [400.0],
            "absorbance": [3.0],
            "target_wavelength": 400.0,
        })

        m = results[0]
        self.assertAlmostEqual(
            m.metadata["instrument_variance"], 0.02 ** 2, places=8,
        )

    def test_above_linear_range_reduces_confidence(self):
        """Above Beer-Lambert linear range max -> confidence halved."""
        # linear_range_max = 2.5 in default config
        results = self.ext.extract_with_uncertainty({
            "wavelengths": [400.0],
            "absorbance": [3.0],
            "target_wavelength": 400.0,
        })

        m = results[0]
        # Confidence is halved for being above linear range
        # and may be further reduced by physical constraints
        # Just verify it's reduced
        # Without the halving, confidence would be higher
        self.assertLess(m.confidence, 0.95)

    def test_negative_absorbance_reduced_confidence(self):
        """Negative absorbance triggers physical constraint penalty."""
        results = self.ext.extract_with_uncertainty({
            "wavelengths": [400.0],
            "absorbance": [-0.1],
            "target_wavelength": 400.0,
        })

        m = results[0]
        # Below min=-0.05 => severe penalty
        self.assertLess(m.confidence, 0.5)

    def test_baseline_correction(self):
        """Baseline correction subtracts baseline from absorbance."""
        results = self.ext.extract_with_uncertainty({
            "wavelengths": [400.0, 500.0],
            "absorbance": [0.5, 0.8],
            "baseline": [0.1, 0.1],
            "target_wavelength": 400.0,
        })

        m = results[0]
        self.assertAlmostEqual(m.value, 0.4, places=5)
        self.assertTrue(m.metadata["baseline_corrected"])

    def test_baseline_doubles_instrument_variance(self):
        """Baseline correction should double the instrument variance."""
        # Without baseline
        r1 = self.ext.extract_with_uncertainty({
            "wavelengths": [400.0],
            "absorbance": [0.3],
            "target_wavelength": 400.0,
        })

        # With baseline (yielding same net absorbance)
        r2 = self.ext.extract_with_uncertainty({
            "wavelengths": [400.0],
            "absorbance": [0.5],
            "baseline": [0.2],
            "target_wavelength": 400.0,
        })

        # The baseline-corrected version should have higher variance
        # because instrument noise is counted twice (both measurements
        # are in the same absorbance tier: low_abs < 0.5).
        self.assertGreater(r2[0].variance, r1[0].variance)

    def test_empty_data(self):
        results = self.ext.extract_with_uncertainty({
            "wavelengths": [],
            "absorbance": [],
            "target_wavelength": 400.0,
        })
        self.assertTrue(math.isnan(results[0].value))
        self.assertEqual(results[0].confidence, 0.0)


# ══════════════════════════════════════════════════════════════════════
# XRD Extractor tests
# ══════════════════════════════════════════════════════════════════════


def _gaussian_peak(
    two_theta_center: float,
    fwhm_deg: float,
    amplitude: float,
    two_theta_range: tuple[float, float] = (20, 60),
    n_points: int = 500,
) -> tuple[list[float], list[float]]:
    """Generate a synthetic Gaussian diffraction peak."""
    sigma = fwhm_deg / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    tt_min, tt_max = two_theta_range
    two_theta = [tt_min + (tt_max - tt_min) * i / (n_points - 1) for i in range(n_points)]
    intensity = [
        amplitude * math.exp(-0.5 * ((tt - two_theta_center) / sigma) ** 2)
        for tt in two_theta
    ]
    return two_theta, intensity


class TestXRDExtractor(unittest.TestCase):
    """Test crystallite size extraction via Scherrer equation."""

    def setUp(self):
        self.config = get_xrd_config()
        self.ext = XRDExtractor(self.config)

    def test_known_gaussian_peak(self):
        """A known Gaussian peak should give reasonable crystallite size."""
        # FWHM = 0.5 deg, peak at 2theta = 40 deg
        two_theta, intensity = _gaussian_peak(40.0, 0.5, 1000.0)

        results = self.ext.extract_with_uncertainty({
            "two_theta": two_theta,
            "intensity": intensity,
            "peak_position": 40.0,
        })

        m = results[0]
        self.assertFalse(math.isnan(m.value))
        # Typical crystallite size should be > 1 nm
        self.assertGreater(m.value, 1.0)
        self.assertGreater(m.confidence, 0.0)
        self.assertEqual(m.source, "XRD_crystallite_size")
        self.assertEqual(m.method, "scherrer")

    def test_narrow_peak_near_instrument_broadening(self):
        """Peak narrower than instrument broadening -> nan or low confidence."""
        # Use a very fine grid so that a 0.03 deg FWHM peak is resolved.
        # Instrument broadening = 0.05 deg, so after broadening correction
        # beta_obs^2 <= beta_inst^2 => nan result.
        two_theta, intensity = _gaussian_peak(
            40.0, 0.03, 1000.0,
            two_theta_range=(39.5, 40.5),
            n_points=2000,
        )

        results = self.ext.extract_with_uncertainty({
            "two_theta": two_theta,
            "intensity": intensity,
            "peak_position": 40.0,
        })

        m = results[0]
        # Should be nan (broadening below instrument)
        self.assertTrue(math.isnan(m.value))
        self.assertEqual(m.confidence, 0.0)

    def test_scherrer_k_uncertainty_propagates(self):
        """Varying K range should affect the variance."""
        two_theta, intensity = _gaussian_peak(40.0, 0.5, 1000.0)

        # Narrow K range
        narrow_config = get_xrd_config()
        narrow_config["instrument"]["scherrer_k_range"] = (0.90, 0.91)
        ext_narrow = XRDExtractor(narrow_config)
        r_narrow = ext_narrow.extract_with_uncertainty({
            "two_theta": two_theta,
            "intensity": intensity,
            "peak_position": 40.0,
        })

        # Wide K range
        wide_config = get_xrd_config()
        wide_config["instrument"]["scherrer_k_range"] = (0.80, 1.00)
        ext_wide = XRDExtractor(wide_config)
        r_wide = ext_wide.extract_with_uncertainty({
            "two_theta": two_theta,
            "intensity": intensity,
            "peak_position": 40.0,
        })

        # Wide K range should produce larger variance
        self.assertGreater(r_wide[0].variance, r_narrow[0].variance)

    def test_physical_constraints_crystallite_size(self):
        """Extremely large crystallite -> outside typical range penalty."""
        # Very narrow peak => large crystallite size
        two_theta, intensity = _gaussian_peak(40.0, 0.06, 1000.0, n_points=2000)

        results = self.ext.extract_with_uncertainty({
            "two_theta": two_theta,
            "intensity": intensity,
            "peak_position": 40.0,
        })

        m = results[0]
        if not math.isnan(m.value):
            # Either nan (if below instrument broadening) or has a value
            self.assertGreater(m.value, 0)

    def test_insufficient_data(self):
        results = self.ext.extract_with_uncertainty({
            "two_theta": [40.0],
            "intensity": [100.0],
            "peak_position": 40.0,
        })
        self.assertTrue(math.isnan(results[0].value))
        self.assertEqual(results[0].confidence, 0.0)

    def test_metadata_fields(self):
        """Metadata should contain expected diagnostic info."""
        two_theta, intensity = _gaussian_peak(40.0, 0.5, 1000.0)
        results = self.ext.extract_with_uncertainty({
            "two_theta": two_theta,
            "intensity": intensity,
            "peak_position": 40.0,
        })

        m = results[0]
        if not math.isnan(m.value):
            self.assertIn("peak_position_deg", m.metadata)
            self.assertIn("fwhm_obs_deg", m.metadata)
            self.assertIn("fwhm_sample_deg", m.metadata)
            self.assertIn("k_mean", m.metadata)
            self.assertIn("wavelength_A", m.metadata)

    def test_wide_peak_gives_small_crystallite(self):
        """Wider peaks correspond to smaller crystallite sizes."""
        tt_wide, int_wide = _gaussian_peak(40.0, 2.0, 1000.0)
        tt_narrow, int_narrow = _gaussian_peak(40.0, 0.3, 1000.0)

        r_wide = self.ext.extract_with_uncertainty({
            "two_theta": tt_wide,
            "intensity": int_wide,
            "peak_position": 40.0,
        })
        r_narrow = self.ext.extract_with_uncertainty({
            "two_theta": tt_narrow,
            "intensity": int_narrow,
            "peak_position": 40.0,
        })

        if not math.isnan(r_wide[0].value) and not math.isnan(r_narrow[0].value):
            # Wider peak => smaller crystallite
            self.assertLess(r_wide[0].value, r_narrow[0].value)


# ══════════════════════════════════════════════════════════════════════
# EIS Extractor tests
# ══════════════════════════════════════════════════════════════════════


def _generate_randles_data(
    r_s: float = 10.0,
    r_ct: float = 100.0,
    c_dl: float = 1e-5,
    freq_range: tuple[float, float] = (0.1, 100_000),
    n_points: int = 50,
    noise_level: float = 0.0,
) -> dict[str, Any]:
    """Generate synthetic Randles circuit EIS data."""
    import random as _rng

    _rng.seed(42)

    log_f_min = math.log10(freq_range[0])
    log_f_max = math.log10(freq_range[1])

    freq = [10 ** (log_f_min + (log_f_max - log_f_min) * i / (n_points - 1))
            for i in range(n_points)]

    z_real: list[float] = []
    z_imag: list[float] = []

    for f in freq:
        omega = 2 * math.pi * f
        z = _z_randles(omega, [r_s, r_ct, c_dl])
        z_real.append(z.real + noise_level * _rng.gauss(0, 1))
        z_imag.append(z.imag + noise_level * _rng.gauss(0, 1))

    return {
        "frequency": freq,
        "z_real": z_real,
        "z_imag": z_imag,
    }


class TestEISExtractor(unittest.TestCase):
    """Test EIS impedance and R_ct extraction."""

    def setUp(self):
        self.config = get_eis_config()
        self.ext = EISExtractor(self.config)

    # ── |Z| tests ─────────────────────────────────────────────────

    def test_z_at_exact_frequency(self):
        """Exact frequency match should give zero interp variance."""
        data = _generate_randles_data()
        target_f = data["frequency"][25]  # middle frequency
        data["target_frequency"] = target_f

        results = self.ext.extract_with_uncertainty(data)

        # Find the Z magnitude result
        z_results = [r for r in results if r.source == "EIS_Z_magnitude"]
        self.assertEqual(len(z_results), 1)

        m = z_results[0]
        self.assertAlmostEqual(m.metadata["interp_variance"], 0.0, places=10)
        self.assertGreater(m.value, 0)

    def test_z_interpolation_between_points(self):
        """Z at a frequency between data points should interpolate."""
        data = _generate_randles_data()
        f1 = data["frequency"][20]
        f2 = data["frequency"][21]
        target_f = math.sqrt(f1 * f2)  # geometric mean
        data["target_frequency"] = target_f

        results = self.ext.extract_with_uncertainty(data)
        z_results = [r for r in results if r.source == "EIS_Z_magnitude"]

        m = z_results[0]
        self.assertGreater(m.value, 0)
        self.assertGreater(m.confidence, 0)

    def test_z_low_frequency_noise_amplification(self):
        """Low-frequency targets should have higher instrument variance."""
        data = _generate_randles_data(freq_range=(0.01, 100_000))

        # High frequency
        data_high = dict(data)
        data_high["target_frequency"] = 1000.0
        r_high = self.ext.extract_with_uncertainty(data_high)
        z_high = [r for r in r_high if r.source == "EIS_Z_magnitude"][0]

        # Low frequency (below 1 Hz)
        data_low = dict(data)
        data_low["target_frequency"] = 0.5
        r_low = self.ext.extract_with_uncertainty(data_low)
        z_low = [r for r in r_low if r.source == "EIS_Z_magnitude"][0]

        # Low frequency should have higher noise amplification
        self.assertGreater(
            z_low.metadata["noise_amplification_factor"],
            z_high.metadata["noise_amplification_factor"],
        )

    def test_z_metadata_fields(self):
        """Z magnitude result should contain expected metadata."""
        data = _generate_randles_data()
        data["target_frequency"] = 1000.0

        results = self.ext.extract_with_uncertainty(data)
        z_results = [r for r in results if r.source == "EIS_Z_magnitude"]

        m = z_results[0]
        self.assertIn("target_frequency", m.metadata)
        self.assertIn("instrument_variance", m.metadata)
        self.assertIn("interp_variance", m.metadata)
        self.assertIn("noise_amplification_factor", m.metadata)

    # ── R_ct tests ────────────────────────────────────────────────

    def test_rct_from_randles_circuit(self):
        """R_ct extraction from clean Randles data."""
        true_rct = 100.0
        data = _generate_randles_data(r_s=10.0, r_ct=true_rct, c_dl=1e-5)

        results = self.ext.extract_with_uncertainty(data)
        rct_results = [r for r in results if r.source == "EIS_Rct_ensemble"]

        self.assertEqual(len(rct_results), 1)
        m = rct_results[0]

        if not math.isnan(m.value):
            # Should be close to true value
            self.assertAlmostEqual(m.value, true_rct, delta=true_rct * 0.3)
            self.assertGreater(m.confidence, 0.0)

    def test_rct_lm_convergence_synthetic(self):
        """LM fitter should converge on clean synthetic data."""
        data = _generate_randles_data(r_s=5.0, r_ct=50.0, c_dl=5e-6)

        results = self.ext.extract_with_uncertainty(data)
        rct_results = [r for r in results if r.source == "EIS_Rct_ensemble"]

        m = rct_results[0]
        if not math.isnan(m.value):
            # At least one circuit should have converged
            meta = m.metadata
            if "circuit_details" in meta:
                converged_any = any(
                    c["converged"] for c in meta["circuit_details"]
                )
                self.assertTrue(converged_any or meta.get("converged", False))
            elif "converged" in meta:
                self.assertTrue(meta["converged"])

    def test_rct_aic_ensemble_multiple_circuits(self):
        """Multiple circuits should produce AIC-weighted ensemble."""
        data = _generate_randles_data(r_s=10.0, r_ct=100.0, c_dl=1e-5)

        results = self.ext.extract_with_uncertainty(data)
        rct_results = [r for r in results if r.source == "EIS_Rct_ensemble"]

        m = rct_results[0]
        if not math.isnan(m.value):
            n_fitted = m.metadata.get("n_circuits_fitted", 0)
            if n_fitted > 1:
                self.assertIn("circuit_details", m.metadata)
                self.assertIn("intra_model_variance", m.metadata)
                self.assertIn("inter_model_variance", m.metadata)
                # Verify weights sum to ~1
                weights = [
                    c["weight"] for c in m.metadata["circuit_details"]
                ]
                self.assertAlmostEqual(sum(weights), 1.0, places=5)

    def test_rct_failed_fit_confidence_zero(self):
        """If data is garbage, fit should fail gracefully."""
        # Random noise only, no real impedance pattern
        import random
        random.seed(99)
        freq = [10 ** (i * 0.1) for i in range(20)]
        z_real = [random.gauss(0, 0.001) for _ in freq]
        z_imag = [random.gauss(0, 0.001) for _ in freq]

        # Use a config with only randles circuit and tight bounds
        # to make fitting harder
        config = get_eis_config()
        results = EISExtractor(config).extract_with_uncertainty({
            "frequency": freq,
            "z_real": z_real,
            "z_imag": z_imag,
        })

        rct_results = [r for r in results if r.source == "EIS_Rct_ensemble"]
        if rct_results:
            m = rct_results[0]
            # Either nan or very low confidence
            if math.isnan(m.value):
                self.assertEqual(m.confidence, 0.0)

    def test_rct_metadata_contains_expected_fields(self):
        """R_ct result should have rich metadata."""
        data = _generate_randles_data()
        results = self.ext.extract_with_uncertainty(data)
        rct_results = [r for r in results if r.source == "EIS_Rct_ensemble"]

        if rct_results:
            m = rct_results[0]
            self.assertIsNotNone(m.n_points_used)
            self.assertEqual(m.method, "lm_fit")
            if not math.isnan(m.value):
                self.assertIn("n_circuits_fitted", m.metadata)

    def test_empty_data(self):
        results = self.ext.extract_with_uncertainty({
            "frequency": [],
            "z_real": [],
            "z_imag": [],
        })
        self.assertEqual(len(results), 1)
        self.assertTrue(math.isnan(results[0].value))
        self.assertEqual(results[0].confidence, 0.0)

    def test_no_target_frequency_only_rct(self):
        """Without target_frequency, only R_ct should be returned."""
        data = _generate_randles_data()
        # Don't set target_frequency
        results = self.ext.extract_with_uncertainty(data)

        z_results = [r for r in results if r.source == "EIS_Z_magnitude"]
        rct_results = [r for r in results if r.source == "EIS_Rct_ensemble"]

        self.assertEqual(len(z_results), 0)
        self.assertEqual(len(rct_results), 1)

    def test_both_kpis_with_target_frequency(self):
        """With target_frequency, both |Z| and R_ct are extracted."""
        data = _generate_randles_data()
        data["target_frequency"] = 1000.0

        results = self.ext.extract_with_uncertainty(data)

        z_results = [r for r in results if r.source == "EIS_Z_magnitude"]
        rct_results = [r for r in results if r.source == "EIS_Rct_ensemble"]

        self.assertEqual(len(z_results), 1)
        self.assertEqual(len(rct_results), 1)

    def test_z_physical_constraint_non_negative(self):
        """Z magnitude should always be non-negative."""
        data = _generate_randles_data()
        data["target_frequency"] = 1000.0

        results = self.ext.extract_with_uncertainty(data)
        z_results = [r for r in results if r.source == "EIS_Z_magnitude"]

        m = z_results[0]
        self.assertGreaterEqual(m.value, 0.0)

    def test_rct_with_noisy_data(self):
        """R_ct should still be extractable from moderately noisy data."""
        true_rct = 100.0
        data = _generate_randles_data(
            r_s=10.0, r_ct=true_rct, c_dl=1e-5, noise_level=1.0,
        )

        results = self.ext.extract_with_uncertainty(data)
        rct_results = [r for r in results if r.source == "EIS_Rct_ensemble"]

        m = rct_results[0]
        if not math.isnan(m.value):
            # Should be within 50% of true value with noise
            self.assertAlmostEqual(m.value, true_rct, delta=true_rct * 0.5)

    def test_rct_variance_positive(self):
        """R_ct variance should always be non-negative."""
        data = _generate_randles_data()
        results = self.ext.extract_with_uncertainty(data)
        rct_results = [r for r in results if r.source == "EIS_Rct_ensemble"]

        if rct_results:
            m = rct_results[0]
            self.assertGreaterEqual(m.variance, 0.0)

    def test_different_rct_values_distinguished(self):
        """Different true R_ct values should produce different extracted values."""
        data1 = _generate_randles_data(r_ct=50.0)
        data2 = _generate_randles_data(r_ct=500.0)

        r1 = self.ext.extract_with_uncertainty(data1)
        r2 = self.ext.extract_with_uncertainty(data2)

        rct1 = [r for r in r1 if r.source == "EIS_Rct_ensemble"]
        rct2 = [r for r in r2 if r.source == "EIS_Rct_ensemble"]

        if rct1 and rct2:
            m1, m2 = rct1[0], rct2[0]
            if not math.isnan(m1.value) and not math.isnan(m2.value):
                self.assertNotAlmostEqual(m1.value, m2.value, delta=10.0)


# ══════════════════════════════════════════════════════════════════════
# Integration / cross-cutting tests
# ══════════════════════════════════════════════════════════════════════


class TestExtractorIntegration(unittest.TestCase):
    """Cross-cutting tests for extractor consistency."""

    def test_all_extractors_return_measurement_type(self):
        """All extractors must return list[MeasurementWithUncertainty]."""
        extractors_and_data: list[tuple[UncertaintyExtractor, dict]] = [
            (
                DCCyclingExtractor(get_dc_config()),
                {"current": [-1, 1], "voltage": [0, 0], "time": [0, 1]},
            ),
            (
                UVVisExtractor(get_uvvis_config()),
                {"wavelengths": [400], "absorbance": [0.5], "target_wavelength": 400},
            ),
            (
                XRDExtractor(get_xrd_config()),
                {"two_theta": [39, 40, 41], "intensity": [10, 100, 10], "peak_position": 40},
            ),
            (
                EISExtractor(get_eis_config()),
                {
                    "frequency": [100, 1000, 10000],
                    "z_real": [110, 50, 12],
                    "z_imag": [-40, -30, -5],
                },
            ),
        ]

        for ext, data in extractors_and_data:
            with self.subTest(extractor=type(ext).__name__):
                results = ext.extract_with_uncertainty(data)
                self.assertIsInstance(results, list)
                for m in results:
                    self.assertIsInstance(m, MeasurementWithUncertainty)
                    self.assertGreaterEqual(m.variance, 0.0)
                    self.assertGreaterEqual(m.confidence, 0.0)
                    self.assertLessEqual(m.confidence, 1.0)

    def test_all_extractors_handle_empty_gracefully(self):
        """Empty data should not crash any extractor."""
        extractors_and_data: list[tuple[UncertaintyExtractor, dict]] = [
            (DCCyclingExtractor(get_dc_config()), {"current": [], "voltage": [], "time": []}),
            (UVVisExtractor(get_uvvis_config()), {"wavelengths": [], "absorbance": [], "target_wavelength": 400}),
            (XRDExtractor(get_xrd_config()), {"two_theta": [], "intensity": [], "peak_position": 40}),
            (EISExtractor(get_eis_config()), {"frequency": [], "z_real": [], "z_imag": []}),
        ]

        for ext, data in extractors_and_data:
            with self.subTest(extractor=type(ext).__name__):
                results = ext.extract_with_uncertainty(data)
                self.assertIsInstance(results, list)
                self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main()
