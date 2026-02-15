"""Tests for domain knowledge configuration modules."""

from __future__ import annotations

import pytest

from optimization_copilot.domain_knowledge.eis import (
    DEFAULT_EIS_CIRCUITS,
    DEFAULT_EIS_INSTRUMENT,
    DEFAULT_MODEL_SELECTION,
    EIS_PHYSICAL_CONSTRAINTS,
    EIS_QUALITY_THRESHOLDS,
    get_circuit_bounds,
    get_eis_config,
    get_instrument_impedance_spec,
    get_noise_amplification,
)
from optimization_copilot.domain_knowledge.dc_cycling import (
    DC_PHYSICAL_CONSTRAINTS,
    DC_QUALITY_THRESHOLDS,
    DEFAULT_DC_INSTRUMENT,
    get_current_accuracy,
    get_dc_config,
    get_dc_instrument_spec,
    get_zero_drift,
)
from optimization_copilot.domain_knowledge.uv_vis import (
    DEFAULT_UVVIS_INSTRUMENT,
    UVVIS_PHYSICAL_CONSTRAINTS,
    get_absorbance_noise,
    get_linear_range_max,
    get_uvvis_config,
)
from optimization_copilot.domain_knowledge.xrd import (
    DEFAULT_XRD_INSTRUMENT,
    XRD_PHYSICAL_CONSTRAINTS,
    get_instrument_broadening,
    get_scherrer_k_range,
    get_wavelength,
    get_xrd_config,
)


# ── EIS Config ──────────────────────────────────────────────────────────


class TestEISConfig:
    def test_instrument_has_impedance(self):
        assert "impedance" in DEFAULT_EIS_INSTRUMENT
        imp = DEFAULT_EIS_INSTRUMENT["impedance"]
        assert imp["z_relative_accuracy"] == pytest.approx(0.001)
        assert imp["phase_accuracy_deg"] == pytest.approx(0.1)

    def test_freq_range(self):
        fr = DEFAULT_EIS_INSTRUMENT["impedance"]["freq_range_hz"]
        assert fr[0] == pytest.approx(0.1)
        assert fr[1] == pytest.approx(100_000)

    def test_circuits_count(self):
        assert len(DEFAULT_EIS_CIRCUITS) == 3

    def test_randles_circuit(self):
        randles = DEFAULT_EIS_CIRCUITS[0]
        assert randles["name"] == "randles"
        assert randles["rct_index"] == 1
        assert "R_ct" in randles["init_bounds"]

    def test_all_circuits_have_bounds(self):
        for c in DEFAULT_EIS_CIRCUITS:
            assert len(c["init_bounds"]) == len(c["params"])
            for p in c["params"]:
                bounds = c["init_bounds"][p]
                assert bounds[0] < bounds[1], f"{c['name']}.{p} invalid bounds"

    def test_physical_constraints(self):
        assert "Rct" in EIS_PHYSICAL_CONSTRAINTS
        assert EIS_PHYSICAL_CONSTRAINTS["Rct"]["min"] == 0
        assert "z_magnitude" in EIS_PHYSICAL_CONSTRAINTS

    def test_quality_thresholds(self):
        assert EIS_QUALITY_THRESHOLDS["confidence_min"] == 0.5
        assert EIS_QUALITY_THRESHOLDS["relative_uncertainty_max"] == 0.5

    def test_model_selection(self):
        assert DEFAULT_MODEL_SELECTION["method"] == "aic_weighted_ensemble"
        assert DEFAULT_MODEL_SELECTION["min_models_for_ensemble"] == 2

    def test_get_eis_config(self):
        cfg = get_eis_config()
        assert "instrument" in cfg
        assert "circuits" in cfg
        assert "physical_constraints" in cfg
        assert "quality_thresholds" in cfg

    def test_get_instrument_impedance_spec(self):
        spec = get_instrument_impedance_spec()
        assert "z_relative_accuracy" in spec

    def test_get_noise_amplification_low_freq(self):
        assert get_noise_amplification(0.5) == pytest.approx(3.0)

    def test_get_noise_amplification_mid_freq(self):
        assert get_noise_amplification(5.0) == pytest.approx(1.5)

    def test_get_noise_amplification_high_freq(self):
        assert get_noise_amplification(1000.0) == pytest.approx(1.0)

    def test_get_circuit_bounds_randles(self):
        bounds = get_circuit_bounds("randles")
        assert bounds is not None
        assert "R_ct" in bounds
        assert bounds["R_ct"] == (1, 50_000)

    def test_get_circuit_bounds_unknown(self):
        assert get_circuit_bounds("nonexistent") is None


# ── DC Cycling Config ───────────────────────────────────────────────────


class TestDCConfig:
    def test_instrument_has_dc(self):
        assert "dc" in DEFAULT_DC_INSTRUMENT
        dc = DEFAULT_DC_INSTRUMENT["dc"]
        assert dc["current_accuracy_a"] == pytest.approx(1e-7)
        assert dc["sampling_rate_hz"] == 100

    def test_physical_constraints_ce(self):
        ce = DC_PHYSICAL_CONSTRAINTS["CE"]
        assert ce["min"] == 0
        assert ce["max"] == 105
        assert ce["warning_range"] == (95, 105)

    def test_quality_thresholds(self):
        assert DC_QUALITY_THRESHOLDS["min_points_for_integration"] == 10

    def test_get_dc_config(self):
        cfg = get_dc_config()
        assert "instrument" in cfg
        assert "physical_constraints" in cfg

    def test_get_dc_instrument_spec(self):
        spec = get_dc_instrument_spec()
        assert "current_accuracy_a" in spec

    def test_get_zero_drift(self):
        assert get_zero_drift() == pytest.approx(5e-7)

    def test_get_current_accuracy(self):
        assert get_current_accuracy() == pytest.approx(1e-7)


# ── UV-Vis Config ───────────────────────────────────────────────────────


class TestUVVisConfig:
    def test_instrument_wavelength_range(self):
        wr = DEFAULT_UVVIS_INSTRUMENT["wavelength_range_nm"]
        assert wr == (200, 900)

    def test_absorbance_noise_ranges(self):
        noise = DEFAULT_UVVIS_INSTRUMENT["absorbance_noise"]
        assert noise["low_abs"] < noise["mid_abs"] < noise["high_abs"]

    def test_linear_range(self):
        assert DEFAULT_UVVIS_INSTRUMENT["linear_range_max"] == 2.5

    def test_physical_constraints_absorbance(self):
        a = UVVIS_PHYSICAL_CONSTRAINTS["absorbance"]
        assert a["min"] == pytest.approx(-0.05)
        assert a["max"] == pytest.approx(4.0)

    def test_get_uvvis_config(self):
        cfg = get_uvvis_config()
        assert "instrument" in cfg
        assert "physical_constraints" in cfg

    def test_get_absorbance_noise_low(self):
        assert get_absorbance_noise(0.1) == pytest.approx(0.002)

    def test_get_absorbance_noise_mid(self):
        assert get_absorbance_noise(1.0) == pytest.approx(0.005)

    def test_get_absorbance_noise_high(self):
        assert get_absorbance_noise(3.0) == pytest.approx(0.02)

    def test_get_linear_range_max(self):
        assert get_linear_range_max() == pytest.approx(2.5)


# ── XRD Config ──────────────────────────────────────────────────────────


class TestXRDConfig:
    def test_instrument_broadening(self):
        assert DEFAULT_XRD_INSTRUMENT["instrument_broadening_deg"] == pytest.approx(0.05)

    def test_scherrer_k_range(self):
        k_range = DEFAULT_XRD_INSTRUMENT["scherrer_k_range"]
        assert k_range[0] == pytest.approx(0.89)
        assert k_range[1] == pytest.approx(0.94)

    def test_wavelength(self):
        assert DEFAULT_XRD_INSTRUMENT["wavelength_angstrom"] == pytest.approx(1.5406)

    def test_physical_constraints_crystallite_size(self):
        cs = XRD_PHYSICAL_CONSTRAINTS["crystallite_size"]
        assert cs["min"] == 1
        assert cs["max"] == 1000

    def test_get_xrd_config(self):
        cfg = get_xrd_config()
        assert "instrument" in cfg
        assert "physical_constraints" in cfg
        assert "quality_thresholds" in cfg

    def test_get_scherrer_k_range(self):
        k_lo, k_hi = get_scherrer_k_range()
        assert k_lo == pytest.approx(0.89)
        assert k_hi == pytest.approx(0.94)

    def test_get_instrument_broadening(self):
        assert get_instrument_broadening() == pytest.approx(0.05)

    def test_get_wavelength(self):
        assert get_wavelength() == pytest.approx(1.5406)
