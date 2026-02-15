"""Tests for Layer 2: KPIValidator (physical range validation)."""

from __future__ import annotations

from optimization_copilot.anomaly.kpi_validator import KPIAnomaly, KPIValidator


# ── KPIAnomaly dataclass tests ─────────────────────────────────────────


class TestKPIAnomalyDataclass:
    def test_creation(self):
        a = KPIAnomaly(
            kpi_name="CE",
            value=120.0,
            expected_range=(0.0, 105.0),
            severity="error",
            message="CE=120 is outside range",
        )
        assert a.kpi_name == "CE"
        assert a.value == 120.0
        assert a.expected_range == (0.0, 105.0)
        assert a.severity == "error"

    def test_range_tuple(self):
        a = KPIAnomaly("x", 1.0, (0.0, 2.0), "warning", "msg")
        assert isinstance(a.expected_range, tuple)
        assert len(a.expected_range) == 2


# ── CE (Coulombic Efficiency) tests ───────────────────────────────────


class TestCEValidation:
    def test_ce_in_range(self):
        v = KPIValidator()
        result = v.validate("CE", 98.0)
        assert result is None

    def test_ce_at_lower_bound(self):
        v = KPIValidator()
        result = v.validate("CE", 0.0)
        assert result is None

    def test_ce_at_upper_bound(self):
        v = KPIValidator()
        result = v.validate("CE", 105.0)
        assert result is None

    def test_ce_out_of_range_high(self):
        v = KPIValidator()
        result = v.validate("CE", 120.0)
        assert result is not None
        assert result.kpi_name == "CE"
        assert result.value == 120.0

    def test_ce_out_of_range_negative(self):
        v = KPIValidator()
        result = v.validate("CE", -5.0)
        assert result is not None

    def test_coulombic_efficiency_alias(self):
        """The alias 'coulombic_efficiency' should also work."""
        v = KPIValidator()
        result = v.validate("coulombic_efficiency", 110.0)
        assert result is not None


# ── Rct (Charge Transfer Resistance) tests ────────────────────────────


class TestRctValidation:
    def test_rct_valid(self):
        v = KPIValidator()
        result = v.validate("Rct", 50.0)
        assert result is None

    def test_rct_zero(self):
        v = KPIValidator()
        result = v.validate("Rct", 0.0)
        assert result is None

    def test_rct_negative(self):
        v = KPIValidator()
        result = v.validate("Rct", -10.0)
        assert result is not None
        assert result.severity == "error"

    def test_rct_very_large(self):
        v = KPIValidator()
        result = v.validate("Rct", 1e9)
        assert result is None  # No upper bound


# ── Grain size tests ──────────────────────────────────────────────────


class TestGrainSizeValidation:
    def test_grain_size_valid(self):
        v = KPIValidator()
        result = v.validate("grain_size", 50.0)
        assert result is None

    def test_grain_size_at_lower_bound(self):
        v = KPIValidator()
        result = v.validate("grain_size", 1.0)
        assert result is None

    def test_grain_size_too_small(self):
        v = KPIValidator()
        result = v.validate("grain_size", 0.5)
        assert result is not None


# ── Absorbance tests ──────────────────────────────────────────────────


class TestAbsorbanceValidation:
    def test_absorbance_in_range(self):
        v = KPIValidator()
        result = v.validate("absorbance", 2.5)
        assert result is None

    def test_absorbance_at_zero(self):
        v = KPIValidator()
        result = v.validate("absorbance", 0.0)
        assert result is None

    def test_absorbance_at_upper(self):
        v = KPIValidator()
        result = v.validate("absorbance", 5.0)
        assert result is None

    def test_absorbance_out_of_range_high(self):
        v = KPIValidator()
        result = v.validate("absorbance", 6.0)
        assert result is not None

    def test_absorbance_negative(self):
        v = KPIValidator()
        result = v.validate("absorbance", -0.5)
        assert result is not None


# ── Conversion / Selectivity / Yield tests ────────────────────────────


class TestConversionSelectivityYield:
    def test_conversion_valid(self):
        v = KPIValidator()
        assert v.validate("conversion", 85.0) is None

    def test_conversion_invalid_high(self):
        v = KPIValidator()
        result = v.validate("conversion", 105.0)
        assert result is not None

    def test_conversion_invalid_negative(self):
        v = KPIValidator()
        result = v.validate("conversion", -1.0)
        assert result is not None

    def test_selectivity_valid(self):
        v = KPIValidator()
        assert v.validate("selectivity", 99.9) is None

    def test_selectivity_invalid(self):
        v = KPIValidator()
        result = v.validate("selectivity", 101.0)
        assert result is not None

    def test_yield_pct_valid(self):
        v = KPIValidator()
        assert v.validate("yield_pct", 50.0) is None

    def test_yield_pct_invalid(self):
        v = KPIValidator()
        result = v.validate("yield_pct", -10.0)
        assert result is not None


# ── Band gap tests ────────────────────────────────────────────────────


class TestBandGapValidation:
    def test_band_gap_valid(self):
        v = KPIValidator()
        assert v.validate("band_gap", 3.2) is None

    def test_band_gap_too_low(self):
        v = KPIValidator()
        result = v.validate("band_gap", 0.1)
        assert result is not None

    def test_band_gap_too_high(self):
        v = KPIValidator()
        result = v.validate("band_gap", 7.0)
        assert result is not None

    def test_band_gap_at_lower_bound(self):
        v = KPIValidator()
        assert v.validate("band_gap", 0.5) is None

    def test_band_gap_at_upper_bound(self):
        v = KPIValidator()
        assert v.validate("band_gap", 6.0) is None


# ── PCE tests ─────────────────────────────────────────────────────────


class TestPCEValidation:
    def test_pce_valid(self):
        v = KPIValidator()
        assert v.validate("PCE", 25.0) is None

    def test_pce_too_high(self):
        v = KPIValidator()
        result = v.validate("PCE", 40.0)
        assert result is not None

    def test_pce_negative(self):
        v = KPIValidator()
        result = v.validate("PCE", -1.0)
        assert result is not None

    def test_pce_alias(self):
        v = KPIValidator()
        result = v.validate("power_conversion_efficiency", 40.0)
        assert result is not None


# ── peak_intensity tests ──────────────────────────────────────────────


class TestPeakIntensityValidation:
    def test_peak_intensity_valid(self):
        v = KPIValidator()
        assert v.validate("peak_intensity", 1000.0) is None

    def test_peak_intensity_zero(self):
        v = KPIValidator()
        assert v.validate("peak_intensity", 0.0) is None

    def test_peak_intensity_negative(self):
        v = KPIValidator()
        result = v.validate("peak_intensity", -1.0)
        assert result is not None


# ── validate_all tests ─────────────────────────────────────────────────


class TestValidateAll:
    def test_validate_all_mixed(self):
        """Some valid, some invalid KPIs."""
        v = KPIValidator()
        results = v.validate_all({
            "CE": 98.0,          # valid
            "Rct": -5.0,         # invalid
            "conversion": 50.0,  # valid
            "band_gap": 0.1,     # invalid
        })
        assert len(results) == 2
        names = {r.kpi_name for r in results}
        assert "Rct" in names
        assert "band_gap" in names

    def test_validate_all_clean(self):
        """All valid -> empty list."""
        v = KPIValidator()
        results = v.validate_all({
            "CE": 98.0,
            "conversion": 50.0,
            "selectivity": 80.0,
        })
        assert results == []

    def test_validate_all_empty(self):
        """Empty dict -> empty list."""
        v = KPIValidator()
        results = v.validate_all({})
        assert results == []

    def test_validate_all_all_invalid(self):
        """All invalid -> anomaly for each."""
        v = KPIValidator()
        results = v.validate_all({
            "CE": -10.0,
            "Rct": -5.0,
            "band_gap": 0.0,
        })
        assert len(results) == 3


# ── Unknown KPI tests ─────────────────────────────────────────────────


class TestUnknownKPI:
    def test_unknown_kpi_passes(self):
        """Unknown KPI names should pass validation."""
        v = KPIValidator()
        result = v.validate("unknown_metric", 999.0)
        assert result is None

    def test_unknown_kpi_in_validate_all(self):
        """Unknown KPIs in validate_all should be skipped."""
        v = KPIValidator()
        results = v.validate_all({"unknown_x": 1.0, "unknown_y": -1.0})
        assert results == []


# ── Case insensitivity tests ──────────────────────────────────────────


class TestCaseInsensitivity:
    def test_uppercase_kpi(self):
        v = KPIValidator()
        result = v.validate("CE", 98.0)
        assert result is None

    def test_lowercase_kpi(self):
        v = KPIValidator()
        result = v.validate("ce", 98.0)
        assert result is None

    def test_mixed_case_kpi(self):
        v = KPIValidator()
        # "Conversion" -> lower -> "conversion"
        result = v.validate("Conversion", 50.0)
        assert result is None


# ── DomainConfig integration tests ────────────────────────────────────


class TestWithDomainConfig:
    def test_without_domain_config(self):
        """Default bounds should work without DomainConfig."""
        v = KPIValidator(domain_config=None)
        result = v.validate("CE", 120.0)
        assert result is not None

    def test_severity_message(self):
        """Check that message contains useful information."""
        v = KPIValidator()
        result = v.validate("CE", 120.0)
        assert result is not None
        assert "CE" in result.message
        assert "120" in result.message
