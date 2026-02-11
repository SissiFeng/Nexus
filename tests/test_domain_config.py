"""Comprehensive tests for domain configuration infrastructure (Phase 1A v5).

Tests cover DomainConfig loader, catalysis module, and perovskite module.
"""

from __future__ import annotations

import pytest

from optimization_copilot.domain_knowledge.loader import DomainConfig
from optimization_copilot.domain_knowledge.catalysis import (
    DEFAULT_CATALYSIS_INSTRUMENTS,
    CATALYSIS_PHYSICAL_CONSTRAINTS,
    CATALYSIS_QUALITY_THRESHOLDS,
    KNOWN_INCOMPATIBILITIES,
    CATALYSTS,
    LIGANDS,
    BASES,
    validate_yield_physics,
    check_catalyst_ligand_compatibility,
    get_catalysis_config,
    get_catalysis_rules,
)
from optimization_copilot.domain_knowledge.perovskite import (
    DEFAULT_PEROVSKITE_INSTRUMENTS,
    PEROVSKITE_PHYSICAL_CONSTRAINTS,
    PEROVSKITE_QUALITY_THRESHOLDS,
    PHASE_STABILITY_RULES,
    validate_pce_physics,
    check_phase_stability,
    check_simplex_constraint,
    get_perovskite_config,
    get_perovskite_rules,
)


# ===================================================================
# DomainConfig class tests
# ===================================================================


class TestDomainConfigClass:
    """Tests for the DomainConfig loader."""

    def test_supported_domains(self) -> None:
        """All 3 domains load without error."""
        for domain in ("electrochemistry", "catalysis", "perovskite"):
            cfg = DomainConfig(domain)
            assert cfg.domain_name == domain

    def test_unknown_domain_raises(self) -> None:
        """ValueError for unsupported domain name."""
        with pytest.raises(ValueError, match="Unknown domain"):
            DomainConfig("unknown")

    def test_repr(self) -> None:
        """String representation includes domain name and counts."""
        cfg = DomainConfig("electrochemistry")
        r = repr(cfg)
        assert "electrochemistry" in r
        assert "instruments=" in r
        assert "constraints=" in r

    # -- Electrochemistry -----------------------------------------------

    def test_electrochemistry_instruments(self) -> None:
        """Electrochemistry has eis, dc, uv_vis, xrd instrument keys."""
        cfg = DomainConfig("electrochemistry")
        instruments = cfg.get_instruments()
        assert "eis" in instruments
        assert "dc" in instruments
        assert "uv_vis" in instruments
        assert "xrd" in instruments

    def test_electrochemistry_constraints(self) -> None:
        """Electrochemistry constraints include Rct, CE, absorbance, etc."""
        cfg = DomainConfig("electrochemistry")
        constraints = cfg.get_constraints()
        assert "Rct" in constraints
        assert "CE" in constraints
        assert "absorbance" in constraints
        assert "crystallite_size" in constraints

    def test_electrochemistry_quality(self) -> None:
        """Electrochemistry quality thresholds have confidence_min."""
        cfg = DomainConfig("electrochemistry")
        qt = cfg.get_quality_thresholds()
        assert "confidence_min" in qt
        assert qt["confidence_min"] == 0.5

    # -- Catalysis via DomainConfig -------------------------------------

    def test_catalysis_instruments(self) -> None:
        """Catalysis has hplc and reactor instrument keys."""
        cfg = DomainConfig("catalysis")
        instruments = cfg.get_instruments()
        assert "hplc" in instruments
        assert "reactor" in instruments

    def test_catalysis_constraints(self) -> None:
        """Catalysis constraints: yield [0,100], temperature, concentration."""
        cfg = DomainConfig("catalysis")
        constraints = cfg.get_constraints()
        assert constraints["yield"]["min"] == 0
        assert constraints["yield"]["max"] == 100
        assert "temperature" in constraints
        assert "concentration" in constraints

    def test_catalysis_quality_thresholds(self) -> None:
        """Catalysis quality thresholds present."""
        cfg = DomainConfig("catalysis")
        qt = cfg.get_quality_thresholds()
        assert qt["confidence_min"] == 0.5
        assert qt["relative_uncertainty_max"] == 0.5

    def test_catalysis_incompatibilities(self) -> None:
        """Catalysis has 2 known incompatibility entries."""
        cfg = DomainConfig("catalysis")
        incompat = cfg.get_known_incompatibilities()
        assert len(incompat) == 2

    def test_catalysis_rules(self) -> None:
        """Catalysis rules include validation functions."""
        cfg = DomainConfig("catalysis")
        rules = cfg.get_rules()
        assert "validate_yield_physics" in rules
        assert "check_catalyst_ligand_compatibility" in rules

    # -- Perovskite via DomainConfig ------------------------------------

    def test_perovskite_instruments(self) -> None:
        """Perovskite has xrd, pl, solar_simulator, spin_coater keys."""
        cfg = DomainConfig("perovskite")
        instruments = cfg.get_instruments()
        assert "xrd" in instruments
        assert "pl" in instruments
        assert "solar_simulator" in instruments
        assert "spin_coater" in instruments

    def test_perovskite_constraints(self) -> None:
        """Perovskite constraints: PCE [0,33], bandgap, composition."""
        cfg = DomainConfig("perovskite")
        constraints = cfg.get_constraints()
        assert constraints["PCE"]["min"] == 0
        assert constraints["PCE"]["max"] == 33
        assert "bandgap" in constraints
        assert "composition" in constraints

    def test_perovskite_quality_thresholds(self) -> None:
        """Perovskite quality thresholds present."""
        cfg = DomainConfig("perovskite")
        qt = cfg.get_quality_thresholds()
        assert qt["confidence_min"] == 0.5

    def test_perovskite_phase_stability_rules(self) -> None:
        """Perovskite config includes phase stability rules."""
        cfg = DomainConfig("perovskite")
        raw = cfg.config
        assert "phase_stability_rules" in raw
        assert len(raw["phase_stability_rules"]) == 2


# ===================================================================
# Catalysis module tests
# ===================================================================


class TestCatalysisConfig:
    """Tests for the catalysis domain module."""

    def test_catalysis_hplc_accuracy(self) -> None:
        """HPLC yield accuracy is 2.0%."""
        assert DEFAULT_CATALYSIS_INSTRUMENTS["hplc"]["yield_accuracy_pct"] == 2.0

    def test_catalysis_reactor_temp_accuracy(self) -> None:
        """Reactor temperature accuracy is 1.0 degC."""
        assert DEFAULT_CATALYSIS_INSTRUMENTS["reactor"]["temperature_accuracy_c"] == 1.0

    def test_catalysis_yield_bounds(self) -> None:
        """Yield constraints are [0, 100]."""
        y = CATALYSIS_PHYSICAL_CONSTRAINTS["yield"]
        assert y["min"] == 0
        assert y["max"] == 100

    def test_catalysis_catalysts_count(self) -> None:
        """4 catalysts in the library."""
        assert len(CATALYSTS) == 4

    def test_catalysis_ligands_count(self) -> None:
        """6 ligands in the library."""
        assert len(LIGANDS) == 6

    def test_catalysis_bases_count(self) -> None:
        """5 bases in the library."""
        assert len(BASES) == 5

    def test_validate_yield_above_100(self) -> None:
        """Yield > 100 is invalid with modifier 0.3."""
        result = validate_yield_physics(105.0, 1.0)
        assert result["valid"] is False
        assert result["confidence_modifier"] == 0.3
        assert "yield_out_of_range" in result["flags"]

    def test_validate_yield_normal(self) -> None:
        """Normal yield is valid with modifier 1.0."""
        result = validate_yield_physics(55.0, 2.0)
        assert result["valid"] is True
        assert result["confidence_modifier"] == 1.0
        assert len(result["flags"]) == 0

    def test_validate_yield_suspicious_high(self) -> None:
        """Yield in warning range flags suspiciously_high_yield."""
        result = validate_yield_physics(97.0, 1.0)
        assert result["valid"] is True
        assert "suspiciously_high_yield" in result["flags"]
        assert result["confidence_modifier"] == 0.7

    def test_validate_yield_near_zero(self) -> None:
        """Yield near zero flags near_zero_yield."""
        result = validate_yield_physics(0.5, 0.1)
        assert result["valid"] is True
        assert "near_zero_yield" in result["flags"]

    def test_compatibility_known_bad(self) -> None:
        """Known bad pair returns diagnosis dict."""
        result = check_catalyst_ligand_compatibility("Pd(OAc)2", "BINAP")
        assert result is not None
        assert result["reason"] == "oxidation state mismatch"

    def test_compatibility_ok(self) -> None:
        """Compatible pair returns None."""
        result = check_catalyst_ligand_compatibility("Pd(OAc)2", "PPh3")
        assert result is None

    def test_get_catalysis_config(self) -> None:
        """get_catalysis_config has required keys."""
        cfg = get_catalysis_config()
        assert "instrument" in cfg
        assert "physical_constraints" in cfg
        assert "quality_thresholds" in cfg
        assert "catalysts" in cfg
        assert "ligands" in cfg
        assert "bases" in cfg
        assert "known_incompatibilities" in cfg
        assert "rules" in cfg


# ===================================================================
# Perovskite module tests
# ===================================================================


class TestPerovskiteConfig:
    """Tests for the perovskite domain module."""

    def test_perovskite_xrd_broadening(self) -> None:
        """XRD instrument broadening is 0.08 deg."""
        assert DEFAULT_PEROVSKITE_INSTRUMENTS["xrd"]["instrument_broadening_deg"] == 0.08

    def test_perovskite_pl_noise(self) -> None:
        """PL intensity noise is 3.0%."""
        assert DEFAULT_PEROVSKITE_INSTRUMENTS["pl"]["intensity_noise_pct"] == 3.0

    def test_perovskite_solar_sim_accuracy(self) -> None:
        """Solar simulator PCE accuracy is 0.5%."""
        assert DEFAULT_PEROVSKITE_INSTRUMENTS["solar_simulator"]["pce_accuracy_pct"] == 0.5

    def test_perovskite_pce_max(self) -> None:
        """PCE max is 33 (Shockley-Queisser limit)."""
        assert PEROVSKITE_PHYSICAL_CONSTRAINTS["PCE"]["max"] == 33

    def test_perovskite_bandgap_target(self) -> None:
        """Bandgap target range is (1.3, 1.5) eV."""
        assert PEROVSKITE_PHYSICAL_CONSTRAINTS["bandgap"]["target_range"] == (1.3, 1.5)

    def test_perovskite_simplex_tolerance(self) -> None:
        """Composition simplex tolerance is 0.01."""
        assert PEROVSKITE_PHYSICAL_CONSTRAINTS["composition"]["simplex_tolerance"] == 0.01

    def test_validate_pce_above_max(self) -> None:
        """PCE > 33 is invalid."""
        result = validate_pce_physics(35.0, 1.0)
        assert result["valid"] is False
        assert "pce_out_of_range" in result["flags"]

    def test_validate_pce_normal(self) -> None:
        """Normal PCE is valid."""
        result = validate_pce_physics(20.0, 0.5)
        assert result["valid"] is True
        assert result["confidence_modifier"] == 1.0

    def test_phase_stability_high_fa(self) -> None:
        """FA > 0.85 triggers delta-phase warning."""
        result = check_phase_stability({"FA": 0.90, "Cs": 0.10, "Br": 0.50})
        assert result is not None
        assert "delta-phase" in result["issue"]

    def test_phase_stability_low_cs_br(self) -> None:
        """Low Cs and Br triggers poor stability warning."""
        result = check_phase_stability({"FA": 0.50, "Cs": 0.03, "Br": 0.05})
        assert result is not None
        assert "poor phase stability" in result["issue"]

    def test_phase_stability_ok(self) -> None:
        """Good composition returns None."""
        result = check_phase_stability({"FA": 0.70, "Cs": 0.15, "Br": 0.30})
        assert result is None

    def test_simplex_valid(self) -> None:
        """Fractions summing to 1.0 pass simplex check."""
        assert check_simplex_constraint({"A": 0.5, "B": 0.3, "C": 0.2}) is True

    def test_simplex_invalid(self) -> None:
        """Fractions summing to 1.5 fail simplex check."""
        assert check_simplex_constraint({"A": 0.5, "B": 0.5, "C": 0.5}) is False

    def test_get_perovskite_config(self) -> None:
        """get_perovskite_config has all required keys."""
        cfg = get_perovskite_config()
        assert "instrument" in cfg
        assert "physical_constraints" in cfg
        assert "quality_thresholds" in cfg
        assert "phase_stability_rules" in cfg
        assert "rules" in cfg
