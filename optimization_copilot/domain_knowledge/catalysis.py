"""Catalysis domain configuration.

Instrument specs, physical constraints, catalyst/ligand/base libraries,
and compatibility rules for Suzuki-Miyaura and related coupling reactions.
"""

from __future__ import annotations

from typing import Any


# -- Instrument Specs -------------------------------------------------------

DEFAULT_CATALYSIS_INSTRUMENTS: dict[str, Any] = {
    "hplc": {
        "model": "HPLC",
        "yield_accuracy_pct": 2.0,
        "detection_limit_pct": 0.5,
        "retention_time_variance_s": 0.3,
    },
    "reactor": {
        "model": "flow_reactor",
        "temperature_accuracy_c": 1.0,
        "flow_rate_accuracy_pct": 0.5,
        "mixing_efficiency": 0.95,
    },
}


# -- Physical Constraints ---------------------------------------------------

CATALYSIS_PHYSICAL_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "yield": {
        "min": 0,
        "max": 100,
        "unit": "%",
        "warning_range": (95, 100),
    },
    "temperature": {
        "min": 20,
        "max": 200,
        "unit": "\u00b0C",
    },
    "concentration": {
        "min": 0.001,
        "max": 2.0,
        "unit": "M",
    },
}


# -- Quality Thresholds -----------------------------------------------------

CATALYSIS_QUALITY_THRESHOLDS: dict[str, float] = {
    "confidence_min": 0.5,
    "relative_uncertainty_max": 0.5,
}


# -- Catalyst / Ligand / Base Libraries ------------------------------------

CATALYSTS: list[str] = ["Pd(OAc)2", "Pd(PPh3)4", "PdCl2", "Pd2(dba)3"]

LIGANDS: list[str] = ["PPh3", "XPhos", "SPhos", "BINAP", "dppf", "PCy3"]

BASES: list[str] = ["K2CO3", "Cs2CO3", "KOtBu", "Et3N", "DBU"]


# -- Known Incompatibilities ------------------------------------------------

KNOWN_INCOMPATIBILITIES: list[dict[str, str]] = [
    {
        "catalyst": "Pd(OAc)2",
        "ligand": "BINAP",
        "reason": "oxidation state mismatch",
    },
    {
        "catalyst": "PdCl2",
        "ligand": "PCy3",
        "reason": "poor activation",
    },
]


# -- Rule Functions ---------------------------------------------------------

def validate_yield_physics(
    yield_value: float,
    yield_variance: float,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Validate a yield measurement against physical constraints.

    Returns a dict with keys: valid, confidence_modifier, flags.
    """
    flags: list[str] = []
    confidence_modifier = 1.0
    valid = True

    constraints = CATALYSIS_PHYSICAL_CONSTRAINTS["yield"]
    y_min = constraints["min"]
    y_max = constraints["max"]

    # Out-of-range check
    if yield_value < y_min or yield_value > y_max:
        valid = False
        confidence_modifier = 0.3
        flags.append("yield_out_of_range")

    # Suspiciously high yield
    warn_lo, warn_hi = constraints["warning_range"]
    if valid and warn_lo <= yield_value <= warn_hi:
        flags.append("suspiciously_high_yield")
        confidence_modifier = 0.7

    # Near-zero yield
    if valid and yield_value < 1.0:
        flags.append("near_zero_yield")
        confidence_modifier = min(confidence_modifier, 0.8)

    # High variance relative to yield
    if valid and yield_value > 0 and yield_variance / max(yield_value, 1e-9) > 0.5:
        flags.append("high_relative_variance")
        confidence_modifier = min(confidence_modifier, 0.6)

    return {
        "valid": valid,
        "confidence_modifier": confidence_modifier,
        "flags": flags,
    }


def check_catalyst_ligand_compatibility(
    catalyst: str,
    ligand: str,
    known_incompatibilities: list[dict[str, str]] | None = None,
) -> dict[str, str] | None:
    """Check whether a catalyst-ligand pair has a known incompatibility.

    Returns the incompatibility dict if found, otherwise None.
    """
    if known_incompatibilities is None:
        known_incompatibilities = KNOWN_INCOMPATIBILITIES

    for entry in known_incompatibilities:
        if entry["catalyst"] == catalyst and entry["ligand"] == ligand:
            return entry
    return None


def get_catalysis_rules() -> dict[str, Any]:
    """Return a dict mapping rule names to their validation functions."""
    return {
        "validate_yield_physics": validate_yield_physics,
        "check_catalyst_ligand_compatibility": check_catalyst_ligand_compatibility,
    }


# -- Convenience Accessor ---------------------------------------------------

def get_catalysis_config() -> dict[str, Any]:
    """Return full catalysis domain configuration as a single dict."""
    return {
        "instrument": DEFAULT_CATALYSIS_INSTRUMENTS,
        "physical_constraints": CATALYSIS_PHYSICAL_CONSTRAINTS,
        "quality_thresholds": CATALYSIS_QUALITY_THRESHOLDS,
        "catalysts": CATALYSTS,
        "ligands": LIGANDS,
        "bases": BASES,
        "known_incompatibilities": KNOWN_INCOMPATIBILITIES,
        "rules": get_catalysis_rules(),
    }
