"""Perovskite solar cell domain configuration.

Instrument specs, physical constraints, composition rules, and
phase stability checks for halide perovskite optimization campaigns.
"""

from __future__ import annotations

from typing import Any


# -- Instrument Specs -------------------------------------------------------

DEFAULT_PEROVSKITE_INSTRUMENTS: dict[str, Any] = {
    "xrd": {
        "model": "lab XRD",
        "instrument_broadening_deg": 0.08,
        "two_theta_accuracy_deg": 0.02,
        "scherrer_k_range": (0.89, 0.94),
        "wavelength_angstrom": 1.5406,
    },
    "pl": {
        "model": "PL spectrometer",
        "wavelength_range_nm": (400, 900),
        "intensity_noise_pct": 3.0,
        "peak_position_accuracy_nm": 0.5,
    },
    "solar_simulator": {
        "model": "AM1.5G",
        "pce_accuracy_pct": 0.5,
        "jsc_accuracy_ma_cm2": 0.1,
        "voc_accuracy_mv": 5.0,
        "ff_accuracy_pct": 1.0,
    },
    "spin_coater": {
        "speed_accuracy_rpm": 10,
        "time_accuracy_s": 0.5,
    },
}


# -- Physical Constraints ---------------------------------------------------

PEROVSKITE_PHYSICAL_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "PCE": {
        "min": 0,
        "max": 33,           # Shockley-Queisser limit for single-junction
        "unit": "%",
        "warning_range": (28, 33),
    },
    "bandgap": {
        "min": 0.5,
        "max": 3.0,
        "unit": "eV",
        "target_range": (1.3, 1.5),
    },
    "stability_hours": {
        "min": 0,
        "unit": "hours",
    },
    "composition": {
        "simplex_tolerance": 0.01,
        "halide_simplex": True,
    },
}


# -- Quality Thresholds -----------------------------------------------------

PEROVSKITE_QUALITY_THRESHOLDS: dict[str, float] = {
    "confidence_min": 0.5,
    "relative_uncertainty_max": 0.5,
}


# -- Phase Stability Rules --------------------------------------------------

PHASE_STABILITY_RULES: list[dict[str, str]] = [
    {
        "condition": "FA > 0.85",
        "issue": "delta-phase formation at room temperature",
    },
    {
        "condition": "Cs < 0.05 and Br < 0.1",
        "issue": "poor phase stability",
    },
]


# -- Rule Functions ---------------------------------------------------------

def validate_pce_physics(
    pce_value: float,
    pce_variance: float,
) -> dict[str, Any]:
    """Validate a PCE measurement against physical constraints.

    Returns a dict with keys: valid, confidence_modifier, flags.
    """
    flags: list[str] = []
    confidence_modifier = 1.0
    valid = True

    constraints = PEROVSKITE_PHYSICAL_CONSTRAINTS["PCE"]
    pce_min = constraints["min"]
    pce_max = constraints["max"]

    # Out-of-range check
    if pce_value < pce_min or pce_value > pce_max:
        valid = False
        confidence_modifier = 0.3
        flags.append("pce_out_of_range")

    # Suspiciously high PCE
    warn_lo, warn_hi = constraints["warning_range"]
    if valid and warn_lo <= pce_value <= warn_hi:
        flags.append("suspiciously_high_pce")
        confidence_modifier = 0.7

    # High variance relative to PCE
    if valid and pce_value > 0 and pce_variance / max(pce_value, 1e-9) > 0.5:
        flags.append("high_relative_variance")
        confidence_modifier = min(confidence_modifier, 0.6)

    return {
        "valid": valid,
        "confidence_modifier": confidence_modifier,
        "flags": flags,
    }


def check_phase_stability(composition: dict[str, float]) -> dict[str, str] | None:
    """Check a perovskite composition against phase stability rules.

    Args:
        composition: dict mapping component names (e.g. "FA", "Cs", "Br")
                     to their fractional amounts.

    Returns:
        The first matching rule dict if a stability issue is detected,
        otherwise None.
    """
    fa = composition.get("FA", 0.0)
    cs = composition.get("Cs", 0.0)
    br = composition.get("Br", 0.0)

    if fa > 0.85:
        return PHASE_STABILITY_RULES[0]

    if cs < 0.05 and br < 0.1:
        return PHASE_STABILITY_RULES[1]

    return None


def check_simplex_constraint(
    composition: dict[str, float],
    tolerance: float = 0.01,
) -> bool:
    """Check whether composition fractions sum to 1.0 within tolerance.

    Args:
        composition: dict mapping component names to fractional amounts.
        tolerance: maximum allowed deviation from 1.0.

    Returns:
        True if the sum is within [1.0 - tolerance, 1.0 + tolerance].
    """
    total = sum(composition.values())
    return abs(total - 1.0) <= tolerance


def get_perovskite_rules() -> dict[str, Any]:
    """Return a dict mapping rule names to their validation functions."""
    return {
        "validate_pce_physics": validate_pce_physics,
        "check_phase_stability": check_phase_stability,
        "check_simplex_constraint": check_simplex_constraint,
    }


# -- Convenience Accessor ---------------------------------------------------

def get_perovskite_config() -> dict[str, Any]:
    """Return full perovskite domain configuration as a single dict."""
    return {
        "instrument": DEFAULT_PEROVSKITE_INSTRUMENTS,
        "physical_constraints": PEROVSKITE_PHYSICAL_CONSTRAINTS,
        "quality_thresholds": PEROVSKITE_QUALITY_THRESHOLDS,
        "phase_stability_rules": PHASE_STABILITY_RULES,
        "rules": get_perovskite_rules(),
    }
