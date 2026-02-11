"""XRD (X-Ray Diffraction) domain configuration.

Instrument specs, physical constraints, and Scherrer equation parameters
for crystallite size determination.
"""

from __future__ import annotations

from typing import Any


# ── Instrument Specs ───────────────────────────────────────────────────

DEFAULT_XRD_INSTRUMENT: dict[str, Any] = {
    "model": "lab XRD",
    "instrument_broadening_deg": 0.05,
    "two_theta_accuracy_deg": 0.02,
    "scherrer_k_range": (0.89, 0.94),     # shape factor range
    "wavelength_angstrom": 1.5406,          # Cu Kα
}


# ── Physical Constraints ───────────────────────────────────────────────

XRD_PHYSICAL_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "crystallite_size": {
        "min": 1,             # < 1 nm is meaningless
        "max": 1000,          # > 1 μm: Scherrer not applicable
        "unit": "nm",
    },
    "two_theta": {
        "min": 5,
        "max": 90,
        "unit": "deg",
    },
}


# ── Quality Thresholds ────────────────────────────────────────────────

XRD_QUALITY_THRESHOLDS: dict[str, float] = {
    "confidence_min": 0.5,
    "relative_uncertainty_max": 0.5,
    "min_peak_snr": 3.0,       # minimum signal-to-noise ratio
}


# ── Convenience Accessors ─────────────────────────────────────────────

def get_xrd_config() -> dict[str, Any]:
    """Return full XRD domain configuration."""
    return {
        "instrument": DEFAULT_XRD_INSTRUMENT,
        "physical_constraints": XRD_PHYSICAL_CONSTRAINTS,
        "quality_thresholds": XRD_QUALITY_THRESHOLDS,
    }


def get_scherrer_k_range(config: dict[str, Any] | None = None) -> tuple[float, float]:
    """Return the Scherrer shape factor (K) range."""
    if config is None:
        config = DEFAULT_XRD_INSTRUMENT
    k_range = config.get("scherrer_k_range", (0.89, 0.94))
    return (k_range[0], k_range[1])


def get_instrument_broadening(config: dict[str, Any] | None = None) -> float:
    """Return instrument broadening in degrees."""
    if config is None:
        config = DEFAULT_XRD_INSTRUMENT
    return config.get("instrument_broadening_deg", 0.05)


def get_wavelength(config: dict[str, Any] | None = None) -> float:
    """Return X-ray wavelength in angstroms."""
    if config is None:
        config = DEFAULT_XRD_INSTRUMENT
    return config.get("wavelength_angstrom", 1.5406)
