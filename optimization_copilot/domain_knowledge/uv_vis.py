"""UV-Vis spectrophotometry domain configuration.

Instrument specs, physical constraints, and absorbance noise models
for UV-Vis measurements (e.g. additive concentration monitoring).
"""

from __future__ import annotations

from typing import Any


# ── Instrument Specs ───────────────────────────────────────────────────

DEFAULT_UVVIS_INSTRUMENT: dict[str, Any] = {
    "model": "Ocean Optics",
    "wavelength_range_nm": (200, 900),
    "absorbance_noise": {
        "low_abs": 0.002,       # A < 0.5
        "mid_abs": 0.005,       # 0.5 < A < 2.0
        "high_abs": 0.02,       # A > 2.0 (S/N drops sharply)
    },
    "linear_range_max": 2.5,   # Beer-Lambert linear range ceiling
}


# ── Physical Constraints ───────────────────────────────────────────────

UVVIS_PHYSICAL_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "absorbance": {
        "min": -0.05,          # small negative = baseline issue
        "max": 4.0,            # above 4.0 is meaningless
        "unit": "AU",
        "negative_confidence_penalty": 0.3,
    },
    "concentration": {
        "min": 0,
        "unit": "mol/L",
    },
}


# ── Quality Thresholds ────────────────────────────────────────────────

UVVIS_QUALITY_THRESHOLDS: dict[str, float] = {
    "confidence_min": 0.5,
    "relative_uncertainty_max": 0.5,
}


# ── Convenience Accessors ─────────────────────────────────────────────

def get_uvvis_config() -> dict[str, Any]:
    """Return full UV-Vis domain configuration."""
    return {
        "instrument": DEFAULT_UVVIS_INSTRUMENT,
        "physical_constraints": UVVIS_PHYSICAL_CONSTRAINTS,
        "quality_thresholds": UVVIS_QUALITY_THRESHOLDS,
    }


def get_absorbance_noise(absorbance: float, config: dict[str, Any] | None = None) -> float:
    """Return the expected absorbance noise σ for a given absorbance value."""
    if config is None:
        config = DEFAULT_UVVIS_INSTRUMENT
    noise_spec = config.get("absorbance_noise", {})
    if absorbance < 0.5:
        return noise_spec.get("low_abs", 0.002)
    if absorbance < 2.0:
        return noise_spec.get("mid_abs", 0.005)
    return noise_spec.get("high_abs", 0.02)


def get_linear_range_max(config: dict[str, Any] | None = None) -> float:
    """Return Beer-Lambert linear range ceiling."""
    if config is None:
        config = DEFAULT_UVVIS_INSTRUMENT
    return config.get("linear_range_max", 2.5)
