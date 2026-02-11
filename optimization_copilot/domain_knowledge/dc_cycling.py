"""DC cycling (coulombic efficiency) domain configuration.

Instrument specs, physical constraints, and quality thresholds for
galvanostatic cycling experiments (e.g. zinc electrodeposition).
"""

from __future__ import annotations

from typing import Any


# ── Instrument Specs ───────────────────────────────────────────────────

DEFAULT_DC_INSTRUMENT: dict[str, Any] = {
    "model": "Squidstat Plus",
    "dc": {
        "current_accuracy_a": 1e-7,        # ± 100 nA
        "voltage_accuracy_v": 1e-4,        # ± 0.1 mV
        "sampling_rate_hz": 100,
        "zero_drift_a_per_hour": 5e-7,     # zero-point drift
    },
}


# ── Physical Constraints ───────────────────────────────────────────────

DC_PHYSICAL_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "CE": {
        "min": 0,
        "max": 105,          # CE > 105 % is physically unreasonable
        "unit": "%",
        "warning_range": (95, 105),
    },
    "capacity": {
        "min": 0,
        "unit": "mAh",
    },
}


# ── Quality Thresholds ────────────────────────────────────────────────

DC_QUALITY_THRESHOLDS: dict[str, float] = {
    "confidence_min": 0.5,
    "relative_uncertainty_max": 0.5,
    "min_points_for_integration": 10,
}


# ── Convenience Accessors ─────────────────────────────────────────────

def get_dc_config() -> dict[str, Any]:
    """Return full DC cycling domain configuration."""
    return {
        "instrument": DEFAULT_DC_INSTRUMENT,
        "physical_constraints": DC_PHYSICAL_CONSTRAINTS,
        "quality_thresholds": DC_QUALITY_THRESHOLDS,
    }


def get_dc_instrument_spec(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract DC instrument spec."""
    if config is None:
        config = DEFAULT_DC_INSTRUMENT
    return config.get("dc", {})


def get_zero_drift(config: dict[str, Any] | None = None) -> float:
    """Return zero-point drift in A/hour."""
    spec = get_dc_instrument_spec(config)
    return spec.get("zero_drift_a_per_hour", 5e-7)


def get_current_accuracy(config: dict[str, Any] | None = None) -> float:
    """Return current measurement accuracy in amperes."""
    spec = get_dc_instrument_spec(config)
    return spec.get("current_accuracy_a", 1e-7)
