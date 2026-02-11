"""EIS (Electrochemical Impedance Spectroscopy) domain configuration.

All instrument specs, physical constraints, and equivalent circuit models
are expressed as plain Python dicts to maintain zero external dependencies.
"""

from __future__ import annotations

from typing import Any


# ── Instrument Specs ───────────────────────────────────────────────────

DEFAULT_EIS_INSTRUMENT: dict[str, Any] = {
    "model": "Squidstat Plus",
    "impedance": {
        "z_relative_accuracy": 0.001,        # |Z| ± 0.1 %
        "phase_accuracy_deg": 0.1,           # phase ± 0.1°
        "freq_range_hz": (0.1, 100_000),
        "low_freq_noise_amplification": {
            "below_1hz": 3.0,
            "below_10hz": 1.5,
        },
    },
}


# ── Physical Constraints ───────────────────────────────────────────────

EIS_PHYSICAL_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "z_magnitude": {
        "min": 0,
        "unit": "ohm",
    },
    "Rct": {
        "min": 0,
        "unit": "ohm",
        "typical_range": (1, 10_000),
    },
}


# ── Equivalent Circuit Models ─────────────────────────────────────────

DEFAULT_EIS_CIRCUITS: list[dict[str, Any]] = [
    {
        "name": "randles",
        "formula": "R_s + (R_ct || C_dl)",
        "params": ["R_s", "R_ct", "C_dl"],
        "rct_index": 1,
        "init_bounds": {
            "R_s": (0.1, 100),
            "R_ct": (1, 50_000),
            "C_dl": (1e-8, 1e-3),
        },
    },
    {
        "name": "randles_warburg",
        "formula": "R_s + (R_ct || C_dl) + W",
        "params": ["R_s", "R_ct", "C_dl", "W_s", "W_n"],
        "rct_index": 1,
        "init_bounds": {
            "R_s": (0.1, 100),
            "R_ct": (1, 50_000),
            "C_dl": (1e-8, 1e-3),
            "W_s": (0.1, 1000),
            "W_n": (0.3, 0.7),
        },
    },
    {
        "name": "2rc",
        "formula": "R_s + (R1 || C1) + (R2 || C2)",
        "params": ["R_s", "R1", "C1", "R2", "C2"],
        "rct_index": 1,
        "init_bounds": {
            "R_s": (0.1, 100),
            "R1": (1, 50_000),
            "C1": (1e-8, 1e-3),
            "R2": (1, 50_000),
            "C2": (1e-8, 1e-3),
        },
    },
]

DEFAULT_MODEL_SELECTION: dict[str, Any] = {
    "method": "aic_weighted_ensemble",
    "min_models_for_ensemble": 2,
    "convergence_max_iter": 200,
}


# ── Quality Thresholds ────────────────────────────────────────────────

EIS_QUALITY_THRESHOLDS: dict[str, float] = {
    "confidence_min": 0.5,
    "relative_uncertainty_max": 0.5,
}


# ── Convenience Accessors ─────────────────────────────────────────────

def get_eis_config() -> dict[str, Any]:
    """Return full EIS domain configuration as a single dict."""
    return {
        "instrument": DEFAULT_EIS_INSTRUMENT,
        "physical_constraints": EIS_PHYSICAL_CONSTRAINTS,
        "circuits": DEFAULT_EIS_CIRCUITS,
        "model_selection": DEFAULT_MODEL_SELECTION,
        "quality_thresholds": EIS_QUALITY_THRESHOLDS,
    }


def get_instrument_impedance_spec(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract impedance instrument spec from a config dict."""
    if config is None:
        config = DEFAULT_EIS_INSTRUMENT
    return config.get("impedance", {})


def get_noise_amplification(freq: float, config: dict[str, Any] | None = None) -> float:
    """Return the noise amplification factor for a given frequency."""
    spec = get_instrument_impedance_spec(config)
    amp = spec.get("low_freq_noise_amplification", {})
    if freq < 1.0:
        return amp.get("below_1hz", 3.0)
    if freq < 10.0:
        return amp.get("below_10hz", 1.5)
    return 1.0


def get_circuit_bounds(circuit_name: str) -> dict[str, tuple[float, float]] | None:
    """Return init_bounds for a named circuit model."""
    for c in DEFAULT_EIS_CIRCUITS:
        if c["name"] == circuit_name:
            return c["init_bounds"]
    return None
