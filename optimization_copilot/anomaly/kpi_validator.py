"""Layer 2: Physical range validation for extracted KPIs.

Checks whether KPI values fall within physically plausible bounds.
Default bounds are provided for common scientific KPIs; domain-specific
bounds can be supplied via a ``DomainConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optimization_copilot.domain_knowledge.loader import DomainConfig


# ── Data types ─────────────────────────────────────────────────────────


@dataclass
class KPIAnomaly:
    """A KPI value that violates its physical range."""

    kpi_name: str
    value: float
    expected_range: tuple[float, float]
    severity: str  # "warning", "error"
    message: str


# ── Default physical bounds ────────────────────────────────────────────

# Mapping from lowercase KPI name to (lower, upper) inclusive bounds.
# These are intentionally permissive to avoid false positives.
_DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "ce": (0.0, 105.0),                  # coulombic efficiency (%)
    "coulombic_efficiency": (0.0, 105.0),
    "rct": (0.0, float("inf")),           # charge transfer resistance (ohm)
    "charge_transfer_resistance": (0.0, float("inf")),
    "grain_size": (1.0, float("inf")),    # grain size (nm)
    "absorbance": (0.0, 5.0),            # UV-Vis absorbance (AU)
    "peak_intensity": (0.0, float("inf")),
    "conversion": (0.0, 100.0),          # catalytic conversion (%)
    "selectivity": (0.0, 100.0),         # catalytic selectivity (%)
    "yield_pct": (0.0, 100.0),           # reaction yield (%)
    "band_gap": (0.5, 6.0),             # band gap (eV)
    "pce": (0.0, 35.0),                 # power conversion efficiency (%)
    "power_conversion_efficiency": (0.0, 35.0),
}


# ── KPIValidator ───────────────────────────────────────────────────────


class KPIValidator:
    """Validate KPI values against physical bounds.

    Parameters
    ----------
    domain_config : DomainConfig | None
        Optional domain configuration.  If provided, its
        ``get_constraints()`` are checked for ``<kpi_name>_range`` keys
        of the form ``(lower, upper)`` which override the defaults.
    """

    def __init__(self, domain_config: DomainConfig | None = None) -> None:
        self._bounds: dict[str, tuple[float, float]] = dict(_DEFAULT_BOUNDS)
        if domain_config is not None:
            self._load_domain_bounds(domain_config)

    def _load_domain_bounds(self, domain_config: DomainConfig) -> None:
        """Override default bounds from domain constraints."""
        constraints = domain_config.get_constraints()
        for key, value in constraints.items():
            # Accept keys like "ce_range" -> kpi "ce"
            if key.endswith("_range") and isinstance(value, (list, tuple)):
                kpi_name = key[: -len("_range")]
                if len(value) == 2:
                    self._bounds[kpi_name] = (float(value[0]), float(value[1]))
            # Also accept direct constraint dicts with "lower" / "upper"
            elif isinstance(value, dict):
                lo = value.get("lower", value.get("min", float("-inf")))
                hi = value.get("upper", value.get("max", float("inf")))
                self._bounds[key] = (float(lo), float(hi))

    def validate(self, kpi_name: str, value: float) -> KPIAnomaly | None:
        """Check a single KPI value against physical bounds.

        Unknown KPI names pass validation (return ``None``).
        """
        key = kpi_name.lower()
        bounds = self._bounds.get(key)
        if bounds is None:
            return None

        lo, hi = bounds
        if value < lo or value > hi:
            severity = "error" if (value < lo - abs(lo) * 0.1 or value > hi + abs(hi) * 0.1) else "warning"
            # For unbounded upper/lower, simplify severity
            if hi == float("inf") and value < lo:
                severity = "error"
            elif lo == float("-inf") and value > hi:
                severity = "error"
            return KPIAnomaly(
                kpi_name=kpi_name,
                value=value,
                expected_range=(lo, hi),
                severity=severity,
                message=(
                    f"{kpi_name}={value} is outside physical range "
                    f"[{lo}, {hi}]"
                ),
            )
        return None

    def validate_all(
        self, kpi_values: dict[str, float]
    ) -> list[KPIAnomaly]:
        """Validate all KPI values and return a list of anomalies."""
        anomalies: list[KPIAnomaly] = []
        for name, value in kpi_values.items():
            result = self.validate(name, value)
            if result is not None:
                anomalies.append(result)
        return anomalies
