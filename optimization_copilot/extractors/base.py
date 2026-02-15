"""Abstract base class for uncertainty-aware KPI extractors.

Every extractor consumes raw measurement data (as a plain dict) and emits
a list of ``MeasurementWithUncertainty`` objects that downstream propagation
can consume.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from optimization_copilot.uncertainty.types import MeasurementWithUncertainty


class UncertaintyExtractor(ABC):
    """Base class that all domain-specific extractors inherit from.

    Parameters
    ----------
    domain_config : dict[str, Any]
        Full domain configuration dict (instrument specs, physical
        constraints, quality thresholds).
    """

    def __init__(self, domain_config: dict[str, Any]) -> None:
        self.domain_config = domain_config

    # ── Abstract interface ────────────────────────────────────────────

    @abstractmethod
    def extract_with_uncertainty(
        self, raw_data: dict[str, Any],
    ) -> list[MeasurementWithUncertainty]:
        """Extract KPI(s) with uncertainty from raw measurement data.

        Parameters
        ----------
        raw_data : dict[str, Any]
            Technique-specific raw data dictionary.

        Returns
        -------
        list[MeasurementWithUncertainty]
            One entry per extracted KPI.
        """

    # ── Shared helpers ────────────────────────────────────────────────

    def _compute_confidence(self, variance: float, value: float) -> float:
        """Map relative uncertainty to a confidence score in [0, 1].

        ``conf = max(0, min(1, 1 - sqrt(variance) / |value|))``

        When *value* is (near-)zero the relative uncertainty is infinite
        and the confidence is clamped to 0.
        """
        if abs(value) < 1e-12:
            return 0.0
        rel = math.sqrt(max(variance, 0.0)) / abs(value)
        return max(0.0, min(1.0, 1.0 - rel))

    def _apply_physical_constraints(
        self,
        measurement: MeasurementWithUncertainty,
        kpi_name: str,
        constraints: dict[str, Any],
    ) -> MeasurementWithUncertainty:
        """Adjust confidence based on physical-constraint violations.

        Checks ``min``, ``max``, and ``typical_range`` keys in
        *constraints* and penalises confidence when the value falls
        outside the physically reasonable window.

        Returns a **new** ``MeasurementWithUncertainty`` (the original
        is not mutated).
        """
        conf = measurement.confidence
        val = measurement.value

        if "min" in constraints and val < constraints["min"]:
            conf *= 0.3  # severe penalty for below-minimum
        if "max" in constraints and val > constraints["max"]:
            conf *= 0.3  # severe penalty for above-maximum

        if "typical_range" in constraints:
            lo, hi = constraints["typical_range"]
            if val < lo or val > hi:
                conf *= 0.7  # mild penalty for atypical

        # Clamp to [0, 1]
        conf = max(0.0, min(1.0, conf))

        return MeasurementWithUncertainty(
            value=measurement.value,
            variance=measurement.variance,
            confidence=conf,
            source=measurement.source,
            fit_residual=measurement.fit_residual,
            n_points_used=measurement.n_points_used,
            method=measurement.method,
            metadata=measurement.metadata,
        )
