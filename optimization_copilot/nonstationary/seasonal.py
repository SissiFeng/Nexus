"""Seasonal pattern detection for optimization campaigns.

Uses autocorrelation analysis to identify periodic patterns in KPI
time series, enabling the optimizer to anticipate and adapt to
recurring environmental effects (e.g., day-of-week, batch cycles).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from optimization_copilot.core.models import CampaignSnapshot


# ── Data Structures ───────────────────────────────────────


@dataclass
class SeasonalPattern:
    """Result of seasonal pattern detection."""

    detected: bool
    period: int | None
    autocorrelation: float
    candidate_periods: list[int]
    candidate_correlations: list[float]
    confidence: float


# ── Helpers ───────────────────────────────────────────────


def _mean(values: list[float]) -> float:
    """Arithmetic mean.  Returns 0.0 for empty list."""
    return sum(values) / len(values) if values else 0.0


# ── Seasonal Detector ─────────────────────────────────────


class SeasonalDetector:
    """Detect periodic (seasonal) patterns in campaign KPI series.

    Uses autocorrelation at candidate lag periods to identify
    repeating cycles.  Pure Python, deterministic.

    Parameters
    ----------
    min_period : int
        Minimum candidate period (lag) to consider.
    max_period_fraction : float
        Maximum period as a fraction of total observations.
    correlation_threshold : float
        Autocorrelation above this value triggers a detection.
    min_observations : int
        Minimum number of successful observations required
        before attempting detection.
    """

    def __init__(
        self,
        min_period: int = 3,
        max_period_fraction: float = 0.4,
        correlation_threshold: float = 0.5,
        min_observations: int = 12,
    ) -> None:
        self.min_period = min_period
        self.max_period_fraction = max_period_fraction
        self.correlation_threshold = correlation_threshold
        self.min_observations = min_observations

    # ── Public API ────────────────────────────────────────

    def detect(self, snapshot: CampaignSnapshot) -> SeasonalPattern:
        """Analyse the primary KPI series for seasonal patterns.

        Extracts the first objective's values from successful observations,
        computes autocorrelation at each candidate lag, and reports whether
        a significant periodic pattern exists.
        """
        # Extract primary KPI values from successful observations.
        successful = snapshot.successful_observations
        if not snapshot.objective_names:
            return self._not_detected()

        primary_kpi = snapshot.objective_names[0]
        values = [
            obs.kpi_values[primary_kpi]
            for obs in successful
            if primary_kpi in obs.kpi_values
        ]

        if len(values) < self.min_observations:
            return self._not_detected()

        # Determine candidate period range.
        max_period = int(len(values) * self.max_period_fraction)
        if max_period < self.min_period:
            return self._not_detected()

        # Compute autocorrelation at each candidate lag.
        candidate_periods: list[int] = []
        candidate_correlations: list[float] = []

        for p in range(self.min_period, max_period + 1):
            ac = self._autocorrelation(values, p)
            candidate_periods.append(p)
            candidate_correlations.append(ac)

        if not candidate_correlations:
            return self._not_detected()

        # Find the best period.
        best_idx = max(range(len(candidate_correlations)), key=lambda i: candidate_correlations[i])
        best_period = candidate_periods[best_idx]
        max_correlation = candidate_correlations[best_idx]

        detected = max_correlation > self.correlation_threshold
        confidence = max(0.0, min(1.0, max_correlation))

        return SeasonalPattern(
            detected=detected,
            period=best_period if detected else None,
            autocorrelation=max_correlation,
            candidate_periods=candidate_periods,
            candidate_correlations=candidate_correlations,
            confidence=confidence,
        )

    # ── Private helpers ───────────────────────────────────

    def _autocorrelation(self, values: list[float], lag: int) -> float:
        """Compute the autocorrelation of *values* at the given *lag*.

        Returns 0.0 for degenerate inputs (zero variance, lag >= len).
        """
        n = len(values)
        if lag >= n:
            return 0.0

        m = _mean(values)
        denom = sum((x - m) ** 2 for x in values)
        if denom == 0.0:
            return 0.0

        num = sum((values[i] - m) * (values[i + lag] - m) for i in range(n - lag))
        return num / denom

    def _not_detected(self) -> SeasonalPattern:
        """Return a default "no seasonality detected" result."""
        return SeasonalPattern(
            detected=False,
            period=None,
            autocorrelation=0.0,
            candidate_periods=[],
            candidate_correlations=[],
            confidence=0.0,
        )
