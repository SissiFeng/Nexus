"""Safety boundary learning from successful observations.

Learns conservative parameter bounds from the successful subset of
campaign history using quantile-based estimation with configurable
margins.  Provides utilities to check and clamp candidates against
discovered boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.models import (
    CampaignSnapshot,
    ParameterSpec,
    VariableType,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SafetyBoundary:
    """Learned safe operating region for each parameter."""

    parameter_bounds: dict[str, tuple[float, float]]
    # param_name -> (safe_lower, safe_upper)

    n_successful: int

    coverage: float
    # average fraction of spec range covered by safe bounds

    per_parameter_quantiles: dict[str, tuple[float, float]]
    # param_name -> (q_low, q_high) raw quantile values

    tightening_ratios: dict[str, float]
    # param_name -> ratio of safe range to spec range


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------

class SafetyBoundaryLearner:
    """Learns safety boundaries from successful campaign observations.

    For each numeric parameter the learner computes configurable quantiles
    over the successful observations, applies a margin, and clips to the
    parameter specification range.

    Parameters
    ----------
    lower_quantile : float
        Lower quantile (e.g. 0.05 for 5th percentile).
    upper_quantile : float
        Upper quantile (e.g. 0.95 for 95th percentile).
    min_observations : int
        Minimum successful observations required before tightening.
    margin_fraction : float
        Fraction of the quantile range to add as margin on each side.
    """

    def __init__(
        self,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95,
        min_observations: int = 5,
        margin_fraction: float = 0.05,
    ) -> None:
        self._lower_quantile = lower_quantile
        self._upper_quantile = upper_quantile
        self._min_observations = min_observations
        self._margin_fraction = margin_fraction

    def learn(self, snapshot: CampaignSnapshot) -> SafetyBoundary:
        """Learn safety boundaries from campaign history.

        Parameters
        ----------
        snapshot :
            Current campaign state.

        Returns
        -------
        SafetyBoundary
        """
        successes = snapshot.successful_observations
        n_successful = len(successes)

        parameter_bounds: dict[str, tuple[float, float]] = {}
        per_parameter_quantiles: dict[str, tuple[float, float]] = {}
        tightening_ratios: dict[str, float] = {}

        numeric_specs = [
            spec for spec in snapshot.parameter_specs
            if spec.type in (VariableType.CONTINUOUS, VariableType.DISCRETE)
        ]

        # Not enough data — return spec bounds as-is (no tightening).
        if n_successful < self._min_observations:
            for spec in numeric_specs:
                lo = spec.lower if spec.lower is not None else 0.0
                hi = spec.upper if spec.upper is not None else 1.0
                parameter_bounds[spec.name] = (lo, hi)
                tightening_ratios[spec.name] = 1.0
            return SafetyBoundary(
                parameter_bounds=parameter_bounds,
                n_successful=n_successful,
                coverage=1.0,
                per_parameter_quantiles=per_parameter_quantiles,
                tightening_ratios=tightening_ratios,
            )

        # Enough data — compute quantile-based bounds.
        for spec in numeric_specs:
            spec_lo = spec.lower if spec.lower is not None else 0.0
            spec_hi = spec.upper if spec.upper is not None else 1.0
            spec_range = spec_hi - spec_lo

            # Collect parameter values from successful observations.
            values: list[float] = []
            for obs in successes:
                v = obs.parameters.get(spec.name)
                if v is not None:
                    try:
                        values.append(float(v))
                    except (TypeError, ValueError):
                        pass

            if not values:
                parameter_bounds[spec.name] = (spec_lo, spec_hi)
                tightening_ratios[spec.name] = 1.0
                continue

            sorted_vals = sorted(values)
            q_low = self._quantile(sorted_vals, self._lower_quantile)
            q_high = self._quantile(sorted_vals, self._upper_quantile)
            per_parameter_quantiles[spec.name] = (q_low, q_high)

            margin = self._margin_fraction * (q_high - q_low)
            safe_lower = max(spec_lo, q_low - margin)
            safe_upper = min(spec_hi, q_high + margin)

            # Fallback: if bounds collapsed, revert to full spec range.
            if safe_lower >= safe_upper:
                safe_lower = spec_lo
                safe_upper = spec_hi

            parameter_bounds[spec.name] = (safe_lower, safe_upper)

            if spec_range > 0:
                tightening_ratios[spec.name] = (safe_upper - safe_lower) / spec_range
            else:
                tightening_ratios[spec.name] = 1.0

        # Coverage is the mean tightening ratio.
        if tightening_ratios:
            coverage = sum(tightening_ratios.values()) / len(tightening_ratios)
        else:
            coverage = 1.0

        return SafetyBoundary(
            parameter_bounds=parameter_bounds,
            n_successful=n_successful,
            coverage=coverage,
            per_parameter_quantiles=per_parameter_quantiles,
            tightening_ratios=tightening_ratios,
        )

    def is_within_bounds(
        self,
        candidate: dict[str, Any],
        boundary: SafetyBoundary,
    ) -> bool:
        """Check whether a candidate falls within all safety bounds.

        Parameters
        ----------
        candidate :
            Parameter values to check.
        boundary :
            Previously learned safety boundary.

        Returns
        -------
        bool
        """
        for param, (lo, hi) in boundary.parameter_bounds.items():
            if param in candidate:
                val = candidate[param]
                if not (lo <= val <= hi):
                    return False
        return True

    def clamp_to_bounds(
        self,
        candidate: dict[str, Any],
        boundary: SafetyBoundary,
    ) -> dict[str, Any]:
        """Clamp candidate parameter values to the safety bounds.

        Non-numeric values are left unchanged.

        Parameters
        ----------
        candidate :
            Parameter values to clamp.
        boundary :
            Previously learned safety boundary.

        Returns
        -------
        dict[str, Any]
            Shallow copy with clamped values.
        """
        result = dict(candidate)
        for param, (lo, hi) in boundary.parameter_bounds.items():
            if param in result and isinstance(result[param], (int, float)):
                result[param] = max(lo, min(hi, result[param]))
        return result

    # -- Private helpers ----------------------------------------------------

    @staticmethod
    def _quantile(sorted_values: list[float], q: float) -> float:
        """Compute the *q*-th quantile of pre-sorted values.

        Uses linear interpolation between adjacent elements (matching
        the ``numpy.percentile`` default ``method='linear'``).

        Parameters
        ----------
        sorted_values :
            Values in ascending order.
        q :
            Quantile in [0, 1].

        Returns
        -------
        float
        """
        n = len(sorted_values)
        if n == 0:
            return 0.0
        if n == 1:
            return sorted_values[0]
        pos = q * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac
