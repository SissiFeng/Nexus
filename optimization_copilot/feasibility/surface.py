"""Failure surface learning and avoidance.

Learns a failure probability surface from campaign history using KNN
and rule-based methods.  Provides:
- ``FailureProbability`` estimates for candidate points
- Safe-bound estimation per parameter
- Danger-zone identification
- Risk-adjusted candidate scoring (objective - lambda * p_fail)

All methods are stdlib-only (no numpy/scipy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Observation,
    ParameterSpec,
    RiskPosture,
    VariableType,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FailureProbability:
    """Failure probability estimate for a single candidate point."""

    p_fail: float  # overall failure probability [0, 1]
    p_by_type: dict[str, float] = field(default_factory=dict)  # per failure type
    n_neighbors: int = 0
    avg_distance: float = 0.0


@dataclass
class DangerZone:
    """A region of parameter space with elevated failure rate."""

    parameter: str
    bound_type: str  # "above" or "below"
    threshold: float
    failure_rate: float  # failure rate in the danger zone
    n_samples: int  # support count


@dataclass
class FailureSurface:
    """Learned failure surface from campaign history."""

    safe_bounds: dict[str, tuple[float, float]]
    # {param_name: (safe_lower, safe_upper)}

    danger_zones: list[DangerZone]

    parameter_failure_density: dict[str, list[float]]
    # {param_name: [density_per_bin]} — failure density histogram

    n_observations: int
    n_failures: int
    overall_failure_rate: float


@dataclass
class FailureAdjustment:
    """Failure-type-specific strategy adjustment."""

    adjustment_type: str  # "reduce_exploration", "tighten_bounds", etc.
    parameters: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------

class FailureSurfaceLearner:
    """Learns failure surfaces from campaign data.

    Parameters
    ----------
    k : int
        Number of neighbors for KNN failure probability estimation.
    n_bins : int
        Number of bins per parameter for failure density histograms.
    danger_zone_threshold : float
        Minimum failure rate to flag a region as dangerous.
    min_samples_for_zone : int
        Minimum observations in a zone to be considered.
    """

    def __init__(
        self,
        k: int = 5,
        n_bins: int = 10,
        danger_zone_threshold: float = 0.4,
        min_samples_for_zone: int = 3,
    ) -> None:
        self.k = k
        self.n_bins = n_bins
        self.danger_zone_threshold = danger_zone_threshold
        self.min_samples_for_zone = min_samples_for_zone

    def learn(self, snapshot: CampaignSnapshot) -> FailureSurface:
        """Learn failure surface from campaign history.

        Returns a ``FailureSurface`` with safe bounds, danger zones,
        and failure density per parameter.
        """
        obs = snapshot.observations
        specs = snapshot.parameter_specs
        n_obs = len(obs)
        n_fail = sum(1 for o in obs if o.is_failure)
        overall_rate = n_fail / max(n_obs, 1)

        safe_bounds = self._compute_safe_bounds(obs, specs)
        danger_zones = self._find_danger_zones(obs, specs)
        density = self._compute_failure_density(obs, specs)

        return FailureSurface(
            safe_bounds=safe_bounds,
            danger_zones=danger_zones,
            parameter_failure_density=density,
            n_observations=n_obs,
            n_failures=n_fail,
            overall_failure_rate=overall_rate,
        )

    def predict_failure(
        self,
        candidate: dict[str, Any],
        snapshot: CampaignSnapshot,
        failure_taxonomy: Any | None = None,
    ) -> FailureProbability:
        """Estimate failure probability for a candidate point via KNN.

        Parameters
        ----------
        candidate :
            Parameter values for the candidate point.
        snapshot :
            Campaign history to learn from.
        failure_taxonomy :
            Optional taxonomy for per-type probability.

        Returns
        -------
        FailureProbability
        """
        obs = snapshot.observations
        specs = snapshot.parameter_specs

        if len(obs) < 2:
            return FailureProbability(
                p_fail=snapshot.failure_rate if obs else 0.0,
                n_neighbors=len(obs),
                avg_distance=0.0,
            )

        # Find k nearest neighbors.
        distances: list[tuple[float, Observation]] = []
        for o in obs:
            d = _normalized_distance(candidate, o.parameters, specs)
            distances.append((d, o))

        distances.sort(key=lambda x: x[0])
        k = min(self.k, len(distances))
        neighbors = distances[:k]

        n_fail = sum(1 for _, o in neighbors if o.is_failure)
        p_fail = n_fail / max(k, 1)
        avg_dist = sum(d for d, _ in neighbors) / max(k, 1)

        # Per-type breakdown if taxonomy available.
        p_by_type: dict[str, float] = {}
        if failure_taxonomy is not None:
            classified = getattr(failure_taxonomy, "classified_failures", [])
            fail_indices = {cf.observation_index for cf in classified}
            type_counts: dict[str, int] = {}

            for _, o in neighbors:
                if o.is_failure:
                    idx = o.iteration
                    for cf in classified:
                        if cf.observation_index == idx:
                            t = cf.failure_type.value if hasattr(cf.failure_type, "value") else str(cf.failure_type)
                            type_counts[t] = type_counts.get(t, 0) + 1
                            break
                    else:
                        type_counts["unknown"] = type_counts.get("unknown", 0) + 1

            for t, c in type_counts.items():
                p_by_type[t] = c / max(k, 1)

        return FailureProbability(
            p_fail=round(p_fail, 4),
            p_by_type=p_by_type,
            n_neighbors=k,
            avg_distance=round(avg_dist, 4),
        )

    def adjust_score(
        self,
        objective_score: float,
        failure_prob: FailureProbability,
        risk_posture: RiskPosture = RiskPosture.MODERATE,
    ) -> float:
        """Risk-adjusted score: objective - lambda * p_fail.

        Lambda is modulated by risk posture:
        - CONSERVATIVE: lambda = 1.0 (strong avoidance)
        - MODERATE: lambda = 0.5
        - AGGRESSIVE: lambda = 0.2 (willing to risk)
        """
        lambda_map = {
            RiskPosture.CONSERVATIVE: 1.0,
            RiskPosture.MODERATE: 0.5,
            RiskPosture.AGGRESSIVE: 0.2,
        }
        lam = lambda_map.get(risk_posture, 0.5)
        return objective_score - lam * failure_prob.p_fail

    def recommend_adjustments(
        self,
        failure_taxonomy: Any | None = None,
        failure_surface: FailureSurface | None = None,
    ) -> list[FailureAdjustment]:
        """Recommend strategy adjustments based on failure patterns.

        Returns type-specific actions:
        - HARDWARE → reduce exploration, lower parallelism
        - CHEMISTRY → tighten bounds from danger zones
        - DATA → add QC gate, increase replicates
        - PROTOCOL → enforce protocol checks
        """
        adjustments: list[FailureAdjustment] = []

        if failure_taxonomy is not None:
            dominant = getattr(failure_taxonomy, "dominant_type", None)
            dtype = dominant.value if dominant and hasattr(dominant, "value") else None

            if dtype == "hardware":
                adjustments.append(FailureAdjustment(
                    adjustment_type="reduce_exploration",
                    parameters={"exploration_reduction": 0.2, "max_batch_size": 2},
                    reason="Hardware failures → reduce exploration, smaller batches",
                ))
            elif dtype == "chemistry":
                # Use danger zones to tighten bounds.
                bounds_to_tighten = {}
                if failure_surface is not None:
                    for dz in failure_surface.danger_zones:
                        if dz.parameter not in bounds_to_tighten:
                            bounds_to_tighten[dz.parameter] = {
                                "bound_type": dz.bound_type,
                                "threshold": dz.threshold,
                            }
                adjustments.append(FailureAdjustment(
                    adjustment_type="tighten_bounds",
                    parameters={"bounds": bounds_to_tighten},
                    reason="Chemistry failures → constrain search to safe region",
                ))
            elif dtype == "data":
                adjustments.append(FailureAdjustment(
                    adjustment_type="increase_replicates",
                    parameters={"min_replicates": 2, "qc_gate": True},
                    reason="Data quality issues → QC gate + replication",
                ))
            elif dtype == "protocol":
                adjustments.append(FailureAdjustment(
                    adjustment_type="enforce_protocol_checks",
                    parameters={"strict_mode": True},
                    reason="Protocol failures → stricter compliance checks",
                ))

        # General: if overall failure rate is high, add conservative adjustment.
        if failure_surface is not None and failure_surface.overall_failure_rate > 0.3:
            adjustments.append(FailureAdjustment(
                adjustment_type="conservative_exploration",
                parameters={"failure_rate": failure_surface.overall_failure_rate},
                reason=f"High failure rate ({failure_surface.overall_failure_rate:.0%}) "
                       "→ conservative exploration",
            ))

        return adjustments

    # -- Private methods --------------------------------------------------

    def _compute_safe_bounds(
        self,
        observations: list[Observation],
        specs: list[ParameterSpec],
    ) -> dict[str, tuple[float, float]]:
        """Infer safe parameter bounds from successful observations."""
        safe_bounds: dict[str, tuple[float, float]] = {}
        successes = [o for o in observations if not o.is_failure]

        for spec in specs:
            if spec.type == VariableType.CATEGORICAL:
                continue

            values = []
            for o in successes:
                v = o.parameters.get(spec.name)
                if v is not None:
                    try:
                        values.append(float(v))
                    except (TypeError, ValueError):
                        pass

            if not values:
                lo = spec.lower if spec.lower is not None else 0.0
                hi = spec.upper if spec.upper is not None else 1.0
                safe_bounds[spec.name] = (lo, hi)
                continue

            safe_lo = min(values)
            safe_hi = max(values)

            # Expand slightly (5%) to avoid being too tight.
            rng = safe_hi - safe_lo
            margin = rng * 0.05 if rng > 0 else 0.01
            safe_lo -= margin
            safe_hi += margin

            # Clip to spec bounds.
            if spec.lower is not None:
                safe_lo = max(safe_lo, spec.lower)
            if spec.upper is not None:
                safe_hi = min(safe_hi, spec.upper)

            safe_bounds[spec.name] = (round(safe_lo, 6), round(safe_hi, 6))

        return safe_bounds

    def _find_danger_zones(
        self,
        observations: list[Observation],
        specs: list[ParameterSpec],
    ) -> list[DangerZone]:
        """Find parameter regions with elevated failure rates."""
        danger_zones: list[DangerZone] = []

        for spec in specs:
            if spec.type == VariableType.CATEGORICAL:
                continue

            # Extract (value, is_failure) pairs.
            pairs: list[tuple[float, bool]] = []
            for o in observations:
                v = o.parameters.get(spec.name)
                if v is not None:
                    try:
                        pairs.append((float(v), o.is_failure))
                    except (TypeError, ValueError):
                        pass

            if len(pairs) < self.min_samples_for_zone * 2:
                continue

            # Split at median and check each half.
            pairs.sort(key=lambda x: x[0])
            mid = len(pairs) // 2
            median_val = pairs[mid][0]

            lower_half = pairs[:mid]
            upper_half = pairs[mid:]

            for half, bound_type, threshold in [
                (lower_half, "below", median_val),
                (upper_half, "above", median_val),
            ]:
                if len(half) < self.min_samples_for_zone:
                    continue
                n_fail = sum(1 for _, f in half if f)
                rate = n_fail / len(half)
                if rate >= self.danger_zone_threshold:
                    danger_zones.append(DangerZone(
                        parameter=spec.name,
                        bound_type=bound_type,
                        threshold=round(threshold, 6),
                        failure_rate=round(rate, 4),
                        n_samples=len(half),
                    ))

        return danger_zones

    def _compute_failure_density(
        self,
        observations: list[Observation],
        specs: list[ParameterSpec],
    ) -> dict[str, list[float]]:
        """Compute per-parameter failure density histograms."""
        density: dict[str, list[float]] = {}

        for spec in specs:
            if spec.type == VariableType.CATEGORICAL:
                continue

            lo = spec.lower if spec.lower is not None else 0.0
            hi = spec.upper if spec.upper is not None else 1.0
            rng = hi - lo
            if rng <= 0:
                density[spec.name] = [0.0] * self.n_bins
                continue

            bins = [0] * self.n_bins
            totals = [0] * self.n_bins

            for o in observations:
                v = o.parameters.get(spec.name)
                if v is not None:
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        continue
                    bin_idx = min(
                        int((fv - lo) / rng * self.n_bins),
                        self.n_bins - 1,
                    )
                    bin_idx = max(0, bin_idx)
                    totals[bin_idx] += 1
                    if o.is_failure:
                        bins[bin_idx] += 1

            density[spec.name] = [
                round(bins[i] / max(totals[i], 1), 4)
                for i in range(self.n_bins)
            ]

        return density


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalized_distance(
    candidate: dict[str, Any],
    obs_params: dict[str, Any],
    specs: list[ParameterSpec],
) -> float:
    """Normalised Euclidean distance between candidate and observation."""
    total = 0.0
    n_dims = 0

    for spec in specs:
        c_val = candidate.get(spec.name)
        o_val = obs_params.get(spec.name)

        if c_val is None or o_val is None:
            total += 1.0
            n_dims += 1
            continue

        if spec.type == VariableType.CATEGORICAL:
            total += 0.0 if c_val == o_val else 1.0
        else:
            lo = spec.lower if spec.lower is not None else 0.0
            hi = spec.upper if spec.upper is not None else 1.0
            rng = hi - lo if hi != lo else 1.0
            diff = (float(c_val) - float(o_val)) / rng
            total += diff * diff

        n_dims += 1

    if n_dims == 0:
        return 0.0
    return math.sqrt(total / n_dims)
