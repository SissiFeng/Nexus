"""Multi-fidelity optimization infrastructure.

Manages optimization across multiple fidelity levels (e.g., coarse DFT,
fine DFT, experimental) to balance exploration cost with result quality.

Strategy:
1. Low fidelity for fast space exploration
2. Promote promising candidates to higher fidelity
3. Dynamic fidelity selection based on budget and results

References:
- Multi-fidelity Bayesian optimization (Kandasamy et al.)
- Continuous multi-fidelity BO with knowledge gradient
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FidelityLevel:
    """Fidelity level definition.

    Attributes:
        level: Fidelity level index (0 = lowest, N = highest).
        name: Human-readable name (e.g. "coarse_dft", "experimental").
        cost_multiplier: Cost relative to lowest fidelity (lowest = 1.0).
        correlation: Expected correlation with highest fidelity results.
    """

    level: int
    name: str
    cost_multiplier: float
    correlation: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "level": self.level,
            "name": self.name,
            "cost_multiplier": self.cost_multiplier,
            "correlation": self.correlation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FidelityLevel:
        """Deserialize from dictionary."""
        return cls(
            level=data["level"],
            name=data["name"],
            cost_multiplier=data["cost_multiplier"],
            correlation=data.get("correlation", 0.8),
        )


class MultiFidelityManager:
    """Multi-fidelity optimization manager.

    Coordinates optimization across multiple fidelity levels to reduce
    overall experimental cost while maintaining result quality.

    Strategy:
    1. Low fidelity for fast space exploration
    2. Promote promising candidates to higher fidelity
    3. Dynamic fidelity selection based on budget and results

    Usage::

        levels = [
            FidelityLevel(0, "coarse_dft", 1.0, 0.6),
            FidelityLevel(1, "fine_dft", 5.0, 0.9),
            FidelityLevel(2, "experimental", 50.0, 1.0),
        ]
        manager = MultiFidelityManager(levels)

        fidelity = manager.suggest_fidelity(candidate, budget_remaining=100.0)
        manager.add_observation({"params": {...}, "objective": 0.85}, fidelity.level)
    """

    def __init__(self, fidelity_levels: list[FidelityLevel]):
        """Initialize with a list of fidelity levels.

        Args:
            fidelity_levels: List of FidelityLevel definitions.
                Will be sorted by level internally.

        Raises:
            ValueError: If no fidelity levels are provided or levels
                contain duplicates.
        """
        if not fidelity_levels:
            raise ValueError("At least one fidelity level is required")

        self._levels = sorted(fidelity_levels, key=lambda f: f.level)

        # Check for duplicate levels
        level_ids = [f.level for f in self._levels]
        if len(level_ids) != len(set(level_ids)):
            raise ValueError("Duplicate fidelity level indices detected")

        self._observations_by_fidelity: dict[int, list[dict]] = {
            f.level: [] for f in self._levels
        }

    @property
    def highest_fidelity(self) -> FidelityLevel:
        """Return the highest fidelity level."""
        return self._levels[-1]

    @property
    def lowest_fidelity(self) -> FidelityLevel:
        """Return the lowest fidelity level."""
        return self._levels[0]

    @property
    def n_levels(self) -> int:
        """Return the number of fidelity levels."""
        return len(self._levels)

    def suggest_fidelity(
        self,
        candidate: dict[str, Any],
        budget_remaining: float | None = None,
    ) -> FidelityLevel:
        """Suggest the optimal fidelity level for evaluating a candidate.

        Decision logic:
        - If budget is tight (can't afford next level), stay at lowest.
        - If candidate has no nearby low-fidelity observations, use lowest.
        - If candidate has promising low-fidelity results, promote to next level.
        - Otherwise, use lowest fidelity for exploration.

        Args:
            candidate: Parameter dictionary for the candidate point.
            budget_remaining: Remaining evaluation budget. None = unlimited.

        Returns:
            Recommended FidelityLevel for evaluation.
        """
        # If only one level, no choice
        if self.n_levels == 1:
            return self._levels[0]

        # Tight budget: stay at lowest fidelity
        if budget_remaining is not None:
            # Can we afford anything above lowest?
            next_level = self._next_fidelity(self.lowest_fidelity.level)
            if next_level is not None and budget_remaining < next_level.cost_multiplier:
                return self.lowest_fidelity

        # Check if candidate has been observed at any fidelity
        # Start from lowest and work up
        for level_obj in self._levels:
            nearby = self._find_nearby_observations(candidate, level_obj.level)
            if not nearby:
                # No observations at this level or below; evaluate at this level
                # But if this is above lowest and no low-fidelity data exists,
                # start at lowest
                if level_obj.level > self.lowest_fidelity.level:
                    return self.lowest_fidelity
                return level_obj

        # Candidate has observations at all levels -- check for promotion
        promotion_threshold = self._compute_promotion_threshold()

        # Find the highest fidelity at which we have nearby observations
        highest_observed_level = self.lowest_fidelity.level
        best_nearby_value: float | None = None

        for level_obj in self._levels:
            nearby = self._find_nearby_observations(candidate, level_obj.level)
            level_only = [
                obs for obs in nearby
                if obs.get("_fidelity_level") == level_obj.level
            ]
            if level_only:
                highest_observed_level = level_obj.level
                # Track best objective value at this level
                for obs in level_only:
                    obj_val = obs.get("objective")
                    if obj_val is not None:
                        if best_nearby_value is None or obj_val > best_nearby_value:
                            best_nearby_value = obj_val

        # If the best nearby value exceeds promotion threshold, promote
        if best_nearby_value is not None and best_nearby_value >= promotion_threshold:
            next_level = self._next_fidelity(highest_observed_level)
            if next_level is not None:
                # Check budget for promotion
                if budget_remaining is None or budget_remaining >= next_level.cost_multiplier:
                    return next_level

        # Default: stay at current highest observed level or lowest
        current = self._get_level(highest_observed_level)
        return current if current is not None else self.lowest_fidelity

    def add_observation(self, observation: dict[str, Any], fidelity: int) -> None:
        """Record an observation at a specific fidelity level.

        The observation dict is augmented with a ``_fidelity_level`` key
        so it can be traced back to its source fidelity.

        Args:
            observation: Observation dictionary (should contain at least
                parameter values and an 'objective' key).
            fidelity: Fidelity level index at which this was evaluated.

        Raises:
            ValueError: If fidelity level is not recognized.
        """
        if fidelity not in self._observations_by_fidelity:
            raise ValueError(
                f"Unknown fidelity level {fidelity}. "
                f"Known levels: {list(self._observations_by_fidelity.keys())}"
            )
        # Tag the observation with its fidelity level
        obs_copy = dict(observation)
        obs_copy["_fidelity_level"] = fidelity
        self._observations_by_fidelity[fidelity].append(obs_copy)

    def build_multi_fidelity_dataset(self) -> list[dict[str, Any]]:
        """Build a unified dataset with fidelity-based weights.

        Each observation is augmented with a ``_weight`` field based on the
        correlation of its fidelity level with the highest fidelity.
        Higher fidelity observations get higher weight (closer to 1.0).

        Returns:
            List of observation dicts, each with ``_fidelity_level``
            and ``_weight`` fields added.
        """
        dataset: list[dict[str, Any]] = []
        for level_obj in self._levels:
            weight = level_obj.correlation
            for obs in self._observations_by_fidelity[level_obj.level]:
                obs_copy = dict(obs)
                obs_copy["_fidelity_level"] = level_obj.level
                obs_copy["_weight"] = weight
                dataset.append(obs_copy)
        return dataset

    def get_observations(self, fidelity: int | None = None) -> list[dict[str, Any]]:
        """Get observations, optionally filtered by fidelity level.

        Args:
            fidelity: If provided, return only observations at this level.
                If None, return all observations across all levels.

        Returns:
            List of observation dictionaries.
        """
        if fidelity is not None:
            return list(self._observations_by_fidelity.get(fidelity, []))

        all_obs: list[dict[str, Any]] = []
        for level_obj in self._levels:
            all_obs.extend(self._observations_by_fidelity[level_obj.level])
        return all_obs

    def promotion_candidates(
        self, top_fraction: float = 0.2
    ) -> list[dict[str, Any]]:
        """Identify candidates from lower fidelities that merit promotion.

        Selects the top ``top_fraction`` of observations at each fidelity
        level (below the highest) whose objective value exceeds the
        promotion threshold.

        Args:
            top_fraction: Fraction of top-performing observations to
                consider (default 0.2 = top 20%).

        Returns:
            List of observation dicts that should be evaluated at the
            next higher fidelity level.
        """
        if self.n_levels < 2:
            return []

        promotion_threshold = self._compute_promotion_threshold()
        candidates: list[dict[str, Any]] = []

        # Check all levels except the highest
        for level_obj in self._levels[:-1]:
            obs_list = self._observations_by_fidelity[level_obj.level]
            if not obs_list:
                continue

            # Filter to those with objective values
            scored = [
                obs for obs in obs_list
                if obs.get("objective") is not None
            ]
            if not scored:
                continue

            # Sort by objective (descending = best first)
            scored.sort(key=lambda o: o.get("objective", float("-inf")), reverse=True)

            # Take top fraction
            n_top = max(1, int(math.ceil(len(scored) * top_fraction)))
            top_obs = scored[:n_top]

            # Filter by promotion threshold
            for obs in top_obs:
                obj_val = obs.get("objective", float("-inf"))
                if obj_val >= promotion_threshold:
                    # Check if already evaluated at next level
                    next_level = self._next_fidelity(level_obj.level)
                    if next_level is not None:
                        already_promoted = self._find_nearby_observations(
                            obs, next_level.level
                        )
                        # Only include if not already at next level
                        next_only = [
                            o for o in already_promoted
                            if o.get("_fidelity_level") == next_level.level
                        ]
                        if not next_only:
                            candidates.append(obs)

        return candidates

    def fidelity_summary(self) -> dict[str, Any]:
        """Generate summary statistics for the multi-fidelity campaign.

        Returns:
            Dictionary with counts per fidelity level, total cost
            estimate, and overall statistics.
        """
        per_level: dict[str, Any] = {}
        total_cost = 0.0
        total_observations = 0

        for level_obj in self._levels:
            n_obs = len(self._observations_by_fidelity[level_obj.level])
            level_cost = n_obs * level_obj.cost_multiplier
            total_cost += level_cost
            total_observations += n_obs

            # Compute objective stats for this level
            obj_values = [
                obs.get("objective")
                for obs in self._observations_by_fidelity[level_obj.level]
                if obs.get("objective") is not None
            ]
            level_stats: dict[str, Any] = {
                "name": level_obj.name,
                "n_observations": n_obs,
                "cost_multiplier": level_obj.cost_multiplier,
                "estimated_cost": level_cost,
                "correlation": level_obj.correlation,
            }
            if obj_values:
                level_stats["best_objective"] = max(obj_values)
                level_stats["mean_objective"] = sum(obj_values) / len(obj_values)
            per_level[str(level_obj.level)] = level_stats

        return {
            "n_levels": self.n_levels,
            "total_observations": total_observations,
            "total_estimated_cost": total_cost,
            "per_level": per_level,
        }

    def _find_nearby_observations(
        self,
        candidate: dict[str, Any],
        max_fidelity: int,
    ) -> list[dict[str, Any]]:
        """Find observations near a candidate at or below a given fidelity.

        Uses a simple parameter-matching approach: an observation is
        considered 'nearby' if all its numeric parameter values are
        within a relative tolerance of the candidate's values.

        Args:
            candidate: Parameter dictionary for the candidate point.
            max_fidelity: Maximum fidelity level to search (inclusive).

        Returns:
            List of nearby observation dicts.
        """
        tolerance = 0.1  # 10% relative tolerance
        nearby: list[dict[str, Any]] = []

        for level_obj in self._levels:
            if level_obj.level > max_fidelity:
                break
            for obs in self._observations_by_fidelity[level_obj.level]:
                if self._is_nearby(candidate, obs, tolerance):
                    nearby.append(obs)

        return nearby

    @staticmethod
    def _is_nearby(
        candidate: dict[str, Any],
        observation: dict[str, Any],
        tolerance: float,
    ) -> bool:
        """Check if an observation is near a candidate within tolerance.

        Compares numeric values in the candidate dict against the
        observation dict. Internal keys (prefixed with ``_``) are skipped.

        Args:
            candidate: Candidate parameter dict.
            observation: Observation dict (may contain extra keys).
            tolerance: Relative tolerance for numeric comparison.

        Returns:
            True if all shared numeric parameters are within tolerance.
        """
        matched_any = False
        for key, cval in candidate.items():
            if key.startswith("_"):
                continue
            if not isinstance(cval, (int, float)):
                continue

            oval = observation.get(key)
            if oval is None or not isinstance(oval, (int, float)):
                continue

            matched_any = True
            # Relative difference
            denom = max(abs(cval), abs(oval), 1e-12)
            if abs(cval - oval) / denom > tolerance:
                return False

        return matched_any

    def _compute_promotion_threshold(self) -> float:
        """Compute the objective threshold for promotion.

        Uses the 80th percentile of highest-fidelity observations.
        If no high-fidelity data is available, falls back to lower
        fidelity levels, and returns 0.0 if no data exists at all.

        Returns:
            Objective value threshold above which candidates qualify
            for promotion to the next fidelity level.
        """
        # Try highest fidelity first, then fall back to lower levels
        for level_obj in reversed(self._levels):
            obj_values = [
                obs.get("objective")
                for obs in self._observations_by_fidelity[level_obj.level]
                if obs.get("objective") is not None
            ]
            if obj_values:
                return self._percentile(obj_values, 80.0)

        return 0.0

    @staticmethod
    def _percentile(values: list[float], pct: float) -> float:
        """Compute percentile of a list of values (pure Python).

        Uses linear interpolation between closest ranks.

        Args:
            values: Non-empty list of numeric values.
            pct: Percentile to compute (0-100).

        Returns:
            The computed percentile value.
        """
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n == 1:
            return sorted_vals[0]

        # Rank (0-indexed) corresponding to this percentile
        rank = (pct / 100.0) * (n - 1)
        lower_idx = int(math.floor(rank))
        upper_idx = int(math.ceil(rank))

        if lower_idx == upper_idx:
            return sorted_vals[lower_idx]

        # Linear interpolation
        frac = rank - lower_idx
        return sorted_vals[lower_idx] * (1.0 - frac) + sorted_vals[upper_idx] * frac

    def _next_fidelity(self, current_level: int) -> FidelityLevel | None:
        """Get the next higher fidelity level.

        Args:
            current_level: Current fidelity level index.

        Returns:
            Next FidelityLevel, or None if already at highest.
        """
        for level_obj in self._levels:
            if level_obj.level > current_level:
                return level_obj
        return None

    def _get_level(self, level: int) -> FidelityLevel | None:
        """Look up a fidelity level by its index.

        Args:
            level: Fidelity level index to find.

        Returns:
            FidelityLevel if found, None otherwise.
        """
        for level_obj in self._levels:
            if level_obj.level == level:
                return level_obj
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manager state for persistence.

        Returns:
            Dictionary representation of the full manager state,
            including levels, observations, and summary.
        """
        return {
            "levels": [f.to_dict() for f in self._levels],
            "observations": {
                str(level): [dict(obs) for obs in obs_list]
                for level, obs_list in self._observations_by_fidelity.items()
            },
            "summary": self.fidelity_summary(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiFidelityManager:
        """Deserialize from dictionary.

        Args:
            data: Dictionary produced by ``to_dict()``.

        Returns:
            Restored MultiFidelityManager instance with observations.
        """
        levels = [FidelityLevel.from_dict(d) for d in data["levels"]]
        manager = cls(levels)

        # Restore observations
        obs_data = data.get("observations", {})
        for level_str, obs_list in obs_data.items():
            level = int(level_str)
            if level in manager._observations_by_fidelity:
                manager._observations_by_fidelity[level] = [
                    dict(obs) for obs in obs_list
                ]

        return manager
