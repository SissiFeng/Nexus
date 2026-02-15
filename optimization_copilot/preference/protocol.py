"""Multi-Objective Preference Protocol.

Applies user-specified objective preferences (weights, epsilon constraints,
scalarization methods, objective subsets) to rank and filter observations
in a multi-objective optimization campaign.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.models import CampaignSnapshot, Observation


@dataclass
class EpsilonConstraint:
    """Bound on a single objective.

    At least one of lower_bound / upper_bound should be set.  An observation
    satisfies the constraint iff its KPI value for *objective* is within the
    specified bounds (inclusive).
    """

    objective: str
    lower_bound: float | None = None
    upper_bound: float | None = None


@dataclass
class ObjectivePreferenceConfig:
    """User preferences for multi-objective optimization.

    Parameters
    ----------
    weights:
        objective_name -> weight.  Missing objectives get weight 1.0.
    epsilon_constraints:
        Hard bounds that observations must satisfy.
    objective_subset:
        If non-empty, only these objectives are used for scalarization.
        Empty means *all* objectives.
    scalarization_method:
        One of ``"weighted_sum"``, ``"tchebycheff"``, ``"achievement"``.
    """

    weights: dict[str, float] = field(default_factory=dict)
    epsilon_constraints: list[EpsilonConstraint] = field(default_factory=list)
    objective_subset: list[str] = field(default_factory=list)
    scalarization_method: str = "weighted_sum"

    # ── serialization ────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "epsilon_constraints": [
                {
                    "objective": ec.objective,
                    "lower_bound": ec.lower_bound,
                    "upper_bound": ec.upper_bound,
                }
                for ec in self.epsilon_constraints
            ],
            "objective_subset": list(self.objective_subset),
            "scalarization_method": self.scalarization_method,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectivePreferenceConfig:
        return cls(
            weights=data.get("weights", {}),
            epsilon_constraints=[
                EpsilonConstraint(**ec)
                for ec in data.get("epsilon_constraints", [])
            ],
            objective_subset=data.get("objective_subset", []),
            scalarization_method=data.get("scalarization_method", "weighted_sum"),
        )


class PreferenceProtocol:
    """Applies user preferences to multi-objective optimization.

    The protocol supports three scalarization methods:

    1. **weighted_sum** -- ``sum(w_i * v_i)`` where ``v_i`` is normalised so
       that *higher is better* for every objective.
    2. **tchebycheff** -- ``min(w_i * v_i)``.  Maximises the minimum weighted
       objective.
    3. **achievement** -- ``-max(w_i * |v_i|)``.  Minimises the maximum
       deviation from the aspiration point (origin after normalisation).
    """

    def __init__(self, config: ObjectivePreferenceConfig) -> None:
        self.config = config

    # ── snapshot integration ─────────────────────────────────

    def apply_to_snapshot(self, snapshot: CampaignSnapshot) -> CampaignSnapshot:
        """Store preference config in *snapshot.metadata*."""
        snapshot.metadata["preference_config"] = self.config.to_dict()
        return snapshot

    # ── scalarization ────────────────────────────────────────

    def compute_scalarized_score(
        self,
        observation: Observation,
        objective_names: list[str],
        objective_directions: list[str],
    ) -> float:
        """Return a scalar score for *observation* under the current config.

        Normalisation: ``"minimize"`` objectives are negated so that
        *higher is always better* across all objectives.  Weights default to
        ``1.0`` for any objective not explicitly listed and are normalised to
        sum to ``1``.
        """
        # Determine active objectives
        active_objectives = self.config.objective_subset or objective_names
        active_directions: list[str] = []
        for obj in active_objectives:
            idx = objective_names.index(obj)
            active_directions.append(objective_directions[idx])

        # Build normalised values and weights
        normalized: list[float] = []
        weights: list[float] = []
        for obj, direction in zip(active_objectives, active_directions):
            val = observation.kpi_values.get(obj, 0.0)
            if direction == "minimize":
                val = -val  # negate so higher == better
            normalized.append(val)
            weights.append(self.config.weights.get(obj, 1.0))

        # Normalise weights to sum to 1
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        method = self.config.scalarization_method

        if method == "weighted_sum":
            return sum(w * v for w, v in zip(weights, normalized))

        if method == "tchebycheff":
            return min(w * v for w, v in zip(weights, normalized))

        if method == "achievement":
            # Minimise the maximum weighted absolute deviation
            return -max(w * abs(v) for w, v in zip(weights, normalized))

        # Fallback: treat as weighted_sum
        return sum(w * v for w, v in zip(weights, normalized))

    # ── epsilon filtering ────────────────────────────────────

    def filter_by_epsilon_constraints(
        self,
        observations: list[Observation],
    ) -> list[Observation]:
        """Return only observations satisfying **all** epsilon constraints."""
        if not self.config.epsilon_constraints:
            return list(observations)

        result: list[Observation] = []
        for obs in observations:
            if self._passes_epsilon(obs):
                result.append(obs)
        return result

    # ── ranking ──────────────────────────────────────────────

    def rank_observations(
        self,
        snapshot: CampaignSnapshot,
    ) -> list[tuple[int, float, str]]:
        """Rank observations by scalarised score.

        Returns a list of ``(index, score, explanation)`` tuples sorted by
        score **descending**.  Failed observations and those violating
        epsilon constraints are excluded.
        """
        feasible_indices: list[int] = []
        for i, obs in enumerate(snapshot.observations):
            if not obs.is_failure and self._passes_epsilon(obs):
                feasible_indices.append(i)

        scored: list[tuple[int, float, str]] = []
        for idx in feasible_indices:
            obs = snapshot.observations[idx]
            score = self.compute_scalarized_score(
                obs, snapshot.objective_names, snapshot.objective_directions,
            )
            explanation = self._generate_explanation(
                obs, score, snapshot.objective_names, snapshot.objective_directions,
            )
            scored.append((idx, score, explanation))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ── private helpers ──────────────────────────────────────

    def _passes_epsilon(self, obs: Observation) -> bool:
        """Return *True* if *obs* satisfies every epsilon constraint."""
        for ec in self.config.epsilon_constraints:
            val = obs.kpi_values.get(ec.objective)
            if val is None:
                return False
            if ec.lower_bound is not None and val < ec.lower_bound:
                return False
            if ec.upper_bound is not None and val > ec.upper_bound:
                return False
        return True

    def _generate_explanation(
        self,
        observation: Observation,
        score: float,
        objective_names: list[str],
        objective_directions: list[str],
    ) -> str:
        """Build a human-readable explanation for *score*."""
        active = self.config.objective_subset or objective_names
        parts: list[str] = []
        for obj in active:
            val = observation.kpi_values.get(obj, 0.0)
            weight = self.config.weights.get(obj, 1.0)
            parts.append(f"{obj}={val:.4g} (weight={weight:.2f})")

        method_name = self.config.scalarization_method
        return f"Score={score:.4f} via {method_name}: " + ", ".join(parts)
