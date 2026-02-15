"""Automatic backend selection for optimization campaigns.

Selects the optimal backend based on problem characteristics,
current optimization phase, and historical performance.

Extends Optuna's AutoSampler concept with:
- Dynamic switching (not static selection)
- Constraint-aware selection
- Cost-aware selection (budget-sensitive)
- Historical portfolio score integration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SelectionResult:
    """Result of backend selection."""
    backend_name: str
    score: float
    scores_breakdown: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    alternatives: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "score": self.score,
            "scores_breakdown": dict(self.scores_breakdown),
            "reason": self.reason,
            "alternatives": list(self.alternatives),
        }


# Phase affinity: which backends work best in which phase
PHASE_AFFINITY: dict[str, dict[str, float]] = {
    "cold_start": {
        "sobol_sampler": 1.0,
        "latin_hypercube_sampler": 0.9,
        "random_sampler": 0.7,
        "tpe_sampler": 0.3,
        "differential_evolution": 0.5,
    },
    "learning": {
        "gaussian_process_bo": 1.0,
        "random_forest_bo": 0.9,
        "tpe_sampler": 0.7,
        "turbo_sampler": 0.6,
    },
    "exploitation": {
        "cmaes_sampler": 1.0,
        "gaussian_process_bo": 0.9,
        "tpe_sampler": 0.6,
        "turbo_sampler": 0.8,
    },
    "stagnation": {
        "differential_evolution": 1.0,
        "sobol_sampler": 0.8,
        "random_sampler": 0.7,
        "latin_hypercube_sampler": 0.6,
    },
    "termination": {
        "gaussian_process_bo": 0.8,
        "tpe_sampler": 0.7,
    },
}

# Data requirement scoring: how well each backend handles different data sizes
DATA_AFFINITY: dict[str, dict[str, float]] = {
    "few": {  # < 5 observations
        "sobol_sampler": 1.0,
        "latin_hypercube_sampler": 1.0,
        "random_sampler": 1.0,
        "tpe_sampler": 0.3,
        "differential_evolution": 0.5,
    },
    "moderate": {  # 5-20 observations
        "tpe_sampler": 1.0,
        "random_forest_bo": 0.9,
        "gaussian_process_bo": 0.7,
        "differential_evolution": 0.7,
    },
    "many": {  # > 20 observations
        "gaussian_process_bo": 1.0,
        "cmaes_sampler": 0.9,
        "random_forest_bo": 0.8,
        "turbo_sampler": 0.9,
        "tpe_sampler": 0.7,
    },
}

# Noise robustness scores
NOISE_ROBUSTNESS: dict[str, float] = {
    "random_forest_bo": 1.0,
    "tpe_sampler": 0.8,
    "differential_evolution": 0.7,
    "gaussian_process_bo": 0.5,
    "random_sampler": 0.5,
    "cmaes_sampler": 0.3,
}

# Computational efficiency scores (cheaper backends score higher when budget is tight)
EFFICIENCY: dict[str, float] = {
    "random_sampler": 1.0,
    "sobol_sampler": 0.95,
    "latin_hypercube_sampler": 0.9,
    "tpe_sampler": 0.8,
    "differential_evolution": 0.7,
    "random_forest_bo": 0.6,
    "nsga2_sampler": 0.5,
    "cmaes_sampler": 0.4,
    "gaussian_process_bo": 0.3,
    "turbo_sampler": 0.25,
}

# Multi-objective capability
MULTI_OBJ_BACKENDS: set[str] = {"nsga2_sampler", "gaussian_process_bo"}

# High-dimensional specialists
HIGH_DIM_BACKENDS: set[str] = {"turbo_sampler", "random_forest_bo", "differential_evolution"}

# Constrained optimization capable
CONSTRAINED_BACKENDS: set[str] = {"gaussian_process_bo", "random_forest_bo"}


class AutoSampler:
    """Automatic backend selector.

    Scores all available backends on multiple criteria and selects
    the best one for the current optimization state.

    Scoring weights:
    - Phase affinity: 3.0
    - Data requirement match: 2.0
    - Noise robustness: 2.0 (only when noise_level is "high")
    - Budget efficiency: 1.5 (only when budget < 10 trials remaining)
    - Portfolio history: 1.0
    - Special capability bonus: 1.5 (multi-obj, high-dim, constrained)
    """

    def __init__(
        self,
        available_backends: list[str] | None = None,
        weights: dict[str, float] | None = None,
    ):
        """Initialize AutoSampler.

        Args:
            available_backends: List of backend names (plugin names).
                If None, uses all known backends.
            weights: Custom scoring weights. Keys:
                phase, data, noise, efficiency, portfolio, capability
        """
        self._backends = available_backends or list(EFFICIENCY.keys())
        self._weights = weights or {
            "phase": 3.0,
            "data": 2.0,
            "noise": 2.0,
            "efficiency": 1.5,
            "portfolio": 1.0,
            "capability": 1.5,
        }
        self._selection_history: list[SelectionResult] = []

    @property
    def available_backends(self) -> list[str]:
        """Return list of available backend names."""
        return list(self._backends)

    @property
    def selection_history(self) -> list[SelectionResult]:
        """Return copy of selection history."""
        return list(self._selection_history)

    def select(
        self,
        phase: str = "learning",
        n_observations: int = 0,
        has_constraints: bool = False,
        is_multi_objective: bool = False,
        n_dimensions: int = 1,
        noise_level: str = "low",
        budget_remaining: float | None = None,
        portfolio_scores: dict[str, float] | None = None,
    ) -> SelectionResult:
        """Select the best backend for current state.

        Args:
            phase: Current optimization phase
                (cold_start, learning, exploitation, stagnation, termination)
            n_observations: Number of observations collected so far
            has_constraints: Whether the problem has constraints
            is_multi_objective: Whether the problem has multiple objectives
            n_dimensions: Number of parameter dimensions
            noise_level: Noise level ("low", "medium", "high")
            budget_remaining: Estimated remaining trial budget (None = unlimited)
            portfolio_scores: Historical backend performance scores {name: score}

        Returns:
            SelectionResult with chosen backend and scoring details
        """
        # Filter compatible backends
        candidates = self._filter_compatible(
            is_multi_objective=is_multi_objective,
            has_constraints=has_constraints,
        )

        if not candidates:
            # Ultimate fallback
            result = SelectionResult(
                backend_name="tpe_sampler",
                score=0.0,
                reason="No compatible backends found, using TPE fallback",
            )
            self._selection_history.append(result)
            return result

        # Determine data size category
        if n_observations < 5:
            data_cat = "few"
        elif n_observations < 20:
            data_cat = "moderate"
        else:
            data_cat = "many"

        # Score each candidate
        scores: dict[str, dict[str, float]] = {}
        for name in candidates:
            breakdown: dict[str, float] = {}

            # Phase affinity
            phase_score = PHASE_AFFINITY.get(phase, {}).get(name, 0.3)
            breakdown["phase"] = phase_score * self._weights["phase"]

            # Data requirement match
            data_score = DATA_AFFINITY.get(data_cat, {}).get(name, 0.3)
            breakdown["data"] = data_score * self._weights["data"]

            # Noise robustness (only matters when noise is high)
            if noise_level == "high":
                noise_score = NOISE_ROBUSTNESS.get(name, 0.3)
                breakdown["noise"] = noise_score * self._weights["noise"]
            else:
                breakdown["noise"] = 0.0

            # Budget efficiency (only matters when budget is tight)
            if budget_remaining is not None and budget_remaining < 10:
                eff_score = EFFICIENCY.get(name, 0.5)
                breakdown["efficiency"] = eff_score * self._weights["efficiency"]
            else:
                breakdown["efficiency"] = 0.0

            # Portfolio history
            if portfolio_scores and name in portfolio_scores:
                breakdown["portfolio"] = portfolio_scores[name] * self._weights["portfolio"]
            else:
                breakdown["portfolio"] = 0.0

            # Special capability bonus
            cap_bonus = 0.0
            if is_multi_objective and name in MULTI_OBJ_BACKENDS:
                cap_bonus += 1.0
            if n_dimensions > 10 and name in HIGH_DIM_BACKENDS:
                cap_bonus += 0.8
            if has_constraints and name in CONSTRAINED_BACKENDS:
                cap_bonus += 0.8
            breakdown["capability"] = cap_bonus * self._weights["capability"]

            scores[name] = breakdown

        # Total scores
        totals = {name: sum(b.values()) for name, b in scores.items()}

        # Select best
        best_name = max(totals, key=totals.get)  # type: ignore[arg-type]
        best_score = totals[best_name]

        # Get top 3 alternatives
        sorted_names = sorted(totals, key=totals.get, reverse=True)  # type: ignore[arg-type]
        alternatives = [n for n in sorted_names[1:4] if totals[n] > 0]

        # Build reason string
        reason = f"Best for phase={phase}, data={data_cat}"
        if noise_level == "high":
            reason += ", noisy"
        if has_constraints:
            reason += ", constrained"
        if is_multi_objective:
            reason += ", multi-objective"

        result = SelectionResult(
            backend_name=best_name,
            score=best_score,
            scores_breakdown=scores[best_name],
            reason=reason,
            alternatives=alternatives,
        )
        self._selection_history.append(result)
        return result

    def _filter_compatible(
        self,
        is_multi_objective: bool = False,
        has_constraints: bool = False,
    ) -> list[str]:
        """Filter backends that meet hard compatibility requirements.

        Args:
            is_multi_objective: Whether multi-objective support is required.
            has_constraints: Whether constraint handling is required.

        Returns:
            List of compatible backend names.
        """
        compatible = []
        for name in self._backends:
            # Multi-objective: only allow backends that support it
            if is_multi_objective and name not in MULTI_OBJ_BACKENDS:
                # Allow some backends that can work with scalarized objectives
                if name not in {"tpe_sampler", "random_sampler", "sobol_sampler",
                               "latin_hypercube_sampler", "differential_evolution"}:
                    continue
            compatible.append(name)
        return compatible

    def reset_history(self) -> None:
        """Clear selection history."""
        self._selection_history.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "backends": list(self._backends),
            "weights": dict(self._weights),
            "history": [r.to_dict() for r in self._selection_history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AutoSampler:
        """Deserialize from dict.

        Args:
            data: Dictionary with 'backends', 'weights', and optional 'history' keys.

        Returns:
            Restored AutoSampler instance.
        """
        sampler = cls(
            available_backends=data.get("backends"),
            weights=data.get("weights"),
        )
        # History is informational; don't need to restore SelectionResult objects
        return sampler
