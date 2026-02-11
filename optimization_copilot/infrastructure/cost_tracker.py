"""Cost-aware optimization infrastructure.

Tracks experiment costs, supports budget management, provides
cost-adjusted acquisition functions and Gittins stopping index.

References:
- LogEIPC acquisition function
- Pandora's Box Gittins Index (stopping criterion)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrialCost:
    """Cost record for a single trial."""
    trial_id: str
    wall_time_seconds: float = 0.0
    resource_cost: float = 0.0        # consumables, reagents, etc.
    compute_cost: float = 0.0         # compute resources
    opportunity_cost: float = 0.0     # opportunity cost
    fidelity_level: int = 0           # fidelity level (for multi-fidelity)
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        return (self.wall_time_seconds + self.resource_cost +
                self.compute_cost + self.opportunity_cost)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "wall_time_seconds": self.wall_time_seconds,
            "resource_cost": self.resource_cost,
            "compute_cost": self.compute_cost,
            "opportunity_cost": self.opportunity_cost,
            "fidelity_level": self.fidelity_level,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrialCost:
        return cls(
            trial_id=data["trial_id"],
            wall_time_seconds=data.get("wall_time_seconds", 0.0),
            resource_cost=data.get("resource_cost", 0.0),
            compute_cost=data.get("compute_cost", 0.0),
            opportunity_cost=data.get("opportunity_cost", 0.0),
            fidelity_level=data.get("fidelity_level", 0),
            timestamp=data.get("timestamp", 0.0),
            metadata=data.get("metadata", {}),
        )


class CostTracker:
    """Cost tracking and cost-aware optimization decisions.

    Features:
    - Budget tracking with remaining budget estimation
    - LogEIPC cost-adjusted acquisition function
    - Cost-adjusted regret computation
    - Gittins Index stopping criterion
    - Per-fidelity cost breakdown
    - Serialization for persistence
    """

    def __init__(self, budget: float | None = None,
                 cost_field: str = "total_cost"):
        self._budget = budget
        self._cost_field = cost_field
        self._history: list[TrialCost] = []

    @property
    def budget(self) -> float | None:
        return self._budget

    @property
    def total_spent(self) -> float:
        return sum(self._get_cost(c) for c in self._history)

    @property
    def remaining_budget(self) -> float | None:
        if self._budget is None:
            return None
        return max(0.0, self._budget - self.total_spent)

    @property
    def n_trials(self) -> int:
        return len(self._history)

    @property
    def average_cost_per_trial(self) -> float:
        if not self._history:
            return 0.0
        return self.total_spent / len(self._history)

    def _get_cost(self, tc: TrialCost) -> float:
        if self._cost_field == "total_cost":
            return tc.total_cost
        return getattr(tc, self._cost_field, tc.total_cost)

    def record_trial(self, cost: TrialCost) -> None:
        """Record a trial cost."""
        self._history.append(cost)

    def estimated_remaining_trials(self) -> int | None:
        """Estimate how many more trials the budget can afford."""
        remaining = self.remaining_budget
        avg = self.average_cost_per_trial
        if remaining is None or avg <= 0:
            return None
        return int(remaining / avg)

    def cost_by_fidelity(self) -> dict[int, float]:
        """Total cost breakdown by fidelity level."""
        result: dict[int, float] = {}
        for tc in self._history:
            level = tc.fidelity_level
            result[level] = result.get(level, 0.0) + self._get_cost(tc)
        return result

    def cost_adjusted_acquisition(
        self,
        acquisition_values: list[float],
        estimated_costs: list[float],
    ) -> list[float]:
        """LogEIPC: cost-adjusted acquisition values.

        LogEIPC(x) = log(EI(x)) / cost(x)

        More numerically stable than EI/cost.
        """
        adjusted: list[float] = []
        for acq, cost in zip(acquisition_values, estimated_costs):
            if cost <= 0:
                cost = 1e-8
            if acq <= 0:
                adjusted.append(-float("inf"))
            else:
                adjusted.append(math.log(acq) / cost)
        return adjusted

    def cost_adjusted_regret(
        self,
        best_values: list[float],
        optimal: float | None = None,
    ) -> list[float]:
        """Cost-adjusted simple regret sequence.

        Instead of per-step regret, compute regret per unit cost.
        """
        if optimal is None or not self._history:
            return []

        cumulative_cost = 0.0
        regrets: list[float] = []
        for i, (bv, tc) in enumerate(zip(best_values, self._history)):
            cumulative_cost += self._get_cost(tc)
            regret = abs(optimal - bv)
            regrets.append(regret / max(cumulative_cost, 1e-8))
        return regrets

    def gittins_stopping_index(
        self,
        current_best: float,
        posterior_mean: float,
        posterior_std: float,
        expected_cost: float,
    ) -> float:
        """Pandora's Box Gittins Index stopping criterion.

        When index <= 0, the expected benefit of continuing
        is less than the expected cost.

        Returns the index value. Positive = continue, negative/zero = stop.
        """
        if posterior_std <= 0 or expected_cost <= 0:
            return 0.0

        z = (posterior_mean - current_best) / posterior_std
        # Standard normal partial expectation:
        # E[max(Z - z, 0)] = phi(z) - z * (1 - Phi(z))
        phi_z = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        Phi_z = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        expected_improvement = phi_z - z * (1.0 - Phi_z)
        expected_improvement *= posterior_std

        return expected_improvement / expected_cost

    def cumulative_cost_series(self) -> list[float]:
        """Return cumulative cost over trials."""
        result: list[float] = []
        total = 0.0
        for tc in self._history:
            total += self._get_cost(tc)
            result.append(total)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "budget": self._budget,
            "cost_field": self._cost_field,
            "history": [tc.to_dict() for tc in self._history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostTracker:
        """Deserialize from dict."""
        tracker = cls(
            budget=data.get("budget"),
            cost_field=data.get("cost_field", "total_cost"),
        )
        for tc_data in data.get("history", []):
            tracker._history.append(TrialCost.from_dict(tc_data))
        return tracker
