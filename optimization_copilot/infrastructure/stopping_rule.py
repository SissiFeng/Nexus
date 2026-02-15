"""Principled stopping criteria for optimization campaigns.

Implements 5 stopping criteria:
1. Budget exhaustion (hard limit)
2. Improvement stagnation (patience-based)
3. Gittins Index (expected benefit < cost)
4. Model convergence (posterior uncertainty below threshold)
5. Pareto front stability (multi-objective)

References:
- Pandora's Box Gittins Index
- BayBE auto-stopping
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StoppingDecision:
    """Result of a stopping check."""
    should_stop: bool
    reason: str
    criterion: str  # which criterion triggered
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_stop": self.should_stop,
            "reason": self.reason,
            "criterion": self.criterion,
            "details": dict(self.details),
        }


class StoppingRule:
    """Principled stopping criteria for optimization.

    Multiple criteria can be configured; stopping is triggered
    when ANY criterion is met.
    """

    def __init__(
        self,
        max_trials: int | None = None,
        max_cost: float | None = None,
        improvement_patience: int = 10,
        improvement_threshold: float = 1e-4,
        min_uncertainty: float | None = None,
        pareto_stability_window: int = 5,
    ):
        self._max_trials = max_trials
        self._max_cost = max_cost
        self._patience = improvement_patience
        self._threshold = improvement_threshold
        self._min_unc = min_uncertainty
        self._pareto_window = pareto_stability_window

    def should_stop(
        self,
        n_trials: int = 0,
        total_cost: float = 0.0,
        best_values: list[float] | None = None,
        current_uncertainty: float | None = None,
        pareto_sizes: list[int] | None = None,
        gittins_index: float | None = None,
    ) -> StoppingDecision:
        """Check all stopping criteria.

        Returns a StoppingDecision. First criterion that triggers wins.
        """
        if best_values is None:
            best_values = []

        # 1. Max trials hard limit
        if self._max_trials is not None and n_trials >= self._max_trials:
            return StoppingDecision(
                should_stop=True,
                reason=f"Reached max trials ({self._max_trials})",
                criterion="max_trials",
                details={"n_trials": n_trials, "max_trials": self._max_trials},
            )

        # 2. Budget exhaustion
        if self._max_cost is not None and total_cost >= self._max_cost:
            return StoppingDecision(
                should_stop=True,
                reason=f"Budget exhausted (spent {total_cost:.2f} of {self._max_cost:.2f})",
                criterion="budget_exhausted",
                details={"total_cost": total_cost, "max_cost": self._max_cost},
            )

        # 3. Improvement stagnation
        if len(best_values) >= self._patience:
            recent = best_values[-self._patience:]
            improvement = abs(recent[-1] - recent[0])
            if improvement < self._threshold:
                return StoppingDecision(
                    should_stop=True,
                    reason=(
                        f"Improvement stagnated: {improvement:.6f} "
                        f"over last {self._patience} trials"
                    ),
                    criterion="stagnation",
                    details={
                        "improvement": improvement,
                        "patience": self._patience,
                        "threshold": self._threshold,
                    },
                )

        # 4. Gittins Index
        if gittins_index is not None and gittins_index <= 0:
            return StoppingDecision(
                should_stop=True,
                reason=(
                    f"Gittins index <= 0 ({gittins_index:.4f}): "
                    f"expected improvement < cost"
                ),
                criterion="gittins",
                details={"gittins_index": gittins_index},
            )

        # 5. Model convergence
        if self._min_unc is not None and current_uncertainty is not None:
            if current_uncertainty < self._min_unc:
                return StoppingDecision(
                    should_stop=True,
                    reason=(
                        f"Model converged: uncertainty {current_uncertainty:.4f} "
                        f"< threshold {self._min_unc}"
                    ),
                    criterion="convergence",
                    details={
                        "uncertainty": current_uncertainty,
                        "threshold": self._min_unc,
                    },
                )

        # 6. Pareto front stability
        if pareto_sizes and len(pareto_sizes) >= self._pareto_window:
            recent = pareto_sizes[-self._pareto_window:]
            if max(recent) == min(recent):
                return StoppingDecision(
                    should_stop=True,
                    reason=f"Pareto front stable for {self._pareto_window} iterations",
                    criterion="pareto_stable",
                    details={
                        "window": self._pareto_window,
                        "front_size": recent[-1],
                    },
                )

        # No criteria met — continue
        return StoppingDecision(
            should_stop=False,
            reason="Continue",
            criterion="none",
        )

    def active_criteria(self) -> list[str]:
        """Return names of criteria that are currently configured."""
        criteria = []
        if self._max_trials is not None:
            criteria.append("max_trials")
        if self._max_cost is not None:
            criteria.append("budget")
        criteria.append("stagnation")  # Always active
        criteria.append("gittins")     # Active if provided at check time
        if self._min_unc is not None:
            criteria.append("convergence")
        criteria.append("pareto_stable")  # Active if provided at check time
        return criteria

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "max_trials": self._max_trials,
            "max_cost": self._max_cost,
            "improvement_patience": self._patience,
            "improvement_threshold": self._threshold,
            "min_uncertainty": self._min_unc,
            "pareto_stability_window": self._pareto_window,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StoppingRule:
        """Deserialize from dict."""
        return cls(
            max_trials=data.get("max_trials"),
            max_cost=data.get("max_cost"),
            improvement_patience=data.get("improvement_patience", 10),
            improvement_threshold=data.get("improvement_threshold", 1e-4),
            min_uncertainty=data.get("min_uncertainty"),
            pareto_stability_window=data.get("pareto_stability_window", 5),
        )
