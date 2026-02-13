"""SDL performance metrics from Nature Communications 2024.

Implements three core metrics for evaluating Self-Driving Laboratory
performance:

* **Acceleration Factor** -- how much faster Bayesian optimisation reaches
  a target value compared to random search.
* **Enhancement Factor** -- how much better the BO result is at a fixed
  evaluation budget.
* **Degree of Autonomy** -- fraction of decisions made without human
  intervention.

Also provides :class:`SDLPerformanceReport` (a summary dataclass) and
:class:`SDLMetricsCalculator` for convenient batch computation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

class AccelerationFactor:
    """How much faster BO reaches *target* compared to random search."""

    @staticmethod
    def compute(
        bo_iterations: int,
        random_iterations: int,
        target: float,
    ) -> float:
        """Return ``random_iterations / bo_iterations``.

        Returns ``1.0`` when *bo_iterations* is non-positive to avoid
        division by zero.
        """
        if bo_iterations <= 0:
            return 1.0
        return random_iterations / bo_iterations


class EnhancementFactor:
    """How much better the BO result is at a fixed budget."""

    @staticmethod
    def compute(
        bo_best: float,
        random_best: float,
        budget: int,
        direction: str = "minimize",
    ) -> float:
        """Return the enhancement ratio.

        For minimisation: ``random_best / bo_best`` (lower is better,
        so BO having a smaller value yields a ratio > 1).

        For maximisation: ``bo_best / random_best`` (higher is better).

        Returns ``1.0`` when the denominator is zero.
        """
        if direction == "minimize":
            if bo_best == 0.0:
                return 1.0
            return random_best / bo_best
        else:
            # maximisation
            if random_best == 0.0:
                return 1.0
            return bo_best / random_best


class DegreeOfAutonomy:
    """Fraction of decisions made without human intervention."""

    @staticmethod
    def compute(
        human_interventions: int,
        total_decisions: int,
    ) -> float:
        """Return ``1.0 - human_interventions / total_decisions``.

        The result is clamped to ``[0, 1]``.  Returns ``1.0`` when
        *total_decisions* is zero (no decisions means full autonomy by
        convention).
        """
        if total_decisions <= 0:
            return 1.0
        ratio = 1.0 - (human_interventions / total_decisions)
        return max(0.0, min(1.0, ratio))


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class SDLPerformanceReport:
    """Aggregated SDL performance report."""

    acceleration_factor: float
    enhancement_factor: float
    degree_of_autonomy: float
    total_iterations: int
    best_value: float
    human_interventions: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class SDLMetricsCalculator:
    """Convenience class that computes all SDL metrics in one call."""

    def evaluate(
        self,
        bo_iterations: int,
        bo_best: float,
        random_iterations: int,
        random_best: float,
        budget: int,
        human_interventions: int,
        total_decisions: int,
        target: float,
        direction: str = "minimize",
    ) -> SDLPerformanceReport:
        """Compute all three metrics and return a :class:`SDLPerformanceReport`."""
        af = AccelerationFactor.compute(bo_iterations, random_iterations, target)
        ef = EnhancementFactor.compute(bo_best, random_best, budget, direction)
        doa = DegreeOfAutonomy.compute(human_interventions, total_decisions)

        return SDLPerformanceReport(
            acceleration_factor=af,
            enhancement_factor=ef,
            degree_of_autonomy=doa,
            total_iterations=bo_iterations,
            best_value=bo_best,
            human_interventions=human_interventions,
            metadata={
                "random_iterations": random_iterations,
                "random_best": random_best,
                "budget": budget,
                "target": target,
                "direction": direction,
                "total_decisions": total_decisions,
            },
        )
