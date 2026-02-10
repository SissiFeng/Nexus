"""Multi-fidelity planner for two-stage optimization.

Manages cheap proxy tests (screening) followed by expensive high-fidelity
tests (refinement) on promising candidates, using successive halving to
allocate budget efficiently across fidelity levels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.models import CampaignSnapshot, Observation


# ── Data Models ──────────────────────────────────────────


@dataclass
class FidelityLevel:
    """Describes one fidelity tier for evaluation.

    Attributes:
        name: Human-readable label (e.g. "low", "high").
        cost_multiplier: Relative cost of one evaluation at this fidelity.
        noise_multiplier: How noisy measurements are (higher = noisier).
        correlation_with_truth: How predictive this fidelity is of the
            ground-truth objective (0 = uncorrelated, 1 = perfect proxy).
    """

    name: str
    cost_multiplier: float
    noise_multiplier: float
    correlation_with_truth: float

    def __post_init__(self) -> None:
        if self.cost_multiplier <= 0:
            raise ValueError("cost_multiplier must be positive")
        if self.noise_multiplier < 0:
            raise ValueError("noise_multiplier must be non-negative")
        if not 0.0 <= self.correlation_with_truth <= 1.0:
            raise ValueError("correlation_with_truth must be in [0, 1]")


@dataclass
class FidelityPlan:
    """A single stage in a multi-fidelity evaluation plan.

    Attributes:
        stage: Stage label ("screening" or "refinement").
        fidelity_level: Which fidelity tier to use.
        n_candidates: Number of candidates to evaluate at this stage.
        promotion_threshold: Top fraction (0-1) promoted to the next stage.
        backend_hint: Suggested optimization algorithm for this stage.
        reason: Human-readable explanation of why this stage is configured
            this way.
    """

    stage: str
    fidelity_level: FidelityLevel
    n_candidates: int
    promotion_threshold: float
    backend_hint: str
    reason: str


@dataclass
class MultiFidelityPlan:
    """Complete multi-fidelity evaluation plan across all stages.

    Attributes:
        stages: Ordered list of fidelity stages to execute.
        total_estimated_cost: Sum of (n_candidates * cost_multiplier)
            across all stages.
        efficiency_gain: Estimated cost savings compared to evaluating
            all candidates at the highest fidelity (0-1 fraction).
    """

    stages: list[FidelityPlan]
    total_estimated_cost: float
    efficiency_gain: float


# ── Default fidelity levels ──────────────────────────────

_DEFAULT_LOW = FidelityLevel(
    name="low",
    cost_multiplier=1.0,
    noise_multiplier=2.0,
    correlation_with_truth=0.6,
)

_DEFAULT_HIGH = FidelityLevel(
    name="high",
    cost_multiplier=10.0,
    noise_multiplier=1.0,
    correlation_with_truth=1.0,
)

DEFAULT_FIDELITY_LEVELS: list[FidelityLevel] = [_DEFAULT_LOW, _DEFAULT_HIGH]


# ── Planner ──────────────────────────────────────────────


class MultiFidelityPlanner:
    """Plans and manages two-stage multi-fidelity optimization campaigns.

    The planner uses successive halving: a large pool of candidates is
    cheaply screened at low fidelity, and only the top fraction is
    promoted to expensive high-fidelity evaluation.

    Args:
        fidelity_levels: Ordered list of fidelity tiers from cheapest to
            most expensive. Defaults to a two-level (low, high) setup.
    """

    def __init__(
        self,
        fidelity_levels: list[FidelityLevel] | None = None,
    ) -> None:
        self.fidelity_levels = (
            list(fidelity_levels) if fidelity_levels is not None
            else list(DEFAULT_FIDELITY_LEVELS)
        )
        if not self.fidelity_levels:
            raise ValueError("At least one fidelity level is required")
        # Sort cheapest first
        self.fidelity_levels.sort(key=lambda fl: fl.cost_multiplier)

    # ── Plan generation ─────────────────────────────────

    def plan(
        self,
        snapshot: CampaignSnapshot,
        budget: float | None = None,
        n_total: int = 20,
    ) -> MultiFidelityPlan:
        """Compute a multi-fidelity evaluation plan.

        Uses successive halving across available fidelity levels.
        Stage 1 (screening) uses the cheapest fidelity to evaluate many
        candidates; subsequent stages promote the top 50 % to
        increasingly expensive fidelities.

        Args:
            snapshot: Current campaign state (used for context; e.g. how
                many observations already exist).
            budget: Optional total cost budget.  When provided the planner
                scales *n_total* down so the plan fits within budget.
            n_total: Starting number of candidates for the screening
                stage (before any budget adjustment).

        Returns:
            A ``MultiFidelityPlan`` with one stage per fidelity level.
        """
        n_levels = len(self.fidelity_levels)

        # Single fidelity level: no screening, just evaluate everything
        if n_levels == 1:
            only = self.fidelity_levels[0]
            cost = n_total * only.cost_multiplier
            stage = FidelityPlan(
                stage="refinement",
                fidelity_level=only,
                n_candidates=n_total,
                promotion_threshold=1.0,
                backend_hint="tpe",
                reason=(
                    f"Single fidelity level '{only.name}'; all {n_total} "
                    "candidates evaluated directly."
                ),
            )
            return MultiFidelityPlan(
                stages=[stage],
                total_estimated_cost=cost,
                efficiency_gain=0.0,
            )

        # --- Multi-level successive halving ---
        # If budget is given, scale n_total so estimated cost fits.
        if budget is not None and budget > 0:
            n_total = self._fit_to_budget(n_total, budget)
            # Ensure at least 2 candidates to make screening meaningful
            n_total = max(n_total, 2)

        stages: list[FidelityPlan] = []
        n_remaining = n_total

        for idx, fl in enumerate(self.fidelity_levels):
            is_last = idx == n_levels - 1

            # Promotion: top 50 % advance, except the last stage
            promotion_fraction = 0.5 if not is_last else 1.0

            if idx == 0:
                # Screening: broad search
                backend_hint = (
                    "latin_hypercube" if n_remaining >= 10 else "random"
                )
                stage_label = "screening"
                reason = (
                    f"Screen {n_remaining} candidates at cheap "
                    f"'{fl.name}' fidelity (cost x{fl.cost_multiplier}). "
                    f"Top {promotion_fraction:.0%} promoted."
                )
            elif is_last:
                # Final refinement
                backend_hint = "tpe"
                stage_label = "refinement"
                reason = (
                    f"Refine {n_remaining} survivors at high "
                    f"'{fl.name}' fidelity (cost x{fl.cost_multiplier}) "
                    "using model-based search."
                )
            else:
                # Intermediate stages
                backend_hint = "tpe"
                stage_label = "screening"
                reason = (
                    f"Intermediate evaluation of {n_remaining} candidates "
                    f"at '{fl.name}' fidelity (cost x{fl.cost_multiplier}). "
                    f"Top {promotion_fraction:.0%} promoted."
                )

            stages.append(
                FidelityPlan(
                    stage=stage_label,
                    fidelity_level=fl,
                    n_candidates=n_remaining,
                    promotion_threshold=promotion_fraction,
                    backend_hint=backend_hint,
                    reason=reason,
                )
            )

            # Successive halving for next stage
            n_remaining = max(1, math.ceil(n_remaining * promotion_fraction))

        total_cost = sum(
            s.n_candidates * s.fidelity_level.cost_multiplier for s in stages
        )

        # Cost of doing everything at highest fidelity
        highest = self.fidelity_levels[-1]
        single_fidelity_cost = n_total * highest.cost_multiplier

        efficiency = self.estimate_efficiency(
            MultiFidelityPlan(stages=stages, total_estimated_cost=total_cost, efficiency_gain=0.0),
            single_fidelity_cost,
        )

        return MultiFidelityPlan(
            stages=stages,
            total_estimated_cost=total_cost,
            efficiency_gain=efficiency,
        )

    # ── Promotion logic ─────────────────────────────────

    def should_promote(
        self,
        observation: Observation,
        threshold_kpi: float,
        maximize: bool,
    ) -> bool:
        """Decide whether an observation meets the promotion threshold.

        Args:
            observation: The screening result to evaluate.
            threshold_kpi: The KPI cutoff value.  For maximisation the
                observation must be >= this value; for minimisation it
                must be <= this value.
            maximize: Whether the objective is being maximised.

        Returns:
            True if the observation should be promoted to the next stage.
        """
        if not observation.kpi_values:
            return False
        # Use the first KPI if there are multiple
        kpi_value = next(iter(observation.kpi_values.values()))
        if maximize:
            return kpi_value >= threshold_kpi
        return kpi_value <= threshold_kpi

    def compute_promotion_threshold(
        self,
        observations: list[Observation],
        obj_name: str,
        top_fraction: float,
        maximize: bool,
    ) -> float:
        """Compute the KPI cutoff that selects the top fraction of results.

        Args:
            observations: Screening-stage results.
            obj_name: Name of the KPI / objective to rank by.
            top_fraction: Fraction of candidates to promote (0-1).
            maximize: Whether higher KPI values are better.

        Returns:
            The KPI value at the promotion boundary.  Observations
            meeting or exceeding (for max) / at or below (for min) this
            value are promoted.

        Raises:
            ValueError: If no observations contain the requested KPI.
        """
        if not observations:
            raise ValueError("No observations provided")
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1]")

        values = [
            obs.kpi_values[obj_name]
            for obs in observations
            if obj_name in obs.kpi_values
        ]
        if not values:
            raise ValueError(
                f"No observations contain KPI '{obj_name}'"
            )

        # Sort descending for maximise, ascending for minimise
        values.sort(reverse=maximize)

        # The cutoff index: how many to promote
        n_promote = max(1, math.ceil(len(values) * top_fraction))
        n_promote = min(n_promote, len(values))

        # The threshold is the value at the boundary
        return values[n_promote - 1]

    # ── Efficiency estimation ───────────────────────────

    def estimate_efficiency(
        self,
        plan: MultiFidelityPlan,
        single_fidelity_cost: float,
    ) -> float:
        """Compute cost savings of a multi-fidelity plan vs single fidelity.

        Args:
            plan: The multi-fidelity plan to evaluate.
            single_fidelity_cost: Cost of evaluating all candidates at the
                highest fidelity level.

        Returns:
            Fractional savings in [0, 1].  E.g. 0.6 means the
            multi-fidelity plan is 60 % cheaper.  Returns 0.0 if the
            multi-fidelity plan is not cheaper.
        """
        if single_fidelity_cost <= 0:
            return 0.0
        savings = 1.0 - (plan.total_estimated_cost / single_fidelity_cost)
        return max(0.0, savings)

    # ── Internal helpers ────────────────────────────────

    def _fit_to_budget(self, n_total: int, budget: float) -> int:
        """Scale n_total so the successive-halving plan fits the budget.

        This iteratively reduces n_total until the estimated cost is
        within the budget, using a binary-search approach.
        """
        lo, hi = 2, n_total
        best = lo

        while lo <= hi:
            mid = (lo + hi) // 2
            cost = self._estimate_cost_for_n(mid)
            if cost <= budget:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        return best

    def _estimate_cost_for_n(self, n_total: int) -> float:
        """Estimate total cost of successive halving starting with n_total."""
        total_cost = 0.0
        n_remaining = n_total
        for idx, fl in enumerate(self.fidelity_levels):
            total_cost += n_remaining * fl.cost_multiplier
            is_last = idx == len(self.fidelity_levels) - 1
            if not is_last:
                n_remaining = max(1, math.ceil(n_remaining * 0.5))
        return total_cost
