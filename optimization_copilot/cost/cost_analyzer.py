"""Cost-aware optimization analysis.

Adds cost-awareness to optimization decisions by tracking spend,
computing cost-efficiency metrics, and adjusting exploration based
on budget pressure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from optimization_copilot.core.models import CampaignSnapshot


@dataclass
class CostSignals:
    """Signals derived from cost analysis of an optimization campaign."""

    cost_per_improvement: float
    """Average cost to achieve one unit of KPI improvement."""

    time_budget_pressure: float
    """0-1, how close to budget exhaustion (cumulative_cost / total_budget)."""

    cost_efficiency_trend: float
    """-1 to 1, whether cost efficiency is improving (+) or worsening (-)."""

    cumulative_cost: float
    """Total cost incurred so far."""

    estimated_remaining_budget: float
    """Budget remaining (total_budget - cumulative_cost), or inf if no budget."""

    cost_optimal_batch_size: int
    """Recommended batch size given current cost dynamics."""


class CostAnalyzer:
    """Analyzes cost dynamics of an optimization campaign.

    Extracts cost from observation metadata or uses timestamp gaps as a
    proxy for cost.  Produces :class:`CostSignals` that downstream
    components (e.g. the meta-controller) can use to modulate exploration
    and batch sizing.

    Parameters
    ----------
    total_budget:
        Optional hard budget cap.  When provided, budget-pressure metrics
        are computed relative to this value.
    cost_field:
        Key in ``Observation.metadata`` that holds the per-observation
        cost.  If absent, the analyzer falls back to timestamp deltas.
    """

    def __init__(
        self,
        total_budget: float | None = None,
        cost_field: str = "cost",
    ) -> None:
        self.total_budget = total_budget
        self.cost_field = cost_field

    # ── public API ────────────────────────────────────────

    def analyze(self, snapshot: CampaignSnapshot) -> CostSignals:
        """Derive cost signals from the current campaign state.

        Parameters
        ----------
        snapshot:
            The complete campaign state including all observations.

        Returns
        -------
        CostSignals
            Computed cost metrics for the campaign.
        """
        observations = snapshot.observations
        if not observations:
            return CostSignals(
                cost_per_improvement=0.0,
                time_budget_pressure=0.0,
                cost_efficiency_trend=0.0,
                cumulative_cost=0.0,
                estimated_remaining_budget=(
                    self.total_budget if self.total_budget is not None else math.inf
                ),
                cost_optimal_batch_size=1,
            )

        # --- extract per-observation costs ---
        costs = self._extract_costs(snapshot)
        cumulative_cost = sum(costs)

        # --- total KPI improvement ---
        total_improvement = self._total_kpi_improvement(snapshot)

        # --- cost per improvement ---
        if total_improvement > 0.0:
            cost_per_improvement = cumulative_cost / total_improvement
        else:
            # No measurable improvement yet; signal infinite cost
            cost_per_improvement = math.inf if cumulative_cost > 0.0 else 0.0

        # --- budget pressure ---
        if self.total_budget is not None and self.total_budget > 0.0:
            time_budget_pressure = min(cumulative_cost / self.total_budget, 1.0)
        else:
            time_budget_pressure = 0.0

        # --- cost-efficiency trend ---
        cost_efficiency_trend = self._cost_efficiency_trend(snapshot, costs)

        # --- remaining budget ---
        if self.total_budget is not None:
            estimated_remaining_budget = max(self.total_budget - cumulative_cost, 0.0)
        else:
            estimated_remaining_budget = math.inf

        # --- optimal batch size ---
        cost_optimal_batch_size = self._optimal_batch_size(
            cumulative_cost, len(observations), time_budget_pressure,
        )

        return CostSignals(
            cost_per_improvement=cost_per_improvement,
            time_budget_pressure=time_budget_pressure,
            cost_efficiency_trend=cost_efficiency_trend,
            cumulative_cost=cumulative_cost,
            estimated_remaining_budget=estimated_remaining_budget,
            cost_optimal_batch_size=cost_optimal_batch_size,
        )

    def adjust_exploration(
        self,
        base_exploration: float,
        signals: CostSignals,
    ) -> float:
        """Adjust exploration strength based on cost signals.

        High budget pressure reduces exploration (favour exploitation of
        known-good regions).  Low cost-per-improvement allows more
        exploration because experiments are cheap relative to gains.

        Parameters
        ----------
        base_exploration:
            The exploration strength before cost adjustment (0-1).
        signals:
            Cost signals from :meth:`analyze`.

        Returns
        -------
        float
            Adjusted exploration strength, clamped to [0, 1].
        """
        adjustment = 0.0

        # High budget pressure -> reduce exploration
        # Linear reduction: at pressure=1.0, subtract up to 0.5
        adjustment -= signals.time_budget_pressure * 0.5

        # Cost efficiency trend: improving efficiency -> slightly more explore
        # trend in [-1, 1]; positive = improving
        adjustment += signals.cost_efficiency_trend * 0.15

        # If cost per improvement is very low relative to remaining budget,
        # we can afford to explore more.
        if (
            signals.estimated_remaining_budget != math.inf
            and signals.cost_per_improvement > 0.0
            and signals.estimated_remaining_budget > 0.0
        ):
            affordable_improvements = (
                signals.estimated_remaining_budget / signals.cost_per_improvement
            )
            if affordable_improvements > 20:
                # Plenty of budget left relative to cost -> explore more
                adjustment += 0.1
            elif affordable_improvements < 5:
                # Tight budget -> exploit
                adjustment -= 0.1

        adjusted = base_exploration + adjustment
        return max(0.0, min(1.0, adjusted))

    # ── private helpers ───────────────────────────────────

    def _extract_costs(self, snapshot: CampaignSnapshot) -> list[float]:
        """Extract per-observation costs.

        Prefers the ``metadata[cost_field]`` value.  Falls back to using
        timestamp deltas as a cost proxy.
        """
        observations = snapshot.observations
        costs: list[float] = []

        # Try metadata first
        has_metadata_cost = any(
            self.cost_field in obs.metadata for obs in observations
        )

        if has_metadata_cost:
            for obs in observations:
                costs.append(float(obs.metadata.get(self.cost_field, 0.0)))
        else:
            # Fallback: timestamp gaps as proxy for cost
            for i, obs in enumerate(observations):
                if i == 0:
                    costs.append(obs.timestamp)
                else:
                    delta = obs.timestamp - observations[i - 1].timestamp
                    costs.append(max(delta, 0.0))

        return costs

    def _total_kpi_improvement(self, snapshot: CampaignSnapshot) -> float:
        """Compute total KPI improvement across the campaign.

        Uses the first objective and its direction to determine
        improvement.  Returns the absolute improvement from the first
        observation to the best observation seen.
        """
        successful = snapshot.successful_observations
        if len(successful) < 2:
            return 0.0

        obj_name = snapshot.objective_names[0]
        direction = snapshot.objective_directions[0]

        values = [
            obs.kpi_values[obj_name]
            for obs in successful
            if obj_name in obs.kpi_values
        ]
        if not values:
            return 0.0

        first_value = values[0]
        if direction == "maximize":
            best_value = max(values)
            improvement = best_value - first_value
        else:
            best_value = min(values)
            improvement = first_value - best_value

        return max(improvement, 0.0)

    def _cost_efficiency_trend(
        self,
        snapshot: CampaignSnapshot,
        costs: list[float],
    ) -> float:
        """Compare recent cost-per-improvement to historical average.

        Returns a value in [-1, 1]:
        - Positive = recent efficiency is *better* (cheaper per improvement)
        - Negative = recent efficiency is *worse* (more expensive)
        - 0 = no change or insufficient data
        """
        successful = snapshot.successful_observations
        if len(successful) < 4:
            return 0.0

        obj_name = snapshot.objective_names[0]
        direction = snapshot.objective_directions[0]

        # Split into two halves
        mid = len(successful) // 2
        first_half = successful[:mid]
        second_half = successful[mid:]

        def _half_cpi(
            half_obs: list,
            half_costs: list[float],
        ) -> float:
            """Cost per improvement for a subset."""
            if len(half_obs) < 2:
                return 0.0
            vals = [
                o.kpi_values.get(obj_name, 0.0) for o in half_obs
            ]
            if direction == "maximize":
                impr = max(vals) - min(vals)
            else:
                impr = max(vals) - min(vals)
            total_c = sum(half_costs) if half_costs else 0.0
            if impr <= 0.0:
                return math.inf
            return total_c / impr

        # Map successful obs indices back to cost indices
        obs_set = {id(o) for o in snapshot.observations}
        obs_to_idx = {id(o): i for i, o in enumerate(snapshot.observations)}

        first_costs = [costs[obs_to_idx[id(o)]] for o in first_half if id(o) in obs_to_idx]
        second_costs = [costs[obs_to_idx[id(o)]] for o in second_half if id(o) in obs_to_idx]

        cpi_first = _half_cpi(first_half, first_costs)
        cpi_second = _half_cpi(second_half, second_costs)

        if cpi_first == math.inf and cpi_second == math.inf:
            return 0.0
        if cpi_first == math.inf:
            return 1.0  # Was infinite, now finite -> improving
        if cpi_second == math.inf:
            return -1.0  # Was finite, now infinite -> worsening
        if cpi_first == 0.0:
            return 0.0

        # Ratio: < 1 means second half is cheaper (better)
        ratio = cpi_second / cpi_first
        # Map to [-1, 1]: ratio=0.5 -> +0.5, ratio=2.0 -> -0.5
        if ratio <= 0.0:
            return 0.0
        trend = 1.0 - ratio
        return max(-1.0, min(1.0, trend))

    def _optimal_batch_size(
        self,
        cumulative_cost: float,
        n_observations: int,
        budget_pressure: float,
    ) -> int:
        """Suggest a batch size based on cost dynamics.

        High budget pressure or high cost -> smaller batches (more
        cautious). Low cost -> larger batches (more parallel evaluation).
        """
        if n_observations == 0:
            return 1

        avg_cost = cumulative_cost / n_observations

        # Base batch size
        base = 4

        # Scale down with budget pressure
        # At pressure=0 -> base, at pressure=1 -> 1
        pressure_factor = 1.0 - budget_pressure * 0.75
        adjusted = base * pressure_factor

        # Scale down if individual observations are expensive
        # (using avg_cost relative to total budget as a proxy)
        if self.total_budget is not None and self.total_budget > 0.0:
            cost_fraction = avg_cost / self.total_budget
            if cost_fraction > 0.1:
                adjusted *= 0.5
            elif cost_fraction < 0.01:
                adjusted *= 1.5

        return max(1, round(adjusted))
