"""Interactive Pareto front exploration.

Provides query-based filtering, nearest-to-ideal search, and pairwise
tradeoff analysis for interactive decision-making on Pareto fronts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ParetoQuery:
    """Query specification for interactive Pareto front exploration.

    At most one of *weights*, *aspiration_levels*, or *bounds* should be set.
    If multiple are set, *weights* takes priority, then *aspiration_levels*,
    then *bounds*.
    """

    weights: dict[str, float] | None = None
    aspiration_levels: dict[str, float] | None = None
    bounds: dict[str, tuple[float, float]] | None = None


@dataclass
class TradeoffAnalysis:
    """Pairwise tradeoff analysis between two objectives on a Pareto front."""

    objective_a: str
    objective_b: str
    slope: float  # Approximate marginal rate of substitution
    correlation: float  # Pearson r between the two objectives on the front
    elasticity: float  # Percent change in obj_b per percent change in obj_a


class InteractiveParetoExplorer:
    """Interactive exploration of a Pareto front.

    Supports weighted-sum queries, aspiration-level proximity search,
    bounds-based filtering, nearest-to-ideal search, and pairwise
    tradeoff analysis.
    """

    def __init__(
        self,
        objective_names: list[str],
        directions: list[str] | None = None,
    ) -> None:
        self.objective_names = objective_names
        if directions is None:
            self.directions = ["minimize"] * len(objective_names)
        else:
            self.directions = list(directions)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        pareto_front: list[dict[str, float]],
        query: ParetoQuery,
    ) -> list[dict[str, float]]:
        """Filter / rank the Pareto front according to *query*.

        Parameters
        ----------
        pareto_front :
            List of dicts mapping objective name to value.
        query :
            The exploration query.

        Returns
        -------
        list[dict[str, float]]
            Ranked or filtered subset of the front.
        """
        if not pareto_front:
            return []

        if query.weights:
            return self._query_weighted(pareto_front, query.weights)
        if query.aspiration_levels:
            return self._query_aspiration(pareto_front, query.aspiration_levels)
        if query.bounds:
            return self._query_bounds(pareto_front, query.bounds)
        return list(pareto_front)

    def _query_weighted(
        self,
        front: list[dict[str, float]],
        weights: dict[str, float],
    ) -> list[dict[str, float]]:
        """Score each point as weighted sum (normalized to minimization)."""

        def score(pt: dict[str, float]) -> float:
            s = 0.0
            for name, d in zip(self.objective_names, self.directions):
                v = pt.get(name, 0.0)
                w = weights.get(name, 0.0)
                if d == "maximize":
                    s += w * (-v)  # flip for minimization
                else:
                    s += w * v
            return s

        return sorted(front, key=score)

    def _query_aspiration(
        self,
        front: list[dict[str, float]],
        aspirations: dict[str, float],
    ) -> list[dict[str, float]]:
        """Sort by Euclidean distance to aspiration levels."""

        def dist(pt: dict[str, float]) -> float:
            return math.sqrt(
                sum(
                    (pt.get(name, 0.0) - aspirations[name]) ** 2
                    for name in aspirations
                )
            )

        return sorted(front, key=dist)

    def _query_bounds(
        self,
        front: list[dict[str, float]],
        bounds: dict[str, tuple[float, float]],
    ) -> list[dict[str, float]]:
        """Filter to points within bounds, sort by first objective."""
        filtered: list[dict[str, float]] = []
        for pt in front:
            in_bounds = True
            for name, (lo, hi) in bounds.items():
                v = pt.get(name, 0.0)
                if v < lo or v > hi:
                    in_bounds = False
                    break
            if in_bounds:
                filtered.append(pt)

        if self.objective_names:
            first_obj = self.objective_names[0]
            filtered.sort(key=lambda p: p.get(first_obj, 0.0))

        return filtered

    # ------------------------------------------------------------------
    # Nearest to ideal
    # ------------------------------------------------------------------

    def nearest_to_ideal(
        self,
        pareto_front: list[dict[str, float]],
        ideal_point: dict[str, float],
    ) -> dict[str, float] | None:
        """Find the front point closest (normalised Euclidean) to *ideal_point*.

        Objectives are normalized to [0, 1] range before distance computation.

        Parameters
        ----------
        pareto_front :
            The Pareto front as list of dicts.
        ideal_point :
            The ideal (utopia) target values.

        Returns
        -------
        dict[str, float] | None
            The nearest point, or ``None`` if the front is empty.
        """
        if not pareto_front:
            return None

        names = self.objective_names

        # Compute min/max per objective for normalization
        mins: dict[str, float] = {}
        maxs: dict[str, float] = {}
        for name in names:
            vals = [pt.get(name, 0.0) for pt in pareto_front]
            mins[name] = min(vals)
            maxs[name] = max(vals)

        def norm_dist(pt: dict[str, float]) -> float:
            d = 0.0
            for name in names:
                rng = maxs[name] - mins[name]
                if rng < 1e-15:
                    continue
                norm_pt = (pt.get(name, 0.0) - mins[name]) / rng
                norm_ideal = (ideal_point.get(name, 0.0) - mins[name]) / rng
                d += (norm_pt - norm_ideal) ** 2
            return math.sqrt(d)

        best_pt = min(pareto_front, key=norm_dist)
        return dict(best_pt)

    # ------------------------------------------------------------------
    # Tradeoff analysis
    # ------------------------------------------------------------------

    def tradeoff_analysis(
        self,
        pareto_front: list[dict[str, float]],
        obj_a: str,
        obj_b: str,
    ) -> TradeoffAnalysis:
        """Compute pairwise tradeoff statistics between two objectives.

        Parameters
        ----------
        pareto_front :
            The Pareto front as list of dicts.
        obj_a, obj_b :
            Names of the two objectives.

        Returns
        -------
        TradeoffAnalysis
        """
        vals_a = [pt.get(obj_a, 0.0) for pt in pareto_front]
        vals_b = [pt.get(obj_b, 0.0) for pt in pareto_front]

        slope = self._compute_slope(vals_a, vals_b)
        corr = self._pearson(vals_a, vals_b)
        elasticity = self._compute_elasticity(vals_a, vals_b)

        return TradeoffAnalysis(
            objective_a=obj_a,
            objective_b=obj_b,
            slope=slope,
            correlation=corr,
            elasticity=elasticity,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_slope(vals_a: list[float], vals_b: list[float]) -> float:
        """Approximate marginal rate of substitution via finite differences.

        Sort by vals_a, then average consecutive finite differences.
        """
        if len(vals_a) < 2:
            return 0.0

        pairs = sorted(zip(vals_a, vals_b), key=lambda x: x[0])
        diffs: list[float] = []
        for i in range(1, len(pairs)):
            da = pairs[i][0] - pairs[i - 1][0]
            db = pairs[i][1] - pairs[i - 1][1]
            if abs(da) > 1e-15:
                diffs.append(db / da)

        if not diffs:
            return 0.0
        return sum(diffs) / len(diffs)

    @staticmethod
    def _pearson(xs: list[float], ys: list[float]) -> float:
        """Pearson correlation coefficient between xs and ys."""
        n = len(xs)
        if n < 2:
            return 0.0

        mean_x = sum(xs) / n
        mean_y = sum(ys) / n

        var_x = sum((x - mean_x) ** 2 for x in xs) / n
        var_y = sum((y - mean_y) ** 2 for y in ys) / n

        std_x = math.sqrt(var_x)
        std_y = math.sqrt(var_y)

        if std_x < 1e-15 or std_y < 1e-15:
            return 0.0

        cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n)) / n
        return cov / (std_x * std_y)

    @staticmethod
    def _compute_elasticity(vals_a: list[float], vals_b: list[float]) -> float:
        """Elasticity: average (% change in b) / (% change in a).

        Uses mean-relative changes computed from consecutive sorted pairs.
        """
        if len(vals_a) < 2:
            return 0.0

        pairs = sorted(zip(vals_a, vals_b), key=lambda x: x[0])
        elasticities: list[float] = []

        for i in range(1, len(pairs)):
            a_prev, b_prev = pairs[i - 1]
            a_curr, b_curr = pairs[i]

            # Use midpoint-relative percent changes to avoid division issues
            a_mid = (a_prev + a_curr) / 2.0
            b_mid = (b_prev + b_curr) / 2.0

            if abs(a_mid) < 1e-15 or abs(b_mid) < 1e-15:
                continue

            pct_a = (a_curr - a_prev) / abs(a_mid)
            pct_b = (b_curr - b_prev) / abs(b_mid)

            if abs(pct_a) < 1e-15:
                continue

            elasticities.append(pct_b / pct_a)

        if not elasticities:
            return 0.0
        return sum(elasticities) / len(elasticities)
