"""Many-objective optimization support (>3 objectives).

Provides hypervolume indicator computation, inverted generational distance,
and hypervolume-contribution-based ranking for many-objective problems.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


class HypervolumeIndicator:
    """Hypervolume indicator for measuring Pareto front quality.

    For 2D problems, uses an exact sorted-sweep algorithm.
    For 3D+ problems, uses Monte Carlo approximation.

    All points are assumed to be in minimization form (lower is better).
    """

    def __init__(self, n_samples: int = 10000, seed: int = 42) -> None:
        self.n_samples = n_samples
        self.seed = seed

    def compute(self, points: list[list[float]], ref_point: list[float]) -> float:
        """Compute hypervolume dominated by *points* bounded by *ref_point*.

        Parameters
        ----------
        points :
            Each element is a list of objective values (minimization).
        ref_point :
            The reference (anti-ideal) point bounding the hypervolume.

        Returns
        -------
        float
            The hypervolume indicator value (>= 0).
        """
        if not points or not ref_point:
            return 0.0

        n_obj = len(ref_point)

        # Filter out points that do not dominate the reference point in every
        # dimension (i.e., keep only points strictly inside the reference box).
        valid = [p for p in points if all(p[k] < ref_point[k] for k in range(n_obj))]
        if not valid:
            return 0.0

        if n_obj == 2:
            return self._exact_2d(valid, ref_point)
        return self._monte_carlo(valid, ref_point, n_obj)

    # ------------------------------------------------------------------
    # 2D exact sweep
    # ------------------------------------------------------------------
    @staticmethod
    def _exact_2d(points: list[list[float]], ref: list[float]) -> float:
        """Exact 2-D hypervolume via sorted sweep."""
        sorted_pts = sorted(points, key=lambda p: p[0])
        volume = 0.0
        prev_y = ref[1]
        for pt in sorted_pts:
            if pt[1] < prev_y:
                volume += (ref[0] - pt[0]) * (prev_y - pt[1])
                prev_y = pt[1]
        return volume

    # ------------------------------------------------------------------
    # Monte Carlo approximation for 3-D+
    # ------------------------------------------------------------------
    def _monte_carlo(
        self,
        points: list[list[float]],
        ref: list[float],
        n_obj: int,
    ) -> float:
        """Monte Carlo approximation of hypervolume for 3+ objectives."""
        # Compute ideal (component-wise minimum) across valid points
        ideal = [min(p[k] for p in points) for k in range(n_obj)]

        # Hyperrectangle volume from ideal to ref
        rect_vol = 1.0
        for k in range(n_obj):
            rect_vol *= ref[k] - ideal[k]
        if rect_vol <= 0.0:
            return 0.0

        rng = random.Random(self.seed)
        dominated_count = 0

        for _ in range(self.n_samples):
            # Random point uniformly in [ideal, ref)
            sample = [
                ideal[k] + rng.random() * (ref[k] - ideal[k]) for k in range(n_obj)
            ]
            # Check if sample is dominated by at least one front point
            for p in points:
                if all(p[k] <= sample[k] for k in range(n_obj)):
                    dominated_count += 1
                    break

        return (dominated_count / self.n_samples) * rect_vol


class IGDMetric:
    """Inverted Generational Distance (IGD).

    Measures how well an obtained front approximates a reference front.
    Lower values indicate better approximation.
    """

    def compute(
        self,
        obtained: list[list[float]],
        reference: list[list[float]],
    ) -> float:
        """Compute IGD from *obtained* front to *reference* front.

        For each point in *reference*, find the minimum Euclidean distance
        to any point in *obtained*, then return the mean of those distances.

        Parameters
        ----------
        obtained :
            The obtained (approximation) Pareto front.
        reference :
            The true/reference Pareto front.

        Returns
        -------
        float
            Mean of minimum distances (0.0 if reference is empty).
        """
        if not reference:
            return 0.0
        if not obtained:
            return float("inf")

        total = 0.0
        for r in reference:
            min_dist = float("inf")
            for o in obtained:
                dist = math.sqrt(sum((ri - oi) ** 2 for ri, oi in zip(r, o)))
                if dist < min_dist:
                    min_dist = dist
            total += min_dist

        return total / len(reference)


class ManyObjectiveRanker:
    """Rank solutions by hypervolume contribution.

    Each solution is ranked by how much hypervolume it uniquely contributes
    to the overall front.  Higher contribution = better rank (rank 1).
    """

    def __init__(self, n_samples: int = 5000, seed: int = 42) -> None:
        self.n_samples = n_samples
        self.seed = seed

    def rank(
        self,
        points: list[list[float]],
        directions: list[str] | None = None,
    ) -> list[int]:
        """Rank points by descending hypervolume contribution.

        Parameters
        ----------
        points :
            Objective vectors for each solution.
        directions :
            ``"minimize"`` or ``"maximize"`` per objective.
            Defaults to all ``"minimize"``.

        Returns
        -------
        list[int]
            Rank for each point (1 = highest contribution).
            Ties are broken by original index (lower index wins).
        """
        if not points:
            return []

        n_obj = len(points[0])
        if directions is None:
            directions = ["minimize"] * n_obj

        # Normalize to minimization
        norm = self._normalize_to_min(points, directions)

        # Reference point: component-wise max + small margin
        ref = [max(p[k] for p in norm) * 1.1 + 1e-9 for k in range(n_obj)]

        hv = HypervolumeIndicator(n_samples=self.n_samples, seed=self.seed)
        total_hv = hv.compute(norm, ref)

        contributions: list[tuple[float, int]] = []
        for i in range(len(norm)):
            subset = norm[:i] + norm[i + 1:]
            hv_without = hv.compute(subset, ref) if subset else 0.0
            contrib = total_hv - hv_without
            contributions.append((contrib, i))

        # Sort by contribution descending, then by index ascending for ties
        contributions.sort(key=lambda x: (-x[0], x[1]))

        ranks = [0] * len(points)
        for rank_val, (_, idx) in enumerate(contributions, start=1):
            ranks[idx] = rank_val

        return ranks

    def contribution(
        self,
        points: list[list[float]],
        index: int,
        directions: list[str] | None = None,
    ) -> float:
        """Compute hypervolume contribution of a single point.

        Parameters
        ----------
        points :
            All objective vectors.
        index :
            Index of the point whose contribution to compute.
        directions :
            ``"minimize"`` or ``"maximize"`` per objective.

        Returns
        -------
        float
            Hypervolume contribution of the point at *index*.
        """
        if not points or index < 0 or index >= len(points):
            return 0.0

        n_obj = len(points[0])
        if directions is None:
            directions = ["minimize"] * n_obj

        norm = self._normalize_to_min(points, directions)
        ref = [max(p[k] for p in norm) * 1.1 + 1e-9 for k in range(n_obj)]

        hv = HypervolumeIndicator(n_samples=self.n_samples, seed=self.seed)
        total_hv = hv.compute(norm, ref)

        subset = norm[:index] + norm[index + 1:]
        hv_without = hv.compute(subset, ref) if subset else 0.0

        return total_hv - hv_without

    @staticmethod
    def _normalize_to_min(
        points: list[list[float]],
        directions: list[str],
    ) -> list[list[float]]:
        """Flip maximize objectives so all are minimization."""
        result: list[list[float]] = []
        for p in points:
            row: list[float] = []
            for k, v in enumerate(p):
                if directions[k] == "maximize":
                    row.append(-v)
                else:
                    row.append(v)
            result.append(row)
        return result
