"""Batch diversification policy.

Ensures that batches of suggested parameter configurations are
diverse rather than clustered, improving exploration coverage.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from optimization_copilot.core.models import (
    Observation,
    ParameterSpec,
    VariableType,
)


@dataclass
class BatchPolicy:
    """Result of a batch diversification operation."""

    points: list[dict[str, float]]
    """The diversified batch of parameter configurations."""

    diversity_score: float
    """0-1, how diverse the selected batch is."""

    coverage_gain: float
    """Estimated exploration coverage improvement from this batch."""

    strategy: str
    """Strategy used: 'maximin', 'coverage', or 'hybrid'."""


class BatchDiversifier:
    """Selects diverse subsets from candidate parameter configurations.

    Given a pool of candidate configurations, selects a batch of
    ``n_select`` points that are well-spread across the parameter
    space, avoiding clustering.

    Parameters
    ----------
    strategy:
        Diversification strategy.  One of ``"maximin"`` (maximize
        minimum pairwise distance), ``"coverage"`` (maximize bin
        coverage), or ``"hybrid"`` (weighted combination of both).
    """

    STRATEGIES = ("maximin", "coverage", "hybrid")

    def __init__(self, strategy: str = "hybrid") -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}; choose from {self.STRATEGIES}"
            )
        self.strategy = strategy

    # ── public API ────────────────────────────────────────

    def diversify(
        self,
        candidates: list[dict],
        specs: list[ParameterSpec],
        n_select: int,
        existing_obs: list[Observation] | None = None,
        seed: int = 42,
    ) -> BatchPolicy:
        """Select a diverse batch from candidates.

        Parameters
        ----------
        candidates:
            Pool of candidate parameter configurations (dicts mapping
            parameter name to value).
        specs:
            Parameter specifications for normalization.
        n_select:
            Number of points to select from the candidate pool.
        existing_obs:
            Already-observed configurations, used by coverage strategy
            to prioritise unexplored regions.
        seed:
            Random seed for reproducibility.

        Returns
        -------
        BatchPolicy
            The selected batch with diversity metrics.
        """
        rng = random.Random(seed)

        if not candidates:
            return BatchPolicy(
                points=[],
                diversity_score=0.0,
                coverage_gain=0.0,
                strategy=self.strategy,
            )

        n_select = min(n_select, len(candidates))

        if n_select <= 1:
            # Just pick the first candidate (or random)
            selected = [candidates[0]] if candidates else []
            return BatchPolicy(
                points=selected,
                diversity_score=0.0 if len(selected) <= 1 else 1.0,
                coverage_gain=self._coverage_gain(
                    selected, existing_obs or [], specs,
                ),
                strategy=self.strategy,
            )

        if self.strategy == "maximin":
            selected = self._select_maximin(candidates, specs, n_select, rng)
        elif self.strategy == "coverage":
            selected = self._select_coverage(
                candidates, specs, n_select, existing_obs or [], rng,
            )
        else:  # hybrid
            selected = self._select_hybrid(
                candidates, specs, n_select, existing_obs or [], rng,
            )

        diversity = self._diversity_score(selected, specs)
        coverage = self._coverage_gain(selected, existing_obs or [], specs)

        return BatchPolicy(
            points=selected,
            diversity_score=diversity,
            coverage_gain=coverage,
            strategy=self.strategy,
        )

    # ── strategy implementations ──────────────────────────

    def _select_maximin(
        self,
        candidates: list[dict],
        specs: list[ParameterSpec],
        n_select: int,
        rng: random.Random,
    ) -> list[dict]:
        """Greedy maximin: iteratively add the candidate farthest from
        the current selection.

        Start with the first candidate and greedily add the one that
        maximises the minimum distance to any already-selected point.
        """
        selected: list[dict] = [candidates[0]]
        remaining = list(range(1, len(candidates)))

        for _ in range(n_select - 1):
            if not remaining:
                break

            best_idx = -1
            best_min_dist = -1.0

            for r_idx in remaining:
                c = candidates[r_idx]
                min_dist = min(
                    self._normalized_distance(c, s, specs) for s in selected
                )
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = r_idx

            selected.append(candidates[best_idx])
            remaining.remove(best_idx)

        return selected

    def _select_coverage(
        self,
        candidates: list[dict],
        specs: list[ParameterSpec],
        n_select: int,
        existing_obs: list[Observation],
        rng: random.Random,
        n_bins: int = 10,
    ) -> list[dict]:
        """Greedy coverage: select candidates that cover the most new bins."""
        # Build set of already-covered bins from existing observations
        covered_bins: set[tuple[int, ...]] = set()
        for obs in existing_obs:
            b = self._point_bins(obs.parameters, specs, n_bins)
            covered_bins.add(b)

        selected: list[dict] = []
        remaining = list(range(len(candidates)))

        for _ in range(n_select):
            if not remaining:
                break

            best_idx = -1
            best_new_bins = -1

            for r_idx in remaining:
                c = candidates[r_idx]
                b = self._point_bins(c, specs, n_bins)
                # Count how many new bins this covers
                new_count = 1 if b not in covered_bins else 0
                # Also count the number of previously-uncovered individual
                # dimension bins for tie-breaking
                dim_new = sum(
                    1 for dim_bin in b
                    if not any(dim_bin == existing[i] for existing in covered_bins for i in range(len(b)) if i < len(existing))
                )
                score = new_count * 1000 + dim_new

                if score > best_new_bins:
                    best_new_bins = score
                    best_idx = r_idx

            if best_idx == -1:
                # All remaining candidates cover the same bins; pick randomly
                best_idx = rng.choice(remaining)

            point = candidates[best_idx]
            selected.append(point)
            covered_bins.add(self._point_bins(point, specs, n_bins))
            remaining.remove(best_idx)

        return selected

    def _select_hybrid(
        self,
        candidates: list[dict],
        specs: list[ParameterSpec],
        n_select: int,
        existing_obs: list[Observation],
        rng: random.Random,
        n_bins: int = 10,
    ) -> list[dict]:
        """Hybrid: score = 0.5 * distance_score + 0.5 * coverage_score."""
        covered_bins: set[tuple[int, ...]] = set()
        for obs in existing_obs:
            b = self._point_bins(obs.parameters, specs, n_bins)
            covered_bins.add(b)

        selected: list[dict] = [candidates[0]]
        covered_bins.add(self._point_bins(candidates[0], specs, n_bins))
        remaining = list(range(1, len(candidates)))

        for _ in range(n_select - 1):
            if not remaining:
                break

            best_idx = -1
            best_score = -math.inf

            for r_idx in remaining:
                c = candidates[r_idx]

                # Distance component: min distance to already selected
                min_dist = min(
                    self._normalized_distance(c, s, specs) for s in selected
                )

                # Coverage component: does it cover a new bin?
                b = self._point_bins(c, specs, n_bins)
                coverage_val = 1.0 if b not in covered_bins else 0.0

                # Normalize distance to [0, 1] approximately
                # Max possible distance is sqrt(n_dims), so divide by that
                n_dims = max(len(specs), 1)
                norm_dist = min_dist / math.sqrt(n_dims)

                score = 0.5 * norm_dist + 0.5 * coverage_val

                if score > best_score:
                    best_score = score
                    best_idx = r_idx

            if best_idx == -1:
                best_idx = rng.choice(remaining)

            point = candidates[best_idx]
            selected.append(point)
            covered_bins.add(self._point_bins(point, specs, n_bins))
            remaining.remove(best_idx)

        return selected

    # ── distance and metrics ──────────────────────────────

    def _normalized_distance(
        self,
        a: dict,
        b: dict,
        specs: list[ParameterSpec],
    ) -> float:
        """Euclidean distance with normalization by parameter range.

        Continuous/discrete parameters are normalized by their range.
        Categorical parameters contribute 0 if same, 1 if different.
        """
        sq_sum = 0.0
        for spec in specs:
            name = spec.name
            va = a.get(name)
            vb = b.get(name)

            if va is None or vb is None:
                continue

            if spec.type == VariableType.CATEGORICAL:
                sq_sum += 0.0 if va == vb else 1.0
            else:
                # Continuous or discrete: normalize by range
                lo = spec.lower if spec.lower is not None else 0.0
                hi = spec.upper if spec.upper is not None else 1.0
                rng_size = hi - lo
                if rng_size <= 0.0:
                    rng_size = 1.0
                diff = (float(va) - float(vb)) / rng_size
                sq_sum += diff * diff

        return math.sqrt(sq_sum)

    def _diversity_score(
        self,
        points: list[dict],
        specs: list[ParameterSpec],
    ) -> float:
        """Average pairwise normalized distance, scaled to [0, 1].

        The maximum possible distance for a single pair is
        ``sqrt(n_dims)`` (when every dimension differs by 1 in
        normalized space), so we divide by that.
        """
        n = len(points)
        if n < 2:
            return 0.0

        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += self._normalized_distance(points[i], points[j], specs)
                count += 1

        avg_dist = total_dist / count
        n_dims = max(len(specs), 1)
        max_dist = math.sqrt(n_dims)

        return min(avg_dist / max_dist, 1.0)

    def _coverage_gain(
        self,
        points: list[dict],
        existing: list[Observation],
        specs: list[ParameterSpec],
        n_bins: int = 10,
    ) -> float:
        """Fraction of new bins covered by the batch.

        Divides the parameter space into ``n_bins`` per dimension and
        computes what fraction of the newly-covered bins (by the batch)
        are previously uncovered.
        """
        if not points or not specs:
            return 0.0

        existing_bins: set[tuple[int, ...]] = set()
        for obs in existing:
            existing_bins.add(self._point_bins(obs.parameters, specs, n_bins))

        new_bins: set[tuple[int, ...]] = set()
        for p in points:
            b = self._point_bins(p, specs, n_bins)
            if b not in existing_bins:
                new_bins.add(b)

        # Total possible bins = n_bins ^ n_dims, but that can be huge.
        # Instead report as fraction of batch points that hit new bins.
        if not points:
            return 0.0
        return len(new_bins) / len(points)

    def _point_bins(
        self,
        point: dict,
        specs: list[ParameterSpec],
        n_bins: int,
    ) -> tuple[int, ...]:
        """Map a point to its bin index in each dimension."""
        bins: list[int] = []
        for spec in specs:
            val = point.get(spec.name)
            if val is None:
                bins.append(0)
                continue

            if spec.type == VariableType.CATEGORICAL:
                # Hash category to a bin
                cats = spec.categories or []
                if val in cats:
                    bins.append(cats.index(val))
                else:
                    bins.append(hash(val) % n_bins)
            else:
                lo = spec.lower if spec.lower is not None else 0.0
                hi = spec.upper if spec.upper is not None else 1.0
                rng = hi - lo
                if rng <= 0.0:
                    bins.append(0)
                else:
                    normalized = (float(val) - lo) / rng
                    normalized = max(0.0, min(1.0, normalized))
                    b = int(normalized * n_bins)
                    # Clamp the last bin
                    if b >= n_bins:
                        b = n_bins - 1
                    bins.append(b)

        return tuple(bins)
