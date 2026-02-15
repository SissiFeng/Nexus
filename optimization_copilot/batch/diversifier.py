"""Batch diversification, failure replanning, and adaptive sizing.

Ensures that batches of suggested parameter configurations are
diverse rather than clustered, improving exploration coverage.
Also handles batch failure replanning and stage-aware adaptive batch sizing.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.models import (
    Observation,
    ParameterSpec,
    Phase,
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


# ---------------------------------------------------------------------------
# Batch failure replanning (Pain Point 6)
# ---------------------------------------------------------------------------

@dataclass
class ReplanResult:
    """Result of batch failure replanning."""
    replacement_points: list[dict[str, float]]
    n_failed: int
    n_replaced: int
    strategy: str


class BatchFailureReplanner:
    """Auto-supplement a batch when some points fail.

    When a batch of experiments is run and some fail, this replanner
    generates replacement points to maintain the intended batch size.

    Parameters
    ----------
    diversifier :
        Optional BatchDiversifier instance for diverse replacements.
        If ``None``, a default hybrid diversifier is used.
    """

    def __init__(self, diversifier: BatchDiversifier | None = None) -> None:
        self._diversifier = diversifier or BatchDiversifier(strategy="hybrid")

    def replan(
        self,
        original_batch: list[dict[str, float]],
        results: list[bool],
        specs: list[ParameterSpec],
        existing_obs: list[Observation] | None = None,
        seed: int = 42,
    ) -> ReplanResult:
        """Generate replacements for failed batch points.

        Parameters
        ----------
        original_batch :
            The original batch of parameter configurations.
        results :
            Boolean per point: ``True`` = succeeded, ``False`` = failed.
        specs :
            Parameter specifications (for generating replacements).
        existing_obs :
            Already-completed observations for diversity.
        seed :
            Random seed.

        Returns
        -------
        ReplanResult with replacement points.
        """
        if len(original_batch) != len(results):
            raise ValueError(
                f"Batch size ({len(original_batch)}) and results size "
                f"({len(results)}) must match"
            )

        failed_indices = [i for i, ok in enumerate(results) if not ok]
        n_failed = len(failed_indices)

        if n_failed == 0:
            return ReplanResult(
                replacement_points=[],
                n_failed=0,
                n_replaced=0,
                strategy="none",
            )

        # Generate candidates by perturbing the successful points
        rng = random.Random(seed)
        successful = [
            original_batch[i] for i, ok in enumerate(results) if ok
        ]

        candidates = self._generate_candidates(
            successful, specs, n_candidates=n_failed * 5, rng=rng,
        )

        if not candidates:
            # No successful points to perturb; generate random candidates
            candidates = self._random_candidates(specs, n_failed * 5, rng)

        # Use diversifier to pick the most diverse replacements
        policy = self._diversifier.diversify(
            candidates, specs, n_select=n_failed,
            existing_obs=existing_obs, seed=seed,
        )

        return ReplanResult(
            replacement_points=policy.points,
            n_failed=n_failed,
            n_replaced=len(policy.points),
            strategy="perturb_successful" if successful else "random",
        )

    @staticmethod
    def _generate_candidates(
        successful: list[dict[str, float]],
        specs: list[ParameterSpec],
        n_candidates: int,
        rng: random.Random,
    ) -> list[dict[str, float]]:
        """Generate candidate points by perturbing successful ones."""
        if not successful:
            return []

        candidates: list[dict[str, float]] = []
        for _ in range(n_candidates):
            base = rng.choice(successful)
            point: dict[str, float] = {}
            for spec in specs:
                val = base.get(spec.name, 0.0)
                lo = spec.lower if spec.lower is not None else 0.0
                hi = spec.upper if spec.upper is not None else 1.0
                rng_size = hi - lo
                if rng_size > 0 and isinstance(val, (int, float)):
                    perturbation = rng.gauss(0, rng_size * 0.1)
                    new_val = max(lo, min(hi, float(val) + perturbation))
                    point[spec.name] = new_val
                else:
                    point[spec.name] = float(val)
            candidates.append(point)

        return candidates

    @staticmethod
    def _random_candidates(
        specs: list[ParameterSpec],
        n_candidates: int,
        rng: random.Random,
    ) -> list[dict[str, float]]:
        """Generate random candidates within parameter bounds."""
        candidates: list[dict[str, float]] = []
        for _ in range(n_candidates):
            point: dict[str, float] = {}
            for spec in specs:
                lo = spec.lower if spec.lower is not None else 0.0
                hi = spec.upper if spec.upper is not None else 1.0
                point[spec.name] = rng.uniform(lo, hi)
            candidates.append(point)
        return candidates


# ---------------------------------------------------------------------------
# Adaptive batch sizing (Pain Point 6)
# ---------------------------------------------------------------------------

@dataclass
class BatchSizeRecommendation:
    """Recommended batch size with reasoning."""
    batch_size: int
    reason: str
    phase_contribution: int
    noise_adjustment: int
    failure_adjustment: int


class AdaptiveBatchSizer:
    """Stage-aware and noise-aware batch size computation.

    Computes optimal batch size based on campaign phase, noise level,
    failure rate, and parameter dimensionality.

    Parameters
    ----------
    cold_start_multiplier :
        Multiply n_params by this for cold-start batch size.
    min_batch :
        Minimum batch size in any phase.
    max_batch :
        Maximum batch size cap.
    """

    def __init__(
        self,
        cold_start_multiplier: float = 2.0,
        min_batch: int = 1,
        max_batch: int = 20,
    ) -> None:
        self._cold_start_mult = cold_start_multiplier
        self._min_batch = min_batch
        self._max_batch = max_batch

    def compute_size(
        self,
        phase: Phase,
        n_params: int,
        noise_estimate: float = 0.0,
        failure_rate: float = 0.0,
        n_observations: int = 0,
    ) -> BatchSizeRecommendation:
        """Compute adaptive batch size.

        Parameters
        ----------
        phase :
            Current campaign phase.
        n_params :
            Number of parameters.
        noise_estimate :
            Coefficient of variation from diagnostics.
        failure_rate :
            Current failure rate.
        n_observations :
            Number of observations so far.

        Returns
        -------
        BatchSizeRecommendation
        """
        reasons: list[str] = []

        # Base size from phase
        if phase == Phase.COLD_START:
            base = max(2, int(n_params * self._cold_start_mult))
            reasons.append(f"cold_start: {n_params}*{self._cold_start_mult}")
        elif phase == Phase.EXPLOITATION:
            base = 1
            reasons.append("exploitation: focused search")
        elif phase == Phase.STAGNATION:
            base = max(3, n_params)
            reasons.append(f"stagnation: restart with {base}")
        elif phase == Phase.LEARNING:
            base = max(2, min(5, n_params))
            reasons.append(f"learning: balanced at {base}")
        else:
            base = max(1, n_params)
            reasons.append(f"default: {base}")

        phase_contribution = base

        # Noise adjustment: high noise → more replicates
        noise_adj = 0
        if noise_estimate > 0.5:
            noise_adj = 2
            reasons.append(f"high noise (+{noise_adj})")
        elif noise_estimate > 0.3:
            noise_adj = 1
            reasons.append(f"moderate noise (+{noise_adj})")

        # Failure adjustment: high failure rate → extra points
        fail_adj = 0
        if failure_rate > 0.3:
            fail_adj = max(1, int(base * failure_rate))
            reasons.append(f"high failures (+{fail_adj})")

        total = base + noise_adj + fail_adj
        total = max(self._min_batch, min(self._max_batch, total))

        return BatchSizeRecommendation(
            batch_size=total,
            reason="; ".join(reasons),
            phase_contribution=phase_contribution,
            noise_adjustment=noise_adj,
            failure_adjustment=fail_adj,
        )
