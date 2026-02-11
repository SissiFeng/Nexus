"""Multi-objective optimization: Pareto front tracking and dominance ranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.models import CampaignSnapshot, Observation


@dataclass
class ParetoResult:
    """Result of multi-objective analysis."""
    pareto_front: list[Observation]  # Non-dominated observations
    pareto_indices: list[int]  # Indices in original observation list
    dominance_ranks: list[int]  # Rank for each observation (1 = Pareto front)
    tradeoff_report: dict[str, dict[str, float]]  # obj_pair -> correlation info


class MultiObjectiveAnalyzer:
    """Analyze multi-objective optimization campaigns."""

    def analyze(
        self,
        snapshot: CampaignSnapshot,
        weights: dict[str, float] | None = None,
    ) -> ParetoResult:
        obs = snapshot.successful_observations
        directions = {
            name: d
            for name, d in zip(snapshot.objective_names, snapshot.objective_directions)
        }

        if len(obs) < 2 or len(snapshot.objective_names) < 2:
            return ParetoResult(
                pareto_front=list(obs),
                pareto_indices=list(range(len(obs))),
                dominance_ranks=[1] * len(obs),
                tradeoff_report={},
            )

        # Normalize objectives (flip sign for maximize so we always minimize)
        obj_values = self._extract_objectives(obs, snapshot.objective_names, directions)

        # Compute Pareto front
        pareto_mask = self._find_pareto_front(obj_values)
        pareto_front = [obs[i] for i, is_p in enumerate(pareto_mask) if is_p]
        pareto_indices = [i for i, is_p in enumerate(pareto_mask) if is_p]

        # Dominance ranking (non-dominated sorting)
        ranks = self._dominance_ranking(obj_values)

        # Tradeoff report
        tradeoff = self._compute_tradeoffs(
            obs, snapshot.objective_names, directions
        )

        return ParetoResult(
            pareto_front=pareto_front,
            pareto_indices=pareto_indices,
            dominance_ranks=ranks,
            tradeoff_report=tradeoff,
        )

    @staticmethod
    def _extract_objectives(
        obs: list[Observation],
        obj_names: list[str],
        directions: dict[str, str],
    ) -> list[list[float]]:
        """Extract objective values, flipping sign for maximize objectives."""
        result = []
        for o in obs:
            vals = []
            for name in obj_names:
                v = o.kpi_values.get(name, 0.0)
                if directions.get(name) == "maximize":
                    v = -v  # Flip so we always minimize
                vals.append(v)
            result.append(vals)
        return result

    @staticmethod
    def _dominates(a: list[float], b: list[float]) -> bool:
        """Check if solution a dominates solution b (all minimizing)."""
        at_least_one_better = False
        for ai, bi in zip(a, b):
            if ai > bi:
                return False
            if ai < bi:
                at_least_one_better = True
        return at_least_one_better

    def _find_pareto_front(self, obj_values: list[list[float]]) -> list[bool]:
        n = len(obj_values)
        is_pareto = [True] * n
        for i in range(n):
            if not is_pareto[i]:
                continue
            for j in range(n):
                if i == j or not is_pareto[j]:
                    continue
                if self._dominates(obj_values[j], obj_values[i]):
                    is_pareto[i] = False
                    break
        return is_pareto

    def _dominance_ranking(self, obj_values: list[list[float]]) -> list[int]:
        """Non-dominated sorting: assign ranks (1 = Pareto front)."""
        n = len(obj_values)
        ranks = [0] * n
        remaining = set(range(n))
        rank = 1

        while remaining:
            current_front = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i != j and self._dominates(obj_values[j], obj_values[i]):
                        dominated = True
                        break
                if not dominated:
                    current_front.append(i)

            if not current_front:
                for i in remaining:
                    ranks[i] = rank
                break

            for i in current_front:
                ranks[i] = rank
                remaining.discard(i)
            rank += 1

        return ranks

    @staticmethod
    def _compute_tradeoffs(
        obs: list[Observation],
        obj_names: list[str],
        directions: dict[str, str],
    ) -> dict[str, dict[str, float]]:
        """Compute pairwise correlation between objectives."""
        if len(obs) < 3:
            return {}

        tradeoffs: dict[str, dict[str, float]] = {}
        for i, name_a in enumerate(obj_names):
            for name_b in obj_names[i + 1:]:
                vals_a = [o.kpi_values.get(name_a, 0.0) for o in obs]
                vals_b = [o.kpi_values.get(name_b, 0.0) for o in obs]
                n = len(vals_a)
                mean_a = sum(vals_a) / n
                mean_b = sum(vals_b) / n
                std_a = (sum((v - mean_a) ** 2 for v in vals_a) / n) ** 0.5
                std_b = (sum((v - mean_b) ** 2 for v in vals_b) / n) ** 0.5

                if std_a < 1e-12 or std_b < 1e-12:
                    corr = 0.0
                else:
                    cov = sum(
                        (vals_a[j] - mean_a) * (vals_b[j] - mean_b) for j in range(n)
                    ) / n
                    corr = cov / (std_a * std_b)

                key = f"{name_a}_vs_{name_b}"
                tradeoffs[key] = {
                    "correlation": round(corr, 4),
                    "tradeoff": "conflict" if corr < -0.3 else (
                        "harmony" if corr > 0.3 else "independent"
                    ),
                }

        return tradeoffs

    def weighted_score(
        self,
        observation: Observation,
        obj_names: list[str],
        directions: dict[str, str],
        weights: dict[str, float],
    ) -> float:
        """Compute weighted scalarized score for an observation."""
        score = 0.0
        for name in obj_names:
            v = observation.kpi_values.get(name, 0.0)
            w = weights.get(name, 1.0)
            if directions.get(name) == "maximize":
                score += w * v
            else:
                score -= w * v
        return score


# ---------------------------------------------------------------------------
# Target band & aspiration-level navigation (Pain Point 5)
# ---------------------------------------------------------------------------

@dataclass
class TargetBand:
    """Acceptable range for an objective.

    Parameters
    ----------
    objective : str
        Name of the objective.
    lower : float | None
        Lower acceptable bound (inclusive).
    upper : float | None
        Upper acceptable bound (inclusive).
    """
    objective: str
    lower: float | None = None
    upper: float | None = None


@dataclass
class AspirationLevel:
    """Aspiration (ideal) target for an objective.

    Parameters
    ----------
    objective : str
        Name of the objective.
    target : float
        Desired aspiration value.
    tolerance : float
        Acceptable deviation from target (used for proximity scoring).
    """
    objective: str
    target: float
    tolerance: float = 0.0


@dataclass
class NavigationResult:
    """Result of Pareto front navigation / filtering."""
    selected: list[Observation]
    selected_indices: list[int]
    distances: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


class ParetoNavigator:
    """Navigate and focus on specific regions of the Pareto front."""

    def filter_to_band(
        self,
        observations: list[Observation],
        bands: list[TargetBand],
    ) -> NavigationResult:
        """Filter observations to those within all target bands.

        Parameters
        ----------
        observations :
            Observations to filter (typically Pareto-front members).
        bands :
            Target band constraints per objective.

        Returns
        -------
        NavigationResult with observations that satisfy all bands.
        """
        selected: list[Observation] = []
        indices: list[int] = []

        for i, obs in enumerate(observations):
            in_band = True
            for band in bands:
                val = obs.kpi_values.get(band.objective)
                if val is None:
                    continue
                if band.lower is not None and val < band.lower:
                    in_band = False
                    break
                if band.upper is not None and val > band.upper:
                    in_band = False
                    break
            if in_band:
                selected.append(obs)
                indices.append(i)

        return NavigationResult(
            selected=selected,
            selected_indices=indices,
            distances=[0.0] * len(selected),
            metadata={"n_bands": len(bands), "n_filtered": len(selected)},
        )

    def focus_region(
        self,
        observations: list[Observation],
        aspirations: list[AspirationLevel],
        directions: dict[str, str] | None = None,
    ) -> NavigationResult:
        """Rank observations by proximity to aspiration levels.

        Parameters
        ----------
        observations :
            Candidate observations.
        aspirations :
            Target aspiration levels per objective.
        directions :
            Objective directions (``"maximize"`` / ``"minimize"``).
            Used for normalization direction.

        Returns
        -------
        NavigationResult sorted by ascending distance to aspiration point.
        """
        if not observations or not aspirations:
            return NavigationResult(
                selected=list(observations),
                selected_indices=list(range(len(observations))),
                distances=[0.0] * len(observations),
            )

        # Compute objective ranges for normalization
        obj_ranges: dict[str, tuple[float, float]] = {}
        for asp in aspirations:
            vals = [
                obs.kpi_values.get(asp.objective, 0.0) for obs in observations
            ]
            lo, hi = min(vals), max(vals)
            obj_ranges[asp.objective] = (lo, hi)

        # Compute normalized distance from each observation to aspiration point
        distances: list[float] = []
        for obs in observations:
            dist_sq = 0.0
            for asp in aspirations:
                val = obs.kpi_values.get(asp.objective, 0.0)
                lo, hi = obj_ranges[asp.objective]
                rng = hi - lo
                if rng == 0:
                    norm_diff = 0.0
                else:
                    norm_diff = (val - asp.target) / rng
                dist_sq += norm_diff ** 2
            distances.append(dist_sq ** 0.5)

        # Sort by distance
        ranked = sorted(range(len(observations)), key=lambda i: distances[i])

        return NavigationResult(
            selected=[observations[i] for i in ranked],
            selected_indices=ranked,
            distances=[distances[i] for i in ranked],
            metadata={"n_aspirations": len(aspirations)},
        )

    @staticmethod
    def auto_aspiration(
        snapshot: CampaignSnapshot,
        progress_fraction: float | None = None,
    ) -> list[AspirationLevel]:
        """Automatically compute aspiration levels from campaign progress.

        Early in the campaign the aspiration is set to the median observed
        value (achievable target).  As the campaign progresses the aspiration
        tightens toward the best observed value.

        Parameters
        ----------
        snapshot :
            Campaign with observations.
        progress_fraction :
            Fraction of budget used (0.0 = start, 1.0 = done).
            If ``None``, estimated from observation count vs budget hints.

        Returns
        -------
        list[AspirationLevel] — one per objective.
        """
        obs = snapshot.successful_observations
        obj_names = snapshot.objective_names
        directions = {
            n: d for n, d in zip(obj_names, snapshot.objective_directions)
        }

        if not obs or not obj_names:
            return []

        # Estimate progress if not given
        if progress_fraction is None:
            budget = snapshot.metadata.get("budget", len(obs) * 2)
            progress_fraction = min(1.0, len(obs) / max(budget, 1))

        aspirations: list[AspirationLevel] = []
        for name in obj_names:
            vals = [o.kpi_values.get(name, 0.0) for o in obs]
            if not vals:
                continue

            maximize = directions.get(name, "maximize") == "maximize"
            best = max(vals) if maximize else min(vals)
            median_val = sorted(vals)[len(vals) // 2]

            # Blend: early → median, late → best
            alpha = progress_fraction  # 0 = median, 1 = best
            target = median_val + alpha * (best - median_val)

            # Tolerance shrinks with progress
            val_range = max(vals) - min(vals) if len(vals) > 1 else 1.0
            tolerance = val_range * (1.0 - progress_fraction) * 0.1

            aspirations.append(AspirationLevel(
                objective=name,
                target=target,
                tolerance=tolerance,
            ))

        return aspirations
