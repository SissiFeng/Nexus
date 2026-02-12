"""Acquisition-based candidate ranking.

Takes surrogate predictions (mean, std) and applies acquisition functions
(EI, UCB, PI) to produce a ranked table of candidates ordered by
experimental priority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------


@dataclass
class RankedCandidate:
    """A single candidate with its ranking information.

    Parameters
    ----------
    rank : int
        Position in the ranked list (1 = best).
    name : str
        Human-readable identifier.
    parameters : dict[str, Any]
        Full parameter dict for this candidate.
    predicted_mean : float
        Posterior mean from the surrogate.
    predicted_std : float
        Posterior standard deviation from the surrogate.
    acquisition_score : float
        Score from the acquisition function.
    """

    rank: int
    name: str
    parameters: dict[str, Any]
    predicted_mean: float
    predicted_std: float
    acquisition_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "name": self.name,
            "parameters": self.parameters,
            "predicted_mean": self.predicted_mean,
            "predicted_std": self.predicted_std,
            "acquisition_score": self.acquisition_score,
        }


@dataclass
class RankedTable:
    """Ranked list of candidates with metadata.

    Parameters
    ----------
    candidates : list[RankedCandidate]
        Candidates sorted by rank (ascending).
    objective_name : str
        Which objective was ranked.
    direction : str
        ``"minimize"`` or ``"maximize"``.
    acquisition_strategy : str
        Which acquisition function was used.
    best_observed : float | None
        Best objective value seen so far.
    """

    candidates: list[RankedCandidate]
    objective_name: str
    direction: str
    acquisition_strategy: str
    best_observed: float | None = None

    def top_n(self, n: int) -> list[RankedCandidate]:
        """Return the top-n candidates."""
        return self.candidates[:n]

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "objective_name": self.objective_name,
            "direction": self.direction,
            "acquisition_strategy": self.acquisition_strategy,
            "best_observed": self.best_observed,
            "n_candidates": self.n_candidates,
        }


# ------------------------------------------------------------------
# Ranker
# ------------------------------------------------------------------


class CandidateRanker:
    """Rank candidates by acquisition function score.

    Supports EI (Expected Improvement), UCB (Upper Confidence Bound),
    and PI (Probability of Improvement).  Handles both minimization and
    maximization objectives by internally mapping to a minimization
    convention before calling the acquisition functions.
    """

    def rank(
        self,
        candidate_names: list[str],
        candidate_params: list[dict[str, Any]],
        predictions: list[tuple[float, float]],
        objective_name: str,
        direction: str = "maximize",
        strategy: str = "ucb",
        kappa: float = 2.0,
        best_observed: float | None = None,
    ) -> RankedTable:
        """Rank candidates by acquisition score.

        Parameters
        ----------
        candidate_names : list[str]
            Human-readable names for each candidate.
        candidate_params : list[dict]
            Full parameter dicts for each candidate.
        predictions : list[tuple[float, float]]
            ``(mean, std)`` from the surrogate for each candidate.
        objective_name : str
            Name of the objective being ranked.
        direction : str
            ``"minimize"`` or ``"maximize"`` (default ``"maximize"``).
        strategy : str
            Acquisition function: ``"ucb"``, ``"ei"``, or ``"pi"``.
        kappa : float
            UCB exploration parameter (default 2.0).
        best_observed : float | None
            Best observed value so far (needed for EI and PI).

        Returns
        -------
        RankedTable
            Candidates sorted by acquisition priority (rank 1 = best).

        Raises
        ------
        ValueError
            If inputs have mismatched lengths or unknown strategy.
        """
        n = len(candidate_names)
        if len(candidate_params) != n or len(predictions) != n:
            raise ValueError(
                f"Mismatched lengths: {n} names, {len(candidate_params)} params, "
                f"{len(predictions)} predictions"
            )
        if strategy not in ("ucb", "ei", "pi"):
            raise ValueError(f"Unknown strategy: {strategy!r}")

        from optimization_copilot.backends._math.acquisition import (
            expected_improvement,
            probability_of_improvement,
            upper_confidence_bound,
        )

        # Map to minimization convention for acquisition functions.
        # For maximization: negate means so "lower mu" = "higher original value".
        if direction == "maximize":
            mapped_preds = [(-mu, sigma) for mu, sigma in predictions]
            best_y = -best_observed if best_observed is not None else None
        else:
            mapped_preds = list(predictions)
            best_y = best_observed

        # Compute acquisition scores
        scores: list[float] = []
        for mu_min, sigma in mapped_preds:
            if strategy == "ucb":
                # UCB for minimization: mu - kappa*sigma → lower is better
                scores.append(upper_confidence_bound(mu_min, sigma, kappa))
            elif strategy == "ei":
                if best_y is None:
                    scores.append(0.0)
                else:
                    scores.append(expected_improvement(mu_min, sigma, best_y))
            else:  # pi
                if best_y is None:
                    scores.append(0.0)
                else:
                    scores.append(probability_of_improvement(mu_min, sigma, best_y))

        # Sort: UCB → ascending (lower is better for minimisation)
        #        EI/PI → descending (higher is better)
        if strategy == "ucb":
            sorted_indices = sorted(range(n), key=lambda i: scores[i])
        else:
            sorted_indices = sorted(range(n), key=lambda i: -scores[i])

        ranked: list[RankedCandidate] = []
        for rank, idx in enumerate(sorted_indices, 1):
            mu_orig, sigma_orig = predictions[idx]
            ranked.append(RankedCandidate(
                rank=rank,
                name=candidate_names[idx],
                parameters=candidate_params[idx],
                predicted_mean=mu_orig,
                predicted_std=sigma_orig,
                acquisition_score=scores[idx],
            ))

        return RankedTable(
            candidates=ranked,
            objective_name=objective_name,
            direction=direction,
            acquisition_strategy=strategy,
            best_observed=best_observed,
        )
