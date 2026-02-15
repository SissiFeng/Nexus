"""Preference learning via Bradley-Terry model with MM algorithm."""

from __future__ import annotations

import math
from collections import defaultdict

from optimization_copilot.core.models import CampaignSnapshot
from optimization_copilot.multi_objective.pareto import MultiObjectiveAnalyzer, ParetoResult

from .models import PairwisePreference, PreferenceModel, PreferenceRanking


class PreferenceLearner:
    """Learn utility scores from pairwise preferences using Bradley-Terry MM algorithm."""

    def __init__(
        self,
        max_iterations: int = 100,
        epsilon: float = 1e-6,
        prior_strength: float = 1.0,
    ) -> None:
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.prior_strength = prior_strength

    def fit(
        self,
        preferences: list[PairwisePreference],
        n_items: int,
    ) -> PreferenceModel:
        """Fit Bradley-Terry model using MM algorithm (Hunter 2004).

        Parameters
        ----------
        preferences:
            List of pairwise preference judgments.
        n_items:
            Total number of items to score.

        Returns
        -------
        PreferenceModel with fitted scores.
        """
        # Validate: winner must differ from loser
        for p in preferences:
            if p.winner_idx == p.loser_idx:
                raise ValueError(
                    f"winner_idx and loser_idx must differ, got {p.winner_idx}"
                )

        # No preferences: return uniform scores
        if not preferences:
            uniform = self.prior_strength
            max_val = uniform if uniform > 0 else 1.0
            scores = {i: uniform / max_val for i in range(n_items)}
            return PreferenceModel(
                scores=scores,
                n_preferences=0,
                n_items=n_items,
                converged=True,
                n_iterations=0,
                log_likelihood=0.0,
            )

        # Initialize scores
        scores = [self.prior_strength] * n_items

        # Build wins: total confidence-weighted wins per item
        wins = [0.0] * n_items
        for p in preferences:
            wins[p.winner_idx] += p.confidence

        # Build symmetric comparison matrix
        comparisons: dict[int, dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for p in preferences:
            comparisons[p.winner_idx][p.loser_idx] += p.confidence
            comparisons[p.loser_idx][p.winner_idx] += p.confidence

        # MM iterations
        converged = False
        n_iter = 0
        for iteration in range(self.max_iterations):
            n_iter = iteration + 1
            new_scores = list(scores)

            for i in range(n_items):
                if i not in comparisons or not comparisons[i]:
                    # No comparisons for this item: keep current score
                    new_scores[i] = scores[i]
                    continue

                numerator = wins[i]
                denominator = 0.0
                for j, comp_weight in comparisons[i].items():
                    denominator += comp_weight / (scores[i] + scores[j])

                if denominator > 0:
                    new_scores[i] = numerator / denominator
                else:
                    new_scores[i] = scores[i]

            # Normalize so max = 1.0
            max_score = max(new_scores) if new_scores else 1.0
            if max_score > 0:
                new_scores = [s / max_score for s in new_scores]
            else:
                new_scores = [1.0 for _ in new_scores]

            # Check convergence
            max_delta = max(
                abs(new_scores[i] - scores[i]) for i in range(n_items)
            )
            scores = new_scores

            if max_delta < self.epsilon:
                converged = True
                break

        # Compute log-likelihood
        log_likelihood = 0.0
        for p in preferences:
            denom = scores[p.winner_idx] + scores[p.loser_idx]
            if denom > 0:
                log_likelihood += (
                    math.log(scores[p.winner_idx] / denom) * p.confidence
                )
            else:
                log_likelihood = -math.inf
                break

        return PreferenceModel(
            scores=dict(enumerate(scores)),
            n_preferences=len(preferences),
            n_items=n_items,
            converged=converged,
            n_iterations=n_iter,
            log_likelihood=log_likelihood,
        )

    def rank_with_preferences(
        self,
        snapshot: CampaignSnapshot,
        preferences: list[PairwisePreference],
        pareto_result: ParetoResult | None = None,
    ) -> PreferenceRanking:
        """Rank observations combining Pareto dominance with learned preferences.

        Parameters
        ----------
        snapshot:
            Campaign snapshot with observations.
        preferences:
            Pairwise preference judgments.
        pareto_result:
            Pre-computed Pareto analysis. If None, will be computed.

        Returns
        -------
        PreferenceRanking combining dominance ranks with preference-based ordering.
        """
        obs = snapshot.successful_observations

        if pareto_result is None:
            pareto_result = MultiObjectiveAnalyzer().analyze(snapshot)

        n_items = len(obs)
        model = self.fit(preferences, n_items)

        # dominance_ranks from pareto_result has length == n_items
        dominance_ranks = pareto_result.dominance_ranks

        # Group indices by dominance rank
        groups: dict[int, list[int]] = defaultdict(list)
        for idx, rank in enumerate(dominance_ranks):
            groups[rank].append(idx)

        # Within each group, sort by utility score descending, tie-break by index
        for rank in groups:
            groups[rank].sort(key=lambda idx: (-model.scores.get(idx, 0.0), idx))

        # Concatenate in rank order (rank 1 first)
        ranked_indices: list[int] = []
        for rank in sorted(groups.keys()):
            ranked_indices.extend(groups[rank])

        # Convert groups to regular dict for the result
        preference_within_rank = dict(groups)

        return PreferenceRanking(
            ranked_indices=ranked_indices,
            utility_scores=model.scores,
            dominance_ranks=dominance_ranks,
            preference_within_rank=preference_within_rank,
            metadata={
                "converged": model.converged,
                "n_iterations": model.n_iterations,
            },
        )

    def add_preference(
        self,
        preferences: list[PairwisePreference],
        winner_idx: int,
        loser_idx: int,
        confidence: float = 1.0,
    ) -> list[PairwisePreference]:
        """Add a new pairwise preference to the list.

        Parameters
        ----------
        preferences:
            Existing list of preferences.
        winner_idx:
            Index of the preferred item.
        loser_idx:
            Index of the less-preferred item.
        confidence:
            Confidence weight for this preference.

        Returns
        -------
        New list with the added preference appended.
        """
        if winner_idx == loser_idx:
            raise ValueError(
                f"winner_idx and loser_idx must differ, got {winner_idx}"
            )
        if winner_idx < 0 or loser_idx < 0:
            raise ValueError(
                f"Indices must be non-negative, got winner={winner_idx}, loser={loser_idx}"
            )
        return preferences + [
            PairwisePreference(
                winner_idx=winner_idx,
                loser_idx=loser_idx,
                confidence=confidence,
            )
        ]
