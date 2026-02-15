"""Failure-strategy learner — maps failure types to effective stabilization approaches.

Tracks which stabilization techniques work best for each failure type
across campaigns, enabling the system to proactively apply the right
stabilization when familiar failure patterns are detected.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from optimization_copilot.meta_learning.models import (
    CampaignOutcome,
    FailureStrategy,
    MetaLearningConfig,
)
from optimization_copilot.meta_learning.experience_store import ExperienceStore


class FailureStrategyLearner:
    """Learns which stabilization approaches best address each failure type.

    Maintains a mapping of failure_type -> best FailureStrategy by tracking
    the quality outcomes of different stabilization techniques across campaigns.
    """

    def __init__(
        self,
        experience_store: ExperienceStore,
        config: MetaLearningConfig | None = None,
    ) -> None:
        self._experience_store = experience_store
        self._config = config or MetaLearningConfig()
        self._strategies: dict[str, FailureStrategy] = {}
        self._strategy_scores: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

    # ── Queries ────────────────────────────────────────────

    def suggest_stabilization(self, failure_type: str) -> FailureStrategy | None:
        """Return the best stabilization strategy for *failure_type*, or None.

        Returns ``None`` when insufficient data has been accumulated
        (fewer than ``config.min_experiences_for_learning`` campaigns).
        """
        strategy = self._strategies.get(failure_type)
        if strategy is None:
            return None
        if strategy.n_campaigns < self._config.min_experiences_for_learning:
            return None
        return strategy

    def suggest_all(self) -> dict[str, FailureStrategy]:
        """Return all learned strategies that have enough supporting data."""
        min_n = self._config.min_experiences_for_learning
        return {
            ft: strat
            for ft, strat in self._strategies.items()
            if strat.n_campaigns >= min_n
        }

    # ── Learning ───────────────────────────────────────────

    def update_from_outcome(self, outcome: CampaignOutcome) -> None:
        """Integrate a completed campaign outcome into the learned strategies.

        For every failure type that occurred (count > 0) in this campaign,
        record the quality achieved by each stabilization backend.  Then
        recompute the best strategy for that failure type.
        """
        for failure_type, count in outcome.failure_type_counts.items():
            if count <= 0:
                continue

            # Record quality for each stabilization used in this campaign.
            for backend, stabilization_key in outcome.stabilization_used.items():
                outcome_quality = outcome.best_kpi / max(outcome.total_iterations, 1)
                self._strategy_scores[failure_type][stabilization_key].append(
                    outcome_quality
                )

            # Recompute the best strategy for this failure type.
            self._recompute_strategy(failure_type)

    # ── Internal ───────────────────────────────────────────

    def _recompute_strategy(self, failure_type: str) -> None:
        """Pick the stabilization with the highest average quality score."""
        scores_by_stab = self._strategy_scores.get(failure_type, {})
        if not scores_by_stab:
            return

        best_key: str | None = None
        best_avg: float = float("-inf")

        for stabilization_key, scores in scores_by_stab.items():
            if not scores:
                continue
            avg = sum(scores) / len(scores)
            if avg > best_avg:
                best_avg = avg
                best_key = stabilization_key

        if best_key is not None:
            # n_campaigns is the total observations for the winning stabilization.
            self._strategies[failure_type] = FailureStrategy(
                failure_type=failure_type,
                best_stabilization=best_key,
                effectiveness_score=best_avg,
                n_campaigns=len(scores_by_stab[best_key]),
            )

    # ── Serialization ──────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize learner state to a plain dict."""
        return {
            "strategies": {
                ft: strat.to_dict() for ft, strat in self._strategies.items()
            },
            "strategy_scores": {
                ft: {sk: list(scores) for sk, scores in by_stab.items()}
                for ft, by_stab in self._strategy_scores.items()
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        experience_store: ExperienceStore,
    ) -> FailureStrategyLearner:
        """Reconstruct a learner from a serialized dict."""
        learner = cls(experience_store=experience_store)

        # Restore strategies.
        for ft, strat_data in data.get("strategies", {}).items():
            learner._strategies[ft] = FailureStrategy.from_dict(strat_data)

        # Restore score history.
        for ft, by_stab in data.get("strategy_scores", {}).items():
            for sk, scores in by_stab.items():
                learner._strategy_scores[ft][sk] = list(scores)

        return learner
