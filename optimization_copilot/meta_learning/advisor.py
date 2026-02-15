"""Top-level meta-learning orchestrator.

Combines all sub-learners (strategy, weight, threshold, failure, drift)
to produce ``MetaAdvice`` that can be injected into the ``MetaController``.
"""

from __future__ import annotations

import json
from typing import Any

from optimization_copilot.core.models import ProblemFingerprint
from optimization_copilot.drift.detector import DriftReport
from optimization_copilot.feasibility.taxonomy import FailureTaxonomy
from optimization_copilot.portfolio.scorer import ScoringWeights
from optimization_copilot.meta_learning.models import (
    CampaignOutcome,
    MetaAdvice,
    MetaLearningConfig,
)
from optimization_copilot.meta_learning.experience_store import ExperienceStore
from optimization_copilot.meta_learning.strategy_learner import StrategyLearner
from optimization_copilot.meta_learning.weight_tuner import WeightTuner
from optimization_copilot.meta_learning.threshold_learner import ThresholdLearner
from optimization_copilot.meta_learning.failure_learner import FailureStrategyLearner
from optimization_copilot.meta_learning.drift_learner import DriftRobustnessTracker


class MetaLearningAdvisor:
    """Orchestrates all meta-learners and produces unified MetaAdvice."""

    def __init__(
        self,
        experience_store: ExperienceStore | None = None,
        config: MetaLearningConfig | None = None,
    ) -> None:
        self._config = config or MetaLearningConfig()
        self._store = experience_store or ExperienceStore(self._config)
        self._strategy_learner = StrategyLearner(self._store, self._config)
        self._weight_tuner = WeightTuner(self._store, self._config)
        self._threshold_learner = ThresholdLearner(self._store, self._config)
        self._failure_learner = FailureStrategyLearner(self._store, self._config)
        self._drift_tracker = DriftRobustnessTracker(self._store, self._config)

    # ── Core ──────────────────────────────────────────────

    def advise(
        self,
        fingerprint: ProblemFingerprint,
        drift_report: DriftReport | None = None,
        failure_taxonomy: FailureTaxonomy | None = None,
    ) -> MetaAdvice:
        """Produce a MetaAdvice combining all sub-learner recommendations."""
        reason_codes: list[str] = []
        confidences: list[float] = []

        # 1. Strategy learner — backend ranking
        ranked = self._strategy_learner.rank_backends(fingerprint)
        recommended_backends = [name for name, _score in ranked]
        if recommended_backends:
            reason_codes.append(
                f"strategy_learner: ranked {len(recommended_backends)} backends"
            )
            confidences.append(0.8)
        else:
            reason_codes.append("strategy_learner: cold_start, no learned rankings")

        # 2. Weight tuner — scoring weights
        learned_weights = self._weight_tuner.suggest_weights(fingerprint)
        scoring_weights: ScoringWeights | None = None
        if learned_weights is not None:
            scoring_weights = self._weight_tuner.to_scoring_weights(learned_weights)
            reason_codes.append(
                f"weight_tuner: learned weights (n={learned_weights.n_campaigns}, "
                f"conf={learned_weights.confidence:.2f})"
            )
            confidences.append(learned_weights.confidence)
        else:
            reason_codes.append("weight_tuner: cold_start, using defaults")

        # 3. Threshold learner — switching thresholds
        learned_thresholds = self._threshold_learner.suggest_thresholds(fingerprint)
        switching_thresholds = None
        if learned_thresholds is not None:
            switching_thresholds = self._threshold_learner.to_switching_thresholds(
                learned_thresholds
            )
            reason_codes.append(
                f"threshold_learner: learned thresholds (n={learned_thresholds.n_campaigns})"
            )
            confidences.append(0.7)
        else:
            reason_codes.append("threshold_learner: cold_start, using defaults")

        # 4. Failure strategy learner
        failure_adjustments: dict[str, str] = {}
        if failure_taxonomy is not None:
            all_strategies = self._failure_learner.suggest_all()
            for ft_name in failure_taxonomy.type_counts:
                if ft_name in all_strategies:
                    failure_adjustments[ft_name] = (
                        all_strategies[ft_name].best_stabilization
                    )
            if failure_adjustments:
                reason_codes.append(
                    f"failure_learner: {len(failure_adjustments)} failure strategies"
                )
                confidences.append(0.6)

        # 5. Drift robustness
        drift_robust_backends: list[str] = []
        if drift_report is not None and drift_report.drift_detected:
            resilience_ranking = self._drift_tracker.rank_by_resilience()
            drift_robust_backends = [dr.backend_name for dr in resilience_ranking]
            if drift_robust_backends:
                reason_codes.append(
                    f"drift_tracker: {len(drift_robust_backends)} drift-robust backends"
                )
                confidences.append(0.5)

        # 6. Overall confidence
        confidence = 0.0
        if confidences:
            confidence = sum(confidences) / len(confidences)

        return MetaAdvice(
            recommended_backends=recommended_backends,
            scoring_weights=scoring_weights,
            switching_thresholds=switching_thresholds,
            failure_adjustments=failure_adjustments,
            drift_robust_backends=drift_robust_backends,
            confidence=confidence,
            reason_codes=reason_codes,
        )

    # ── Learning ──────────────────────────────────────────

    def learn_from_outcome(
        self,
        outcome: CampaignOutcome,
        weights_used: ScoringWeights | None = None,
    ) -> None:
        """Record a campaign outcome and update all sub-learners."""
        # 1. Store the outcome
        self._store.record_outcome(outcome)

        # 2. Update weight tuner (needs to know which weights were used)
        if weights_used is None:
            weights_used = ScoringWeights()  # defaults
        self._weight_tuner.update_from_outcome(outcome, weights_used)

        # 3. Update threshold learner
        self._threshold_learner.update_from_outcome(outcome)

        # 4. Update failure learner
        self._failure_learner.update_from_outcome(outcome)

        # 5. Update drift tracker
        self._drift_tracker.update_from_outcome(outcome)

    # ── Introspection ─────────────────────────────────────

    def experience_count(self) -> int:
        """Return total number of stored campaign outcomes."""
        return self._store.count()

    def has_learned(self, fingerprint: ProblemFingerprint) -> bool:
        """Check if advisor has enough data to give learned advice."""
        return self._strategy_learner.has_enough_data(fingerprint)

    @property
    def experience_store(self) -> ExperienceStore:
        """Access the underlying experience store."""
        return self._store

    # ── Serialization ─────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "min_experiences_for_learning": self._config.min_experiences_for_learning,
                "similarity_decay": self._config.similarity_decay,
                "weight_learning_rate": self._config.weight_learning_rate,
                "threshold_learning_rate": self._config.threshold_learning_rate,
                "recency_halflife": self._config.recency_halflife,
            },
            "experience_store": self._store.to_dict(),
            "weight_tuner": self._weight_tuner.to_dict(),
            "threshold_learner": self._threshold_learner.to_dict(),
            "failure_learner": self._failure_learner.to_dict(),
            "drift_tracker": self._drift_tracker.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetaLearningAdvisor:
        config = MetaLearningConfig(**data.get("config", {}))
        store = ExperienceStore.from_dict(data.get("experience_store", {}))
        advisor = cls(experience_store=store, config=config)
        advisor._weight_tuner = WeightTuner.from_dict(
            data.get("weight_tuner", {}), store
        )
        advisor._threshold_learner = ThresholdLearner.from_dict(
            data.get("threshold_learner", {}), store, config
        )
        advisor._failure_learner = FailureStrategyLearner.from_dict(
            data.get("failure_learner", {}), store
        )
        advisor._drift_tracker = DriftRobustnessTracker.from_dict(
            data.get("drift_tracker", {}), store
        )
        return advisor

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> MetaLearningAdvisor:
        return cls.from_dict(json.loads(json_str))
