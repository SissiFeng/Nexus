"""Meta-learning weight tuner for ScoringWeights per problem fingerprint class.

Learns optimal ``ScoringWeights`` (gain, fail, cost, drift, incompatibility)
from historical ``CampaignOutcome`` data using exponential moving averages.
When enough experience has been accumulated for a fingerprint class, the
tuner overrides the default scoring weights with learned values.
"""

from __future__ import annotations

from optimization_copilot.core.models import ProblemFingerprint
from optimization_copilot.portfolio.scorer import ScoringWeights
from optimization_copilot.meta_learning.models import (
    CampaignOutcome,
    LearnedWeights,
    MetaLearningConfig,
)
from optimization_copilot.meta_learning.experience_store import ExperienceStore


class WeightTuner:
    """Learns optimal ScoringWeights per problem fingerprint class.

    Uses EMA (exponential moving average) updates driven by campaign outcome
    signals to nudge scoring weights toward values that improve downstream
    backend selection quality.

    Parameters
    ----------
    experience_store :
        Shared cross-campaign experience store.
    config :
        Meta-learning hyper-parameters.  Falls back to defaults when *None*.
    """

    def __init__(
        self,
        experience_store: ExperienceStore,
        config: MetaLearningConfig | None = None,
    ) -> None:
        self._store = experience_store
        self._config = config or MetaLearningConfig()
        self._learned: dict[str, LearnedWeights] = {}

    # ── Public API ────────────────────────────────────────────

    def suggest_weights(
        self, fingerprint: ProblemFingerprint
    ) -> LearnedWeights | None:
        """Return learned weights for *fingerprint* if enough data exists.

        First tries exact key match, then falls back to similarity-based
        lookup using continuous fingerprint kernel.  Returns ``None``
        during cold start.
        """
        fingerprint_key = str(fingerprint.to_tuple())
        learned = self._learned.get(fingerprint_key)
        if learned is not None:
            if learned.n_campaigns >= self._config.min_experiences_for_learning:
                return learned

        # Similarity-based fallback: find most similar learned fingerprint
        best_similarity = 0.0
        best_weights: LearnedWeights | None = None
        for key, lw in self._learned.items():
            if lw.n_campaigns < self._config.min_experiences_for_learning:
                continue
            records = self._store.get_by_fingerprint(key)
            if not records:
                continue
            sim = self._store._fingerprint_similarity(
                fingerprint, records[0].outcome.fingerprint
            )
            if sim > best_similarity:
                best_similarity = sim
                best_weights = lw

        # Only return if similarity exceeds threshold (0.5)
        if best_weights is not None and best_similarity > 0.5:
            return best_weights
        return None

    def update_from_outcome(
        self, outcome: CampaignOutcome, weights_used: ScoringWeights
    ) -> None:
        """Update learned weights from a completed campaign.

        Computes outcome signals (failure rate, drift score, KPI efficiency)
        and nudges the current weight estimate via EMA toward targets that
        would have improved backend selection for this campaign.
        """
        fingerprint_key = str(outcome.fingerprint.to_tuple())
        lr = self._config.weight_learning_rate

        # -- Outcome signals -------------------------------------------
        backend_perfs = outcome.backend_performances
        if backend_perfs:
            avg_failure_rate = sum(
                bp.failure_rate for bp in backend_perfs
            ) / len(backend_perfs)
            max_drift_score = max(
                (bp.drift_score for bp in backend_perfs), default=0.0
            )
        else:
            avg_failure_rate = 0.0
            max_drift_score = 0.0

        outcome_quality = outcome.best_kpi / max(outcome.total_iterations, 1)

        # -- Current weights (seed from weights_used on first call) ----
        if fingerprint_key in self._learned:
            lw = self._learned[fingerprint_key]
            old_gain = lw.gain
            old_fail = lw.fail
            old_cost = lw.cost
            old_drift = lw.drift
            old_incompat = lw.incompatibility
            n_prev = lw.n_campaigns
        else:
            old_gain = weights_used.gain
            old_fail = weights_used.fail
            old_cost = weights_used.cost
            old_drift = weights_used.drift
            old_incompat = weights_used.incompatibility
            n_prev = 0

        # -- Determine adjustment targets ------------------------------
        # Nudge fail weight up when average failure rate is high.
        target_fail = old_fail
        if avg_failure_rate > 0.2:
            target_fail = min(old_fail + lr, 1.0)

        # Nudge drift weight up when max drift score is high.
        target_drift = old_drift
        if max_drift_score > 0.3:
            target_drift = min(old_drift + lr, 1.0)

        # Nudge gain toward a base level proportional to outcome quality.
        target_gain = max(0.0, min(outcome_quality, 1.0))

        # Cost and incompatibility hold steady (no explicit signal).
        target_cost = old_cost
        target_incompat = old_incompat

        # -- Apply EMA -------------------------------------------------
        new_gain = (1.0 - lr) * old_gain + lr * target_gain
        new_fail = (1.0 - lr) * old_fail + lr * target_fail
        new_cost = (1.0 - lr) * old_cost + lr * target_cost
        new_drift = (1.0 - lr) * old_drift + lr * target_drift
        new_incompat = (1.0 - lr) * old_incompat + lr * target_incompat

        # -- Normalize to sum to 1.0 -----------------------------------
        total = new_gain + new_fail + new_cost + new_drift + new_incompat
        if total > 0:
            new_gain /= total
            new_fail /= total
            new_cost /= total
            new_drift /= total
            new_incompat /= total
        else:
            # Degenerate case: reset to uniform.
            new_gain = new_fail = new_cost = new_drift = new_incompat = 0.2

        # -- Update bookkeeping ----------------------------------------
        n_campaigns = n_prev + 1
        confidence = min(1.0, n_campaigns / 10.0)

        self._learned[fingerprint_key] = LearnedWeights(
            fingerprint_key=fingerprint_key,
            gain=new_gain,
            fail=new_fail,
            cost=new_cost,
            drift=new_drift,
            incompatibility=new_incompat,
            n_campaigns=n_campaigns,
            confidence=confidence,
        )

    def to_scoring_weights(self, learned: LearnedWeights) -> ScoringWeights:
        """Convert a ``LearnedWeights`` instance to ``ScoringWeights``."""
        return ScoringWeights(
            gain=learned.gain,
            fail=learned.fail,
            cost=learned.cost,
            drift=learned.drift,
            incompatibility=learned.incompatibility,
        )

    # ── Serialization ─────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize internal state to a plain dict."""
        return {
            "learned": {
                key: lw.to_dict() for key, lw in self._learned.items()
            },
        }

    @classmethod
    def from_dict(
        cls, data: dict, experience_store: ExperienceStore
    ) -> WeightTuner:
        """Reconstruct a ``WeightTuner`` from a serialized dict."""
        tuner = cls(experience_store=experience_store)
        for key, lw_data in data.get("learned", {}).items():
            tuner._learned[key] = LearnedWeights.from_dict(lw_data)
        return tuner
