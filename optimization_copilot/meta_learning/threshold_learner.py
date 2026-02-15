"""Learns optimal SwitchingThresholds from phase transition analysis.

Observes how campaigns transition between phases and uses the timings
of those transitions, combined with campaign outcome quality, to learn
fingerprint-specific threshold values via exponential moving averages.
"""

from __future__ import annotations

from typing import Any

from optimization_copilot.core.models import ProblemFingerprint
from optimization_copilot.meta_controller.controller import SwitchingThresholds
from optimization_copilot.meta_learning.models import (
    CampaignOutcome,
    LearnedThresholds,
    MetaLearningConfig,
)
from optimization_copilot.meta_learning.experience_store import ExperienceStore


class ThresholdLearner:
    """Learns optimal phase-switching thresholds from campaign outcomes.

    For each ``ProblemFingerprint`` class, maintains an EMA-updated set of
    ``LearnedThresholds`` derived from phase transition timings in campaigns
    that achieved good outcome quality (avg regret < 1.0).
    """

    def __init__(
        self,
        experience_store: ExperienceStore,
        config: MetaLearningConfig | None = None,
    ) -> None:
        self._experience_store = experience_store
        self._config = config or MetaLearningConfig()
        self._learned: dict[str, LearnedThresholds] = {}

    # ── Public API ─────────────────────────────────────────

    def suggest_thresholds(
        self, fingerprint: ProblemFingerprint
    ) -> LearnedThresholds | None:
        """Return learned thresholds for *fingerprint* if enough data exists.

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
        best_thresholds: LearnedThresholds | None = None
        for key, lt in self._learned.items():
            if lt.n_campaigns < self._config.min_experiences_for_learning:
                continue
            records = self._experience_store.get_by_fingerprint(key)
            if not records:
                continue
            sim = self._experience_store._fingerprint_similarity(
                fingerprint, records[0].outcome.fingerprint
            )
            if sim > best_similarity:
                best_similarity = sim
                best_thresholds = lt

        # Only return if similarity exceeds threshold (0.5)
        if best_thresholds is not None and best_similarity > 0.5:
            return best_thresholds
        return None

    def update_from_outcome(self, outcome: CampaignOutcome) -> None:
        """Update learned thresholds from a completed campaign outcome.

        Only learns from "good" outcomes where avg regret < 1.0 (quality > 1.0).
        Extracts threshold hints from phase transition timings and applies
        exponential moving average updates.
        """
        # Compute outcome quality from backend regrets.
        backend_regrets = [
            bp.regret for bp in outcome.backend_performances
        ]
        if not backend_regrets:
            return
        avg_regret = sum(backend_regrets) / len(backend_regrets)
        quality = 1.0 / max(avg_regret, 0.01)

        # Only learn from good outcomes (quality > 1.0 <==> avg_regret < 1.0).
        if quality <= 1.0:
            return

        fingerprint_key = str(outcome.fingerprint.to_tuple())
        lr = self._config.threshold_learning_rate

        # Extract threshold observations from phase transitions.
        observed_cold_start: float | None = None
        observed_plateau: float | None = None
        observed_gain: float | None = None

        for from_phase, to_phase, iteration in outcome.phase_transitions:
            if from_phase == "cold_start" and to_phase == "learning":
                # The iteration at which we left cold start suggests
                # the number of observations needed.
                observed_cold_start = float(iteration)

            elif from_phase == "learning" and to_phase == "exploitation":
                # Duration in the learning phase suggests plateau length.
                # Compute duration since the last relevant transition.
                learning_start = self._find_transition_iteration(
                    outcome.phase_transitions, "cold_start", "learning"
                )
                if learning_start is not None:
                    observed_plateau = float(iteration - learning_start)
                else:
                    observed_plateau = float(iteration)

            elif from_phase == "exploitation" and to_phase == "stagnation":
                # Use a proxy for the exploitation gain threshold: the
                # inverse of the exploitation duration suggests how fast
                # gains diminished (steeper slope = shorter duration).
                exploitation_start = self._find_transition_iteration(
                    outcome.phase_transitions, "learning", "exploitation"
                )
                if exploitation_start is not None:
                    duration = max(iteration - exploitation_start, 1)
                    observed_gain = -1.0 / duration

        # Nothing observed -- skip.
        if all(v is None for v in (observed_cold_start, observed_plateau, observed_gain)):
            return

        # Apply EMA updates (or initialize).
        existing = self._learned.get(fingerprint_key)

        if existing is None:
            # Initialize from the first observation.
            self._learned[fingerprint_key] = LearnedThresholds(
                fingerprint_key=fingerprint_key,
                cold_start_min_observations=(
                    observed_cold_start
                    if observed_cold_start is not None
                    else 10.0
                ),
                learning_plateau_length=(
                    observed_plateau
                    if observed_plateau is not None
                    else 5.0
                ),
                exploitation_gain_threshold=(
                    observed_gain
                    if observed_gain is not None
                    else -0.1
                ),
                n_campaigns=1,
            )
        else:
            # EMA update: new = (1 - lr) * old + lr * observed
            if observed_cold_start is not None:
                existing.cold_start_min_observations = (
                    (1.0 - lr) * existing.cold_start_min_observations
                    + lr * observed_cold_start
                )
            if observed_plateau is not None:
                existing.learning_plateau_length = (
                    (1.0 - lr) * existing.learning_plateau_length
                    + lr * observed_plateau
                )
            if observed_gain is not None:
                existing.exploitation_gain_threshold = (
                    (1.0 - lr) * existing.exploitation_gain_threshold
                    + lr * observed_gain
                )
            existing.n_campaigns += 1

    def to_switching_thresholds(
        self, learned: LearnedThresholds
    ) -> SwitchingThresholds:
        """Convert ``LearnedThresholds`` into a ``SwitchingThresholds`` instance."""
        return SwitchingThresholds(
            cold_start_min_observations=round(learned.cold_start_min_observations),
            learning_plateau_length=round(learned.learning_plateau_length),
            exploitation_improvement_slope=learned.exploitation_gain_threshold,
        )

    # ── Serialization ──────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the learner state to a plain dict."""
        return {
            "learned": {
                key: lt.to_dict() for key, lt in self._learned.items()
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        experience_store: ExperienceStore,
        config: MetaLearningConfig | None = None,
    ) -> ThresholdLearner:
        """Reconstruct a ``ThresholdLearner`` from a serialized dict."""
        learner = cls(experience_store=experience_store, config=config)
        for key, lt_data in data.get("learned", {}).items():
            learner._learned[key] = LearnedThresholds.from_dict(lt_data)
        return learner

    # ── Internal helpers ───────────────────────────────────

    @staticmethod
    def _find_transition_iteration(
        transitions: list[tuple[str, str, int]],
        from_phase: str,
        to_phase: str,
    ) -> int | None:
        """Find the iteration of a specific phase transition."""
        for fp, tp, iteration in transitions:
            if fp == from_phase and tp == to_phase:
                return iteration
        return None
