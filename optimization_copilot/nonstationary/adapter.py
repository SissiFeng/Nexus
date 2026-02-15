"""Non-stationary adapter — integrates time weighting, seasonal detection, and drift
to assess non-stationarity and adapt optimization strategy decisions accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.models import (
    CampaignSnapshot,
    Phase,
    RiskPosture,
    StabilizeSpec,
    StrategyDecision,
)
from optimization_copilot.nonstationary.weighter import TimeWeighter, TimeWeights
from optimization_copilot.nonstationary.seasonal import SeasonalDetector, SeasonalPattern


# ── Data Structures ───────────────────────────────────────


@dataclass
class NonStationaryAssessment:
    """Consolidated non-stationarity assessment."""

    time_weights: TimeWeights
    seasonal_pattern: SeasonalPattern
    drift_report: Any  # DriftReport or None
    is_nonstationary: bool
    recommended_window: int
    recommended_strategy: str  # "static", "reweight", "sliding_window", "seasonal_adjust"
    adaptation_metadata: dict[str, Any] = field(default_factory=dict)


# ── Non-Stationary Adapter ────────────────────────────────


class NonStationaryAdapter:
    """Assess non-stationarity and adapt strategy decisions.

    Combines three complementary detectors:

    * :class:`TimeWeighter` — recency-based observation weighting
    * :class:`SeasonalDetector` — periodic pattern identification
    * :class:`DriftDetector` — concept drift detection (lazy-imported)

    Parameters
    ----------
    time_weighter : TimeWeighter | None
        Custom time weighter; defaults to exponential decay (rate=0.1).
    seasonal_detector : SeasonalDetector | None
        Custom seasonal detector; defaults to standard settings.
    drift_detector : Any
        A :class:`DriftDetector` instance.  If *None*, one is lazily
        imported from ``optimization_copilot.drift.detector`` with
        default parameters.
    """

    def __init__(
        self,
        time_weighter: TimeWeighter | None = None,
        seasonal_detector: SeasonalDetector | None = None,
        drift_detector: Any = None,
    ) -> None:
        self._time_weighter = time_weighter or TimeWeighter(
            strategy="exponential", decay_rate=0.1
        )
        self._seasonal_detector = seasonal_detector or SeasonalDetector()

        if drift_detector is not None:
            self._drift_detector = drift_detector
        else:
            # Lazy import to avoid hard circular dependency.
            from optimization_copilot.drift.detector import DriftDetector
            self._drift_detector = DriftDetector()

    # ── Public API ────────────────────────────────────────

    def assess(self, snapshot: CampaignSnapshot) -> NonStationaryAssessment:
        """Perform a full non-stationarity assessment on *snapshot*.

        Returns a :class:`NonStationaryAssessment` consolidating time
        weighting, seasonal detection, and drift analysis.
        """
        time_weights = self._time_weighter.compute_weights(snapshot)
        seasonal = self._seasonal_detector.detect(snapshot)

        # Run drift detection.
        drift_report: Any = None
        if self._drift_detector is not None:
            drift_report = self._drift_detector.detect(snapshot)

        n_obs = snapshot.n_observations

        # Determine strategy and recommended window.
        strategy = self._determine_strategy(drift_report, seasonal, time_weights)
        recommended_window = self._compute_recommended_window(
            snapshot, drift_report, seasonal
        )

        # Determine whether the campaign is non-stationary.
        drift_detected = getattr(drift_report, "drift_detected", False)
        is_nonstationary = (
            drift_detected
            or seasonal.detected
            or (n_obs > 0 and time_weights.effective_window < 0.5 * n_obs)
        )

        return NonStationaryAssessment(
            time_weights=time_weights,
            seasonal_pattern=seasonal,
            drift_report=drift_report,
            is_nonstationary=is_nonstationary,
            recommended_window=recommended_window,
            recommended_strategy=strategy,
            adaptation_metadata={
                "n_observations": n_obs,
                "effective_window": time_weights.effective_window,
                "drift_detected": drift_detected,
                "seasonal_detected": seasonal.detected,
            },
        )

    def adapt_decision(
        self,
        decision: StrategyDecision,
        assessment: NonStationaryAssessment,
    ) -> StrategyDecision:
        """Return a (potentially modified) strategy decision adapted for non-stationarity.

        If the assessment indicates stationarity, the original *decision*
        is returned unchanged.  Otherwise a new :class:`StrategyDecision`
        is created with adapted parameters.
        """
        if not assessment.is_nonstationary:
            return decision

        # Build a new decision — never mutate the original.
        new_stabilize = StabilizeSpec(
            noise_smoothing_window=decision.stabilize_spec.noise_smoothing_window,
            outlier_rejection_sigma=decision.stabilize_spec.outlier_rejection_sigma,
            failure_handling=decision.stabilize_spec.failure_handling,
            censored_data_policy=decision.stabilize_spec.censored_data_policy,
            constraint_tightening_rate=decision.stabilize_spec.constraint_tightening_rate,
            reweighting_strategy=decision.stabilize_spec.reweighting_strategy,
            retry_normalization=decision.stabilize_spec.retry_normalization,
        )

        exploration_strength = decision.exploration_strength
        reason_codes = list(decision.reason_codes)

        strategy = assessment.recommended_strategy

        # Apply strategy-specific adjustments.
        if strategy == "reweight":
            new_stabilize.reweighting_strategy = "recency"
        elif strategy in ("sliding_window", "seasonal_adjust"):
            new_stabilize.noise_smoothing_window = assessment.recommended_window

        # Drift-based exploration boost.
        drift_score = getattr(assessment.drift_report, "drift_score", 0.0)
        if drift_score > 0.3:
            exploration_strength = min(1.0, exploration_strength + 0.1)

        # Seasonal exploration boost.
        if assessment.seasonal_pattern.detected:
            exploration_strength = min(1.0, exploration_strength + 0.05)

        # Record the adaptation reason.
        reason_codes.append(f"nonstationary:{strategy}")

        return StrategyDecision(
            backend_name=decision.backend_name,
            stabilize_spec=new_stabilize,
            exploration_strength=exploration_strength,
            batch_size=decision.batch_size,
            batch_control_hints=dict(decision.batch_control_hints),
            risk_posture=decision.risk_posture,
            phase=decision.phase,
            reason_codes=reason_codes,
            fallback_events=list(decision.fallback_events),
            decision_metadata=dict(decision.decision_metadata),
        )

    # ── Private helpers ───────────────────────────────────

    def _determine_strategy(
        self,
        drift_report: Any,
        seasonal: SeasonalPattern,
        time_weights: TimeWeights,
    ) -> str:
        """Choose the best adaptation strategy given the detectors' outputs."""
        drift_detected = getattr(drift_report, "drift_detected", False)
        drift_score = getattr(drift_report, "drift_score", 0.0)

        if not drift_detected and not seasonal.detected:
            return "static"

        if seasonal.detected and not drift_detected:
            return "seasonal_adjust"

        if drift_detected and drift_score < 0.5:
            return "reweight"

        # Strong drift, or drift + seasonal.
        return "sliding_window"

    def _compute_recommended_window(
        self,
        snapshot: CampaignSnapshot,
        drift_report: Any,
        seasonal: SeasonalPattern,
    ) -> int:
        """Compute a recommended lookback window size."""
        n = snapshot.n_observations
        if n == 0:
            return 0

        base = max(5, n // 2)

        drift_score = getattr(drift_report, "drift_score", 0.0)
        if drift_score > 0.3:
            base = max(5, int(n * max(0.25, 1.0 - 0.75 * drift_score)))

        if seasonal.detected and seasonal.period:
            base = max(base, 2 * seasonal.period)

        return min(base, n)
