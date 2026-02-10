"""Drift-triggered strategy adaptation.

Maps DriftReport signals to concrete, reproducible strategy adjustments.
When drift is detected the adapter produces a list of ``DriftAction`` items
that can be applied to the current ``StrategyDecision`` — e.g. phase reset,
window shrink, exploration boost, re-screening, or backend switch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.models import (
    Phase,
    RiskPosture,
    StrategyDecision,
    StabilizeSpec,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DriftAction:
    """A single structured action triggered by drift detection."""

    action_type: str
    # One of: "phase_reset", "window_shrink", "exploration_boost",
    #         "re_screen", "backend_switch", "recency_reweight"
    parameters: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    severity_threshold: float = 0.0  # minimum drift severity that triggers this


@dataclass
class DriftAdaptation:
    """Complete adaptation result from drift analysis."""

    actions: list[DriftAction]
    original_phase: Phase
    adapted_phase: Phase | None  # None if unchanged
    exploration_delta: float  # how much to add to exploration
    reweighting: str | None  # new reweighting strategy, or None
    reason_codes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Strategy adapter
# ---------------------------------------------------------------------------

class DriftStrategyAdapter:
    """Produces strategy adjustments based on drift signals.

    Parameters
    ----------
    phase_reset_threshold : float
        Minimum severity to trigger a phase reset back to LEARNING.
    exploration_boost_threshold : float
        Minimum severity to boost exploration strength.
    rescreen_threshold : float
        Minimum severity to trigger re-screening of variables.
    backend_switch_threshold : float
        Minimum severity to recommend switching to a drift-robust backend.
    """

    def __init__(
        self,
        phase_reset_threshold: float = 0.7,
        exploration_boost_threshold: float = 0.3,
        rescreen_threshold: float = 0.5,
        backend_switch_threshold: float = 0.6,
    ) -> None:
        self.phase_reset_threshold = phase_reset_threshold
        self.exploration_boost_threshold = exploration_boost_threshold
        self.rescreen_threshold = rescreen_threshold
        self.backend_switch_threshold = backend_switch_threshold

    def adapt(
        self,
        drift_report: Any,
        current_phase: Phase,
        current_exploration: float = 0.5,
    ) -> DriftAdaptation:
        """Determine which actions to take based on drift report.

        Parameters
        ----------
        drift_report :
            A ``DriftReport`` with drift_detected, drift_score, drift_type,
            affected_parameters, recommended_action.
        current_phase :
            The phase before drift adaptation.
        current_exploration :
            Current exploration strength (0-1).

        Returns
        -------
        DriftAdaptation
        """
        actions: list[DriftAction] = []
        reason_codes: list[str] = []
        adapted_phase: Phase | None = None
        exploration_delta = 0.0
        reweighting: str | None = None

        severity = getattr(drift_report, "drift_score", 0.0)
        drift_detected = getattr(drift_report, "drift_detected", False)
        drift_type = getattr(drift_report, "drift_type", "none")
        affected_params = getattr(drift_report, "affected_parameters", [])

        if not drift_detected or severity < self.exploration_boost_threshold:
            return DriftAdaptation(
                actions=actions,
                original_phase=current_phase,
                adapted_phase=None,
                exploration_delta=0.0,
                reweighting=None,
                reason_codes=["drift_below_threshold"],
            )

        # -- Action 1: Recency reweighting (mild drift) ------------------
        if severity >= self.exploration_boost_threshold:
            reweighting = "recency"
            actions.append(DriftAction(
                action_type="recency_reweight",
                parameters={"strategy": "recency"},
                reason=f"Drift severity {severity:.2f} warrants recent-data emphasis",
                severity_threshold=self.exploration_boost_threshold,
            ))
            reason_codes.append("drift_recency_reweight")

        # -- Action 2: Exploration boost ----------------------------------
        if severity >= self.exploration_boost_threshold:
            # Scale boost by severity: +0.1 at threshold, up to +0.4 at 1.0.
            boost = 0.1 + 0.3 * min(
                (severity - self.exploration_boost_threshold)
                / (1.0 - self.exploration_boost_threshold + 1e-12),
                1.0,
            )
            # Don't boost if already high.
            if current_exploration < 0.8:
                exploration_delta = boost
                actions.append(DriftAction(
                    action_type="exploration_boost",
                    parameters={"delta": round(boost, 3)},
                    reason=f"Boost exploration by {boost:.3f} due to drift severity {severity:.2f}",
                    severity_threshold=self.exploration_boost_threshold,
                ))
                reason_codes.append("drift_exploration_boost")

        # -- Action 3: Re-screening (moderate drift) ---------------------
        if severity >= self.rescreen_threshold and len(affected_params) > 0:
            actions.append(DriftAction(
                action_type="re_screen",
                parameters={"affected_parameters": list(affected_params)},
                reason=f"Parameter relationships changed for: {affected_params}",
                severity_threshold=self.rescreen_threshold,
            ))
            reason_codes.append("drift_re_screen")

        # -- Action 4: Backend switch (high drift) -----------------------
        if severity >= self.backend_switch_threshold:
            # Recommend drift-robust backends.
            drift_robust = ["random", "latin_hypercube"]
            actions.append(DriftAction(
                action_type="backend_switch",
                parameters={"recommended_backends": drift_robust},
                reason=f"Severe drift ({severity:.2f}) — switch to space-filling backend",
                severity_threshold=self.backend_switch_threshold,
            ))
            reason_codes.append("drift_backend_switch")

        # -- Action 5: Phase reset (severe drift) ------------------------
        if severity >= self.phase_reset_threshold:
            if current_phase in (Phase.EXPLOITATION, Phase.STAGNATION):
                adapted_phase = Phase.LEARNING
                actions.append(DriftAction(
                    action_type="phase_reset",
                    parameters={
                        "from_phase": current_phase.value,
                        "to_phase": Phase.LEARNING.value,
                    },
                    reason=f"Severe drift ({severity:.2f}) resets {current_phase.value} → learning",
                    severity_threshold=self.phase_reset_threshold,
                ))
                reason_codes.append("drift_phase_reset")

        # -- Action 6: Window shrink (always when drift detected) ---------
        if severity >= self.exploration_boost_threshold:
            # Keep fraction = max(0.25, 1.0 - 0.75 * severity).
            keep_fraction = max(0.25, 1.0 - 0.75 * severity)
            actions.append(DriftAction(
                action_type="window_shrink",
                parameters={"keep_fraction": round(keep_fraction, 3)},
                reason=f"Shrink training window to {keep_fraction:.0%} of data",
                severity_threshold=self.exploration_boost_threshold,
            ))
            reason_codes.append("drift_window_shrink")

        return DriftAdaptation(
            actions=actions,
            original_phase=current_phase,
            adapted_phase=adapted_phase,
            exploration_delta=round(exploration_delta, 4),
            reweighting=reweighting,
            reason_codes=reason_codes,
        )

    def apply(
        self,
        decision: StrategyDecision,
        adaptation: DriftAdaptation,
    ) -> StrategyDecision:
        """Apply drift adaptation to an existing strategy decision.

        Returns a *new* StrategyDecision with drift-adapted fields.
        The original decision is not mutated.
        """
        # Start from current values.
        new_phase = adaptation.adapted_phase or decision.phase
        new_exploration = min(
            1.0,
            max(0.0, decision.exploration_strength + adaptation.exploration_delta),
        )

        # Build new stabilize spec with recency reweighting if needed.
        new_stabilize = decision.stabilize_spec
        if adaptation.reweighting is not None:
            new_stabilize = StabilizeSpec(
                noise_smoothing_window=decision.stabilize_spec.noise_smoothing_window,
                outlier_rejection_sigma=decision.stabilize_spec.outlier_rejection_sigma,
                failure_handling=decision.stabilize_spec.failure_handling,
                censored_data_policy=decision.stabilize_spec.censored_data_policy,
                constraint_tightening_rate=decision.stabilize_spec.constraint_tightening_rate,
                reweighting_strategy=adaptation.reweighting,
                retry_normalization=decision.stabilize_spec.retry_normalization,
            )

        # Check for backend switch recommendation.
        new_backend = decision.backend_name
        for action in adaptation.actions:
            if action.action_type == "backend_switch":
                recommended = action.parameters.get("recommended_backends", [])
                if recommended:
                    new_backend = recommended[0]

        # Risk posture: more conservative under drift.
        new_risk = decision.risk_posture
        if adaptation.adapted_phase is not None:
            new_risk = RiskPosture.CONSERVATIVE

        # Merge reason codes.
        new_reasons = list(decision.reason_codes) + adaptation.reason_codes

        # Merge drift actions into decision metadata.
        new_metadata = dict(decision.decision_metadata)
        new_metadata["drift_adaptation"] = {
            "n_actions": len(adaptation.actions),
            "action_types": [a.action_type for a in adaptation.actions],
            "exploration_delta": adaptation.exploration_delta,
            "phase_changed": adaptation.adapted_phase is not None,
        }

        return StrategyDecision(
            backend_name=new_backend,
            stabilize_spec=new_stabilize,
            exploration_strength=round(new_exploration, 4),
            batch_size=decision.batch_size,
            batch_control_hints=decision.batch_control_hints,
            risk_posture=new_risk,
            phase=new_phase,
            reason_codes=new_reasons,
            fallback_events=list(decision.fallback_events),
            decision_metadata=new_metadata,
        )
