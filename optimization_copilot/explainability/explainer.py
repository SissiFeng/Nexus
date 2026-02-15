"""Explainability layer: human-readable decision reports without hallucination."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optimization_copilot.core.models import (
    Phase,
    ProblemFingerprint,
    StrategyDecision,
)


@dataclass
class DecisionReport:
    """Human-readable explanation of a StrategyDecision."""
    summary: str
    selected_strategy: str
    triggering_diagnostics: list[str]
    phase_transition: str | None
    risk_assessment: str
    coverage_status: str
    remaining_uncertainty: str
    details: dict[str, Any] = field(default_factory=dict)


PHASE_DESCRIPTIONS = {
    Phase.COLD_START: "Exploring the parameter space to gather initial data.",
    Phase.LEARNING: "Building surrogate models and discovering structure.",
    Phase.EXPLOITATION: "Aggressively optimizing near the best found region.",
    Phase.STAGNATION: "Progress has stalled; switching strategy or restarting.",
    Phase.TERMINATION: "Approaching termination criteria.",
}

RISK_DESCRIPTIONS = {
    "conservative": "Prioritizing safe, well-explored regions.",
    "moderate": "Balancing exploration and exploitation.",
    "aggressive": "Pursuing high-risk, high-reward regions.",
}


class DecisionExplainer:
    """Generate factual explanations for StrategyDecision.

    Only reports what the algorithm actually computed — no speculation.
    """

    def explain(
        self,
        decision: StrategyDecision,
        fingerprint: ProblemFingerprint | None = None,
        diagnostics: dict[str, float] | None = None,
        previous_phase: Phase | None = None,
    ) -> DecisionReport:
        # Strategy
        strategy_text = (
            f"Selected '{decision.backend_name}' with "
            f"exploration_strength={decision.exploration_strength:.2f}"
        )

        # Triggering diagnostics
        triggers = []
        if decision.reason_codes:
            triggers = [f"[{code}]" for code in decision.reason_codes]
        if diagnostics:
            key_signals = self._identify_key_signals(diagnostics)
            triggers.extend(key_signals)

        # Phase transition
        phase_transition = None
        if previous_phase and previous_phase != decision.phase:
            phase_transition = (
                f"{previous_phase.value} → {decision.phase.value}: "
                f"{PHASE_DESCRIPTIONS.get(decision.phase, '')}"
            )

        # Risk
        risk_text = RISK_DESCRIPTIONS.get(decision.risk_posture.value, "Unknown posture.")

        # Coverage
        coverage_text = self._assess_coverage(diagnostics)

        # Uncertainty
        uncertainty_text = self._assess_uncertainty(diagnostics, decision)

        # Summary
        summary_parts = [
            f"Phase: {decision.phase.value}.",
            strategy_text + ".",
        ]
        if phase_transition:
            summary_parts.append(f"Transitioned: {phase_transition}")
        if decision.fallback_events:
            summary_parts.append(
                f"Fallbacks triggered: {', '.join(decision.fallback_events)}."
            )
        summary = " ".join(summary_parts)

        # UQ health
        uq_health_text = self._assess_uq_health(diagnostics)

        return DecisionReport(
            summary=summary,
            selected_strategy=strategy_text,
            triggering_diagnostics=triggers,
            phase_transition=phase_transition,
            risk_assessment=risk_text,
            coverage_status=coverage_text,
            remaining_uncertainty=uncertainty_text,
            details={
                "backend": decision.backend_name,
                "phase": decision.phase.value,
                "exploration_strength": decision.exploration_strength,
                "batch_size": decision.batch_size,
                "reason_codes": decision.reason_codes,
                "fallback_events": decision.fallback_events,
                "uq_health": uq_health_text,
            },
        )

    @staticmethod
    def _identify_key_signals(diagnostics: dict[str, float]) -> list[str]:
        """Identify diagnostic signals that likely drove the decision."""
        key = []
        if diagnostics.get("convergence_trend", 0) < -0.1:
            key.append("convergence declining")
        if diagnostics.get("failure_rate", 0) > 0.3:
            key.append(f"high failure rate ({diagnostics['failure_rate']:.0%})")
        if diagnostics.get("kpi_plateau_length", 0) > 5:
            key.append(f"plateau for {int(diagnostics['kpi_plateau_length'])} iterations")
        if diagnostics.get("noise_estimate", 0) > 0.5:
            key.append("high noise detected")
        if diagnostics.get("exploration_coverage", 0) < 0.3:
            key.append("low exploration coverage")
        if diagnostics.get("variance_contraction", 1.0) < 0.5:
            key.append("variance contracting (converging)")
        if diagnostics.get("miscalibration_score", 0) > 0.3:
            key.append(f"UQ miscalibrated ({diagnostics['miscalibration_score']:.2f})")
        if diagnostics.get("overconfidence_rate", 0) > 0.3:
            key.append(f"UQ overconfident ({diagnostics['overconfidence_rate']:.2f})")
        snr = diagnostics.get("signal_to_noise_ratio", 0)
        if 0 < snr < 3.0:
            key.append(f"low SNR ({snr:.1f}) — consider repeat measurements")
        return key

    @staticmethod
    def _assess_coverage(diagnostics: dict[str, float] | None) -> str:
        if not diagnostics:
            return "Coverage data unavailable."
        cov = diagnostics.get("exploration_coverage", 0)
        if cov > 0.7:
            return f"Good coverage ({cov:.0%} of parameter space explored)."
        elif cov > 0.3:
            return f"Moderate coverage ({cov:.0%}). More exploration may help."
        else:
            return f"Low coverage ({cov:.0%}). Significant unexplored regions remain."

    @staticmethod
    def _assess_uq_health(diagnostics: dict[str, float] | None) -> str:
        """Assess uncertainty quantification calibration health."""
        if not diagnostics:
            return "UQ health data unavailable."
        miscal = diagnostics.get("miscalibration_score", 0)
        overconf = diagnostics.get("overconfidence_rate", 0)
        if miscal <= 0.1 and overconf <= 0.1:
            return (
                f"UQ well-calibrated (miscalibration={miscal:.2f}, "
                f"overconfidence={overconf:.2f})."
            )
        parts = []
        if miscal > 0.3:
            parts.append(
                f"prediction intervals are poorly calibrated ({miscal:.2f})"
            )
        elif miscal > 0.1:
            parts.append(
                f"mild calibration drift ({miscal:.2f})"
            )
        if overconf > 0.3:
            parts.append(
                f"model is overconfident ({overconf:.2f}) — "
                "more observations fall outside predicted bands than expected"
            )
        elif overconf > 0.1:
            parts.append(
                f"slight overconfidence ({overconf:.2f})"
            )
        if not parts:
            return (
                f"UQ acceptable (miscalibration={miscal:.2f}, "
                f"overconfidence={overconf:.2f})."
            )
        return "UQ concern: " + "; ".join(parts) + "."

    @staticmethod
    def _assess_uncertainty(
        diagnostics: dict[str, float] | None,
        decision: StrategyDecision,
    ) -> str:
        if not diagnostics:
            return "Uncertainty assessment unavailable."
        uncertainty = diagnostics.get("model_uncertainty", 0)
        noise = diagnostics.get("noise_estimate", 0)

        if uncertainty > 0.5 or noise > 0.5:
            return (
                f"High uncertainty (model={uncertainty:.2f}, noise={noise:.2f}). "
                "Recommendations should be treated with caution."
            )
        elif uncertainty > 0.2:
            return f"Moderate uncertainty (model={uncertainty:.2f}). Results are directionally reliable."
        else:
            return f"Low uncertainty (model={uncertainty:.2f}). High confidence in recommendations."
