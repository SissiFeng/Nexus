"""LLM safety wrapper for agent feedback validation.

Provides confidence gating and physics-based validation for
agent feedback before it reaches the optimization loop.
"""

from __future__ import annotations

import math
from typing import Any

from optimization_copilot.agents.base import (
    AgentContext,
    OptimizationFeedback,
)


class LLMSafetyWrapper:
    """Wraps agent feedback with confidence gating and physics checks.

    Validates feedback against configurable thresholds and optional
    domain-specific physical constraints before allowing it to reach
    the optimization loop.

    Parameters
    ----------
    confidence_threshold : float
        Minimum confidence for feedback to pass validation.
    enable_physics_check : bool
        Whether to check parameter suggestions against physical bounds.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        enable_physics_check: bool = True,
        execution_guard: Any = None,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._enable_physics_check = enable_physics_check
        self._execution_guard = execution_guard

    @property
    def confidence_threshold(self) -> float:
        """Current confidence threshold."""
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {value}")
        self._confidence_threshold = value

    def validate_feedback(
        self,
        feedback: OptimizationFeedback,
        context: AgentContext | None = None,
    ) -> tuple[bool, list[str]]:
        """Validate feedback against safety criteria.

        Parameters
        ----------
        feedback : OptimizationFeedback
            The feedback to validate.
        context : AgentContext | None
            Optional context for physics-based checks.

        Returns
        -------
        tuple[bool, list[str]]
            ``(is_valid, reasons)`` where ``reasons`` lists any
            validation failures.
        """
        reasons: list[str] = []

        # Check 1: Confidence threshold
        if feedback.confidence < self._confidence_threshold:
            reasons.append(
                f"Confidence {feedback.confidence:.3f} below threshold "
                f"{self._confidence_threshold:.3f}"
            )

        # Check 2: Non-empty payload for actionable feedback types
        actionable_types = {"prior_update", "constraint_addition", "reweight"}
        if feedback.feedback_type in actionable_types and not feedback.payload:
            reasons.append(
                f"Empty payload for actionable feedback type '{feedback.feedback_type}'"
            )

        # Check 3: Valid confidence range
        if not 0.0 <= feedback.confidence <= 1.0:
            reasons.append(
                f"Confidence {feedback.confidence} outside valid range [0, 1]"
            )

        # Check 4: Agent name present
        if not feedback.agent_name:
            reasons.append("Missing agent_name")

        # Check 5: Feedback type is valid
        valid_types = {"prior_update", "constraint_addition", "reweight", "hypothesis", "warning"}
        if feedback.feedback_type not in valid_types:
            reasons.append(
                f"Unknown feedback type: '{feedback.feedback_type}'"
            )

        # Check 6: Physics bounds check
        if self._enable_physics_check and context is not None:
            physics_issues = self._check_physics_bounds(feedback, context)
            reasons.extend(physics_issues)

        # Check 7: Execution trace validation
        if self._execution_guard is not None:
            trace_valid, trace_issues = self._execution_guard.validate_feedback(
                feedback
            )
            if not trace_valid:
                reasons.extend(trace_issues)

        is_valid = len(reasons) == 0
        return is_valid, reasons

    def gate_feedback(
        self,
        feedback: OptimizationFeedback,
        context: AgentContext | None = None,
    ) -> OptimizationFeedback | None:
        """Return feedback if valid, None if rejected.

        Parameters
        ----------
        feedback : OptimizationFeedback
            The feedback to gate.
        context : AgentContext | None
            Optional context for physics checks.

        Returns
        -------
        OptimizationFeedback | None
            The original feedback if valid, *None* if rejected.
        """
        is_valid, _ = self.validate_feedback(feedback, context)
        return feedback if is_valid else None

    def validate_batch(
        self,
        feedbacks: list[OptimizationFeedback],
        context: AgentContext | None = None,
    ) -> list[OptimizationFeedback]:
        """Filter a batch of feedbacks, returning only valid ones.

        Parameters
        ----------
        feedbacks : list[OptimizationFeedback]
            Feedbacks to validate.
        context : AgentContext | None
            Optional context for physics checks.

        Returns
        -------
        list[OptimizationFeedback]
            Only feedbacks that pass all validation checks.
        """
        results: list[OptimizationFeedback] = []
        for fb in feedbacks:
            gated = self.gate_feedback(fb, context)
            if gated is not None:
                results.append(gated)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_physics_bounds(
        feedback: OptimizationFeedback,
        context: AgentContext,
    ) -> list[str]:
        """Check parameter suggestions against physical constraints.

        Looks for parameter values in the feedback payload and validates
        them against DomainConfig constraints.
        """
        issues: list[str] = []

        if context.domain_config is None:
            return issues

        constraints = context.domain_config.get_constraints()
        if not constraints:
            return issues

        # Check prior_update payloads
        param_priors = feedback.payload.get("parameter_priors", {})
        for param_name, prior_info in param_priors.items():
            if param_name not in constraints:
                continue

            c = constraints[param_name]
            mean = prior_info.get("mean")
            std = prior_info.get("std")

            if mean is not None:
                try:
                    fmean = float(mean)
                except (TypeError, ValueError):
                    continue

                if not math.isfinite(fmean):
                    issues.append(
                        f"Parameter '{param_name}' has non-finite mean: {fmean}"
                    )
                    continue

                pmin = c.get("min")
                pmax = c.get("max")

                if pmin is not None:
                    try:
                        fmin = float(pmin)
                        if fmean < fmin:
                            issues.append(
                                f"Parameter '{param_name}' mean {fmean} below "
                                f"physical minimum {fmin}"
                            )
                    except (TypeError, ValueError):
                        pass

                if pmax is not None:
                    try:
                        fmax = float(pmax)
                        if fmean > fmax:
                            issues.append(
                                f"Parameter '{param_name}' mean {fmean} above "
                                f"physical maximum {fmax}"
                            )
                    except (TypeError, ValueError):
                        pass

            if std is not None:
                try:
                    fstd = float(std)
                    if fstd < 0:
                        issues.append(
                            f"Parameter '{param_name}' has negative std: {fstd}"
                        )
                    elif not math.isfinite(fstd):
                        issues.append(
                            f"Parameter '{param_name}' has non-finite std: {fstd}"
                        )
                except (TypeError, ValueError):
                    pass

        return issues
