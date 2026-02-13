"""Interactive optimization steering for human-in-the-loop workflows.

Allows humans to redirect the optimization process by accepting,
rejecting, modifying, or constraining candidate suggestions in real time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from optimization_copilot.core.models import ParameterSpec


class SteeringAction(Enum):
    """Actions a human can take to steer optimization."""

    ACCEPT = "accept"
    REJECT = "reject"
    MODIFY = "modify"
    FOCUS_REGION = "focus_region"
    AVOID_REGION = "avoid_region"
    CHANGE_OBJECTIVE = "change_objective"


@dataclass
class SteeringDirective:
    """A single steering instruction from a human operator.

    Attributes:
        action: The type of steering action.
        parameters: For MODIFY actions, a mapping of parameter names to new values.
        region_bounds: For FOCUS_REGION/AVOID_REGION, parameter bounds as
            {param_name: (lower, upper)}.
        reason: Human-provided rationale for the directive.
        timestamp: When the directive was issued (epoch seconds).
    """

    action: SteeringAction
    parameters: dict[str, Any] = field(default_factory=dict)
    region_bounds: dict[str, tuple[float, float]] | None = None
    reason: str = ""
    timestamp: float = 0.0


class SteeringEngine:
    """Engine for applying human steering directives to candidate suggestions.

    Maintains a history of all directives and applies filtering/modification
    logic based on the action type.
    """

    def __init__(self) -> None:
        self._history: list[SteeringDirective] = []

    def apply_directive(
        self,
        directive: SteeringDirective,
        candidates: list[dict[str, float]],
        specs: list[ParameterSpec] | None = None,
    ) -> list[dict[str, float]]:
        """Apply a steering directive to a list of candidate suggestions.

        Args:
            directive: The steering directive to apply.
            candidates: List of candidate parameter dictionaries.
            specs: Optional parameter specifications (reserved for future use).

        Returns:
            Filtered or modified list of candidates.
        """
        self._history.append(directive)

        if directive.action == SteeringAction.ACCEPT:
            return list(candidates)

        if directive.action == SteeringAction.REJECT:
            return []

        if directive.action == SteeringAction.MODIFY:
            result: list[dict[str, float]] = []
            for candidate in candidates:
                modified = dict(candidate)
                for key, value in directive.parameters.items():
                    if key in modified:
                        modified[key] = value
                result.append(modified)
            return result

        if directive.action == SteeringAction.FOCUS_REGION:
            if directive.region_bounds is None:
                return list(candidates)
            return [
                c for c in candidates
                if self._within_bounds(c, directive.region_bounds)
            ]

        if directive.action == SteeringAction.AVOID_REGION:
            if directive.region_bounds is None:
                return list(candidates)
            return [
                c for c in candidates
                if not self._within_bounds(c, directive.region_bounds)
            ]

        if directive.action == SteeringAction.CHANGE_OBJECTIVE:
            # Handled at a higher level; pass through unchanged
            return list(candidates)

        return list(candidates)

    @staticmethod
    def _within_bounds(
        candidate: dict[str, float],
        bounds: dict[str, tuple[float, float]],
    ) -> bool:
        """Check whether a candidate falls within all specified bounds.

        Args:
            candidate: Parameter dictionary to check.
            bounds: Mapping of parameter names to (lower, upper) tuples.

        Returns:
            True if the candidate is within bounds for every specified parameter.
        """
        for param, (lo, hi) in bounds.items():
            if param not in candidate:
                continue
            if not (lo <= candidate[param] <= hi):
                return False
        return True

    @property
    def directive_history(self) -> list[SteeringDirective]:
        """Return the full history of applied directives."""
        return list(self._history)

    @property
    def n_directives(self) -> int:
        """Number of directives that have been applied."""
        return len(self._history)
