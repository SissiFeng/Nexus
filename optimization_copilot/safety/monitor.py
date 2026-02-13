"""Real-time safety monitoring for optimization experiments.

Evaluates parameter points against a HazardRegistry and generates
SafetyEvents when values approach or exceed safe operating boundaries.

Uses configurable warning and danger margins to provide early alerts
before hard limits are breached.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from optimization_copilot.safety.hazards import HazardRegistry, HazardSpec


class SafetyStatus(Enum):
    """Overall safety status of a parameter point."""

    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    EMERGENCY = "emergency"


@dataclass
class SafetyEvent:
    """Record of a safety-relevant event.

    Attributes
    ----------
    timestamp : float
        Time the event was generated (``time.time()``).
    status : SafetyStatus
        Severity of the event.
    parameter : str
        Name of the parameter that triggered the event.
    value : float
        Observed value of the parameter.
    limit : float
        The boundary (lower or upper safe limit) that was approached
        or exceeded.
    message : str
        Human-readable description of the event.
    """

    timestamp: float
    status: SafetyStatus
    parameter: str
    value: float
    limit: float
    message: str


class SafetyMonitor:
    """Monitors parameter points against a hazard registry.

    Generates SafetyEvents when parameter values approach (WARNING) or
    exceed (DANGER) safe operating boundaries. Multiple simultaneous
    DANGER events escalate to EMERGENCY status.

    Parameters
    ----------
    registry : HazardRegistry
        The hazard registry to monitor against.
    warning_margin : float
        Fraction of the safe range within which a WARNING is generated.
        For example, 0.1 means the inner 10% at each boundary triggers
        a warning. Default is 0.1.
    danger_margin : float
        Fraction of the safe range beyond which DANGER is triggered.
        A value of 0.0 means any value outside the safe range is DANGER.
        Default is 0.0.

    Examples
    --------
    >>> from optimization_copilot.safety.hazards import (
    ...     HazardRegistry, HazardSpec, HazardCategory, HazardLevel,
    ... )
    >>> registry = HazardRegistry()
    >>> registry.register(HazardSpec(
    ...     parameter_name="temperature",
    ...     category=HazardCategory.THERMAL,
    ...     level=HazardLevel.HIGH,
    ...     lower_safe=20.0,
    ...     upper_safe=200.0,
    ... ))
    >>> monitor = SafetyMonitor(registry, warning_margin=0.1)
    >>> status, events = monitor.check_point({"temperature": 195.0})
    >>> status
    <SafetyStatus.WARNING: 'warning'>
    """

    def __init__(
        self,
        registry: HazardRegistry,
        warning_margin: float = 0.1,
        danger_margin: float = 0.0,
    ) -> None:
        self._registry = registry
        self._warning_margin = warning_margin
        self._danger_margin = danger_margin
        self._event_history: list[SafetyEvent] = []

    def check_point(
        self, params: dict[str, float]
    ) -> tuple[SafetyStatus, list[SafetyEvent]]:
        """Evaluate a parameter point against all registered hazards.

        Parameters
        ----------
        params : dict[str, float]
            Mapping of parameter names to their values.

        Returns
        -------
        tuple[SafetyStatus, list[SafetyEvent]]
            The worst safety status across all hazards, and a list of
            events for any non-SAFE situations.
        """
        events: list[SafetyEvent] = []
        now = time.time()

        for hazard in self._registry.all_hazards():
            value = params.get(hazard.parameter_name)
            if value is None:
                continue

            event = self._evaluate_hazard(hazard, value, now)
            if event is not None:
                events.append(event)

        # Determine overall status
        if not events:
            return SafetyStatus.SAFE, events

        n_danger = sum(1 for e in events if e.status == SafetyStatus.DANGER)
        has_emergency = any(e.status == SafetyStatus.EMERGENCY for e in events)

        if has_emergency or n_danger > 1:
            # Multiple DANGER events escalate to EMERGENCY
            overall = SafetyStatus.EMERGENCY
        elif n_danger == 1:
            overall = SafetyStatus.DANGER
        elif any(e.status == SafetyStatus.WARNING for e in events):
            overall = SafetyStatus.WARNING
        else:
            overall = SafetyStatus.SAFE

        self._event_history.extend(events)
        return overall, events

    def check_batch(
        self, batch: list[dict[str, float]]
    ) -> list[dict[str, float]]:
        """Filter a batch of points, returning only safe ones.

        A point is considered acceptable if its overall status is
        SAFE or WARNING. Points with DANGER or EMERGENCY status
        are excluded.

        Parameters
        ----------
        batch : list[dict[str, float]]
            List of parameter points to evaluate.

        Returns
        -------
        list[dict[str, float]]
            Points that are SAFE or WARNING.
        """
        safe_points: list[dict[str, float]] = []
        for params in batch:
            status, _ = self.check_point(params)
            if status in (SafetyStatus.SAFE, SafetyStatus.WARNING):
                safe_points.append(params)
        return safe_points

    def safety_margin(self, params: dict[str, float]) -> dict[str, float]:
        """Compute the safety margin for each hazard parameter.

        The margin is defined as::

            min(value - lower_safe, upper_safe - value) / (upper_safe - lower_safe)

        A positive margin means the value is inside the safe range.
        A negative margin means the value is outside the safe range.

        Parameters
        ----------
        params : dict[str, float]
            Mapping of parameter names to their values.

        Returns
        -------
        dict[str, float]
            Mapping of parameter names to their safety margin fractions.
            Only parameters with registered hazards are included. If a
            parameter has multiple hazards, the minimum margin is used.
        """
        margins: dict[str, float] = {}
        for hazard in self._registry.all_hazards():
            value = params.get(hazard.parameter_name)
            if value is None:
                continue

            safe_range = hazard.upper_safe - hazard.lower_safe
            if safe_range <= 0:
                # Degenerate range; mark as zero margin
                margins[hazard.parameter_name] = 0.0
                continue

            dist_from_lower = value - hazard.lower_safe
            dist_from_upper = hazard.upper_safe - value
            margin = min(dist_from_lower, dist_from_upper) / safe_range

            # If multiple hazards for the same parameter, keep the minimum
            if hazard.parameter_name in margins:
                margins[hazard.parameter_name] = min(
                    margins[hazard.parameter_name], margin
                )
            else:
                margins[hazard.parameter_name] = margin

        return margins

    @property
    def event_history(self) -> list[SafetyEvent]:
        """All SafetyEvents generated by this monitor."""
        return list(self._event_history)

    def _evaluate_hazard(
        self, hazard: HazardSpec, value: float, timestamp: float
    ) -> SafetyEvent | None:
        """Evaluate a single hazard for a parameter value.

        Returns a SafetyEvent if the value is in WARNING or DANGER zone,
        or None if SAFE.
        """
        safe_range = hazard.upper_safe - hazard.lower_safe
        if safe_range <= 0:
            # Degenerate range: any value outside the single point is danger
            if value != hazard.lower_safe:
                return SafetyEvent(
                    timestamp=timestamp,
                    status=SafetyStatus.DANGER,
                    parameter=hazard.parameter_name,
                    value=value,
                    limit=hazard.lower_safe,
                    message=(
                        f"{hazard.parameter_name}={value} outside degenerate "
                        f"safe range [{hazard.lower_safe}, {hazard.upper_safe}]"
                    ),
                )
            return None

        warning_band = safe_range * self._warning_margin

        # Check if outside safe range (DANGER)
        if value < hazard.lower_safe:
            return SafetyEvent(
                timestamp=timestamp,
                status=SafetyStatus.DANGER,
                parameter=hazard.parameter_name,
                value=value,
                limit=hazard.lower_safe,
                message=(
                    f"{hazard.parameter_name}={value} below lower safe "
                    f"limit {hazard.lower_safe}"
                ),
            )
        if value > hazard.upper_safe:
            return SafetyEvent(
                timestamp=timestamp,
                status=SafetyStatus.DANGER,
                parameter=hazard.parameter_name,
                value=value,
                limit=hazard.upper_safe,
                message=(
                    f"{hazard.parameter_name}={value} above upper safe "
                    f"limit {hazard.upper_safe}"
                ),
            )

        # Check if within warning margin of boundaries (WARNING)
        if value < hazard.lower_safe + warning_band:
            return SafetyEvent(
                timestamp=timestamp,
                status=SafetyStatus.WARNING,
                parameter=hazard.parameter_name,
                value=value,
                limit=hazard.lower_safe,
                message=(
                    f"{hazard.parameter_name}={value} within warning margin "
                    f"of lower limit {hazard.lower_safe}"
                ),
            )
        if value > hazard.upper_safe - warning_band:
            return SafetyEvent(
                timestamp=timestamp,
                status=SafetyStatus.WARNING,
                parameter=hazard.parameter_name,
                value=value,
                limit=hazard.upper_safe,
                message=(
                    f"{hazard.parameter_name}={value} within warning margin "
                    f"of upper limit {hazard.upper_safe}"
                ),
            )

        return None
