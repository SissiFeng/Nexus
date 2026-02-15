"""Emergency stop protocol for optimization experiments.

Evaluates safety events to determine whether an experiment should
continue, pause, fall back to safe parameters, or stop entirely.

Provides an append-only log of emergency evaluations for audit trails.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

from optimization_copilot.safety.monitor import SafetyEvent, SafetyStatus


class EmergencyAction(Enum):
    """Action to take in response to safety events."""

    CONTINUE = "continue"
    PAUSE = "pause"
    FALLBACK = "fallback"
    STOP = "stop"


@dataclass
class EmergencyEvaluation:
    """Result of evaluating safety events for emergency response.

    Attributes
    ----------
    action : EmergencyAction
        The recommended action.
    timestamp : float
        Time the evaluation was performed.
    n_warnings : int
        Number of WARNING-level events.
    n_dangers : int
        Number of DANGER-level events.
    n_emergencies : int
        Number of EMERGENCY-level events.
    message : str
        Human-readable summary of the evaluation.
    """

    action: EmergencyAction
    timestamp: float
    n_warnings: int
    n_dangers: int
    n_emergencies: int
    message: str


class EmergencyProtocol:
    """Evaluates safety events and determines emergency actions.

    Decision logic (evaluated in order of severity):
    1. Any EMERGENCY events -> STOP
    2. n_dangers >= n_dangers_to_stop -> STOP
    3. Any DANGER events -> FALLBACK
    4. n_warnings >= n_warnings_to_pause -> PAUSE
    5. Otherwise -> CONTINUE

    Parameters
    ----------
    n_warnings_to_pause : int
        Number of WARNING events that triggers a PAUSE action.
        Default is 3.
    n_dangers_to_stop : int
        Number of DANGER events that triggers a STOP action.
        Default is 1.
    fallback_params : dict[str, float] | None
        Safe fallback parameter values to suggest when FALLBACK
        action is recommended. Default is None.

    Examples
    --------
    >>> protocol = EmergencyProtocol(
    ...     n_warnings_to_pause=2,
    ...     n_dangers_to_stop=1,
    ...     fallback_params={"temperature": 100.0, "pressure": 1.0},
    ... )
    """

    def __init__(
        self,
        n_warnings_to_pause: int = 3,
        n_dangers_to_stop: int = 1,
        fallback_params: dict[str, float] | None = None,
    ) -> None:
        self._n_warnings_to_pause = n_warnings_to_pause
        self._n_dangers_to_stop = n_dangers_to_stop
        self._fallback_params = dict(fallback_params) if fallback_params else None

    def evaluate(self, events: list[SafetyEvent]) -> EmergencyEvaluation:
        """Evaluate a list of safety events and determine the action.

        Parameters
        ----------
        events : list[SafetyEvent]
            Safety events to evaluate.

        Returns
        -------
        EmergencyEvaluation
            The recommended action and supporting details.
        """
        n_warnings = sum(
            1 for e in events if e.status == SafetyStatus.WARNING
        )
        n_dangers = sum(
            1 for e in events if e.status == SafetyStatus.DANGER
        )
        n_emergencies = sum(
            1 for e in events if e.status == SafetyStatus.EMERGENCY
        )

        now = time.time()

        if n_emergencies > 0:
            return EmergencyEvaluation(
                action=EmergencyAction.STOP,
                timestamp=now,
                n_warnings=n_warnings,
                n_dangers=n_dangers,
                n_emergencies=n_emergencies,
                message=(
                    f"EMERGENCY STOP: {n_emergencies} emergency event(s) detected"
                ),
            )

        if n_dangers >= self._n_dangers_to_stop:
            return EmergencyEvaluation(
                action=EmergencyAction.STOP,
                timestamp=now,
                n_warnings=n_warnings,
                n_dangers=n_dangers,
                n_emergencies=n_emergencies,
                message=(
                    f"STOP: {n_dangers} danger event(s) >= "
                    f"threshold {self._n_dangers_to_stop}"
                ),
            )

        if n_dangers > 0:
            return EmergencyEvaluation(
                action=EmergencyAction.FALLBACK,
                timestamp=now,
                n_warnings=n_warnings,
                n_dangers=n_dangers,
                n_emergencies=n_emergencies,
                message=(
                    f"FALLBACK: {n_dangers} danger event(s) detected, "
                    f"reverting to safe parameters"
                ),
            )

        if n_warnings >= self._n_warnings_to_pause:
            return EmergencyEvaluation(
                action=EmergencyAction.PAUSE,
                timestamp=now,
                n_warnings=n_warnings,
                n_dangers=n_dangers,
                n_emergencies=n_emergencies,
                message=(
                    f"PAUSE: {n_warnings} warning event(s) >= "
                    f"threshold {self._n_warnings_to_pause}"
                ),
            )

        return EmergencyEvaluation(
            action=EmergencyAction.CONTINUE,
            timestamp=now,
            n_warnings=n_warnings,
            n_dangers=n_dangers,
            n_emergencies=n_emergencies,
            message="CONTINUE: no safety thresholds exceeded",
        )

    def suggest_fallback(self) -> dict[str, float] | None:
        """Return the configured fallback parameters, if any.

        Returns
        -------
        dict[str, float] | None
            A copy of the fallback parameters, or None if not configured.
        """
        if self._fallback_params is None:
            return None
        return dict(self._fallback_params)


class EmergencyLog:
    """Append-only log of emergency evaluations.

    Provides an audit trail of all emergency actions taken during
    an optimization campaign.

    Examples
    --------
    >>> log = EmergencyLog()
    >>> len(log)
    0
    >>> log.has_stop()
    False
    """

    def __init__(self) -> None:
        self._entries: list[EmergencyEvaluation] = []

    def log(self, evaluation: EmergencyEvaluation) -> None:
        """Append an emergency evaluation to the log.

        Parameters
        ----------
        evaluation : EmergencyEvaluation
            The evaluation to record.
        """
        self._entries.append(evaluation)

    def get_log(self) -> list[EmergencyEvaluation]:
        """Return all logged evaluations.

        Returns
        -------
        list[EmergencyEvaluation]
            A copy of all logged entries.
        """
        return list(self._entries)

    def has_stop(self) -> bool:
        """Check whether any STOP action has been logged.

        Returns
        -------
        bool
            True if any evaluation in the log has action STOP.
        """
        return any(e.action == EmergencyAction.STOP for e in self._entries)

    def latest(self) -> EmergencyEvaluation | None:
        """Return the most recent evaluation, or None if empty.

        Returns
        -------
        EmergencyEvaluation | None
            The last logged evaluation.
        """
        if not self._entries:
            return None
        return self._entries[-1]

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[EmergencyEvaluation]:
        return iter(self._entries)
