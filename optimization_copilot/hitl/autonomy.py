"""Progressive autonomy framework for human-in-the-loop optimization.

Provides a graduated trust model that adjusts the level of human
oversight based on tracked AI performance outcomes.
"""

from __future__ import annotations

from enum import IntEnum


class AutonomyLevel(IntEnum):
    """Levels of AI autonomy in the optimization loop.

    MANUAL: Human decides all parameters.
    SUPERVISED: AI suggests, human approves before execution.
    COLLABORATIVE: AI decides, human can veto.
    AUTONOMOUS: AI decides, human monitors passively.
    """

    MANUAL = 0
    SUPERVISED = 1
    COLLABORATIVE = 2
    AUTONOMOUS = 3


class AutonomyPolicy:
    """Policy governing transitions between autonomy levels.

    Controls when to escalate (give AI more autonomy) or
    de-escalate (require more human oversight) based on
    configurable thresholds.

    Attributes:
        escalation_threshold: Trust score above which escalation is considered.
        de_escalation_threshold: Trust score below which de-escalation is considered.
        consecutive_required: Number of consecutive good outcomes needed to escalate.
    """

    def __init__(
        self,
        initial_level: AutonomyLevel = AutonomyLevel.SUPERVISED,
        escalation_threshold: float = 0.8,
        de_escalation_threshold: float = 0.3,
        consecutive_required: int = 5,
    ) -> None:
        self._level = initial_level
        self.escalation_threshold = escalation_threshold
        self.de_escalation_threshold = de_escalation_threshold
        self.consecutive_required = consecutive_required

    @property
    def current_level(self) -> AutonomyLevel:
        """Return the current autonomy level."""
        return self._level

    def escalate(self) -> AutonomyLevel:
        """Move up one autonomy level, capped at AUTONOMOUS.

        Returns:
            The new autonomy level after escalation.
        """
        new_value = min(self._level.value + 1, AutonomyLevel.AUTONOMOUS.value)
        self._level = AutonomyLevel(new_value)
        return self._level

    def de_escalate(self) -> AutonomyLevel:
        """Move down one autonomy level, capped at MANUAL.

        Returns:
            The new autonomy level after de-escalation.
        """
        new_value = max(self._level.value - 1, AutonomyLevel.MANUAL.value)
        self._level = AutonomyLevel(new_value)
        return self._level

    def set_level(self, level: AutonomyLevel) -> None:
        """Manually override the current autonomy level.

        Args:
            level: The desired autonomy level.
        """
        self._level = level

    def should_consult_human(self, decision_type: str = "suggest") -> bool:
        """Determine whether a human should be consulted for this decision.

        Args:
            decision_type: The kind of decision being made. "critical" and
                "switch" decisions require human input at COLLABORATIVE level.

        Returns:
            True if human consultation is required.
        """
        if self._level == AutonomyLevel.MANUAL:
            return True
        if self._level == AutonomyLevel.SUPERVISED:
            return True
        if self._level == AutonomyLevel.COLLABORATIVE:
            return decision_type in ("critical", "switch")
        # AUTONOMOUS
        return False


class TrustTracker:
    """Tracks AI trustworthiness via an exponential moving average.

    Records outcomes of AI suggestions and computes a running trust
    score used to decide when to escalate or de-escalate autonomy.

    Attributes:
        decay_rate: Weight for exponential moving average (0 < decay_rate < 1).
    """

    def __init__(self, decay_rate: float = 0.95) -> None:
        self._decay_rate = decay_rate
        self._trust: float = 0.5
        self._consecutive_good: int = 0
        self._n_records: int = 0

    def record_outcome(
        self,
        ai_suggested: bool,
        human_approved: bool,
        result_quality: float,
    ) -> None:
        """Record the outcome of one optimization step.

        Updates trust score using an exponential moving average and
        tracks consecutive good outcomes.

        Args:
            ai_suggested: Whether the AI provided the suggestion.
            human_approved: Whether the human approved the suggestion.
            result_quality: Quality of the result, 0.0 to 1.0.
        """
        self._n_records += 1

        if ai_suggested and human_approved and result_quality >= 0.5:
            # Good outcome: blend trust upward
            self._trust = self._decay_rate * self._trust + (1.0 - self._decay_rate) * result_quality
            self._consecutive_good += 1
        elif ai_suggested and not human_approved:
            # Human rejected AI suggestion: trust goes down
            self._trust = self._decay_rate * self._trust + (1.0 - self._decay_rate) * 0.0
            self._consecutive_good = 0
        else:
            # Other cases (e.g., human suggested, or approved but low quality)
            self._trust = self._decay_rate * self._trust + (1.0 - self._decay_rate) * result_quality
            if result_quality >= 0.5:
                self._consecutive_good += 1
            else:
                self._consecutive_good = 0

    @property
    def trust_score(self) -> float:
        """Current trust score as an exponential moving average, in [0, 1]."""
        return self._trust

    @property
    def consecutive_good(self) -> int:
        """Number of consecutive good outcomes."""
        return self._consecutive_good

    @property
    def n_records(self) -> int:
        """Total number of recorded outcomes."""
        return self._n_records

    def should_escalate(self, threshold: float = 0.8, min_consecutive: int = 5) -> bool:
        """Check whether trust is high enough to escalate autonomy.

        Args:
            threshold: Minimum trust score required.
            min_consecutive: Minimum consecutive good outcomes required.

        Returns:
            True if both conditions are met.
        """
        return self._trust >= threshold and self._consecutive_good >= min_consecutive

    def should_de_escalate(self, threshold: float = 0.3) -> bool:
        """Check whether trust is low enough to de-escalate autonomy.

        Args:
            threshold: Trust score below which de-escalation is recommended.

        Returns:
            True if trust score is below the threshold.
        """
        return self._trust < threshold
