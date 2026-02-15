"""Trial and TrialBatch models for the execution engine.

A Trial represents a single experimental evaluation of a parameter
configuration. Trials track state through their lifecycle (pending,
running, completed, failed, abandoned) and can be converted to core
Observation objects for integration with the optimization loop.

A TrialBatch groups multiple trials belonging to the same iteration,
enabling parallel evaluation and batch-level status tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from optimization_copilot.core.models import Observation, StrategyDecision


# ── Enums ──────────────────────────────────────────────


class TrialState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


# ── Trial ──────────────────────────────────────────────


@dataclass
class Trial:
    """A single experimental trial within an optimization campaign.

    Tracks the full lifecycle of evaluating one parameter configuration,
    from creation through execution to completion or failure.
    """

    trial_id: str
    iteration: int
    parameters: dict[str, Any]
    state: TrialState = TrialState.PENDING
    kpi_values: dict[str, float] = field(default_factory=dict)
    is_failure: bool = False
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    attempt: int = 1
    timestamp: float = 0.0

    def complete(
        self,
        kpi_values: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mark this trial as successfully completed.

        Args:
            kpi_values: Observed KPI measurements for this trial.
            metadata: Optional additional metadata to merge into
                the trial's existing metadata.
        """
        self.state = TrialState.COMPLETED
        self.kpi_values = kpi_values
        if metadata is not None:
            self.metadata.update(metadata)

    def fail(self, reason: str) -> None:
        """Mark this trial as failed.

        Args:
            reason: Human-readable explanation of the failure.
        """
        self.state = TrialState.FAILED
        self.is_failure = True
        self.failure_reason = reason

    def abandon(self, reason: str = "") -> None:
        """Mark this trial as abandoned.

        Abandoned trials are ones that will not be retried, distinct
        from failures which may be retried.

        Args:
            reason: Optional explanation for why the trial was abandoned.
        """
        self.state = TrialState.ABANDONED
        if reason:
            self.failure_reason = reason

    def to_observation(self) -> Observation:
        """Convert this trial to a core Observation.

        Works for both COMPLETED and FAILED trials. Failed trials
        produce Observations with ``is_failure=True`` and empty
        KPI values.

        Returns:
            An Observation representing this trial's outcome.

        Raises:
            ValueError: If the trial is still PENDING, RUNNING,
                or ABANDONED and cannot be converted.
        """
        if self.state not in (TrialState.COMPLETED, TrialState.FAILED):
            raise ValueError(
                f"Cannot convert trial in state {self.state.value} "
                f"to Observation. Only COMPLETED and FAILED trials "
                f"can be converted."
            )
        return Observation(
            iteration=self.iteration,
            parameters=dict(self.parameters),
            kpi_values=dict(self.kpi_values),
            qc_passed=not self.is_failure,
            is_failure=self.is_failure,
            failure_reason=self.failure_reason,
            timestamp=self.timestamp,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict:
        """Serialize the trial to a plain dictionary."""
        return {
            "trial_id": self.trial_id,
            "iteration": self.iteration,
            "parameters": dict(self.parameters),
            "state": self.state.value,
            "kpi_values": dict(self.kpi_values),
            "is_failure": self.is_failure,
            "failure_reason": self.failure_reason,
            "metadata": dict(self.metadata),
            "attempt": self.attempt,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Trial:
        """Reconstruct a Trial from a dictionary.

        Args:
            data: Dictionary as produced by ``to_dict()``.

        Returns:
            A new Trial instance.
        """
        data = data.copy()
        data["state"] = TrialState(data["state"])
        return cls(**data)


# ── TrialBatch ─────────────────────────────────────────


@dataclass
class TrialBatch:
    """A batch of trials evaluated together in a single iteration.

    Groups trials that share an iteration and were proposed by the
    same strategy decision, enabling batch-level status tracking.
    """

    batch_id: str
    iteration: int
    trials: list[Trial] = field(default_factory=list)
    strategy_decision: StrategyDecision | None = None

    @property
    def all_completed(self) -> bool:
        """True if every trial in the batch completed successfully."""
        return bool(self.trials) and all(
            t.state == TrialState.COMPLETED for t in self.trials
        )

    @property
    def all_failed(self) -> bool:
        """True if every trial in the batch failed."""
        return bool(self.trials) and all(
            t.state == TrialState.FAILED for t in self.trials
        )

    @property
    def n_completed(self) -> int:
        """Number of completed trials in the batch."""
        return sum(1 for t in self.trials if t.state == TrialState.COMPLETED)

    @property
    def n_failed(self) -> int:
        """Number of failed trials in the batch."""
        return sum(1 for t in self.trials if t.state == TrialState.FAILED)

    @property
    def n_pending(self) -> int:
        """Number of trials still pending in the batch."""
        return sum(1 for t in self.trials if t.state == TrialState.PENDING)

    def to_dict(self) -> dict:
        """Serialize the batch to a plain dictionary."""
        return {
            "batch_id": self.batch_id,
            "iteration": self.iteration,
            "trials": [t.to_dict() for t in self.trials],
            "strategy_decision": (
                self.strategy_decision.to_dict()
                if self.strategy_decision is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrialBatch:
        """Reconstruct a TrialBatch from a dictionary.

        Args:
            data: Dictionary as produced by ``to_dict()``.

        Returns:
            A new TrialBatch instance.
        """
        data = data.copy()
        data["trials"] = [Trial.from_dict(t) for t in data["trials"]]
        if data.get("strategy_decision") is not None:
            # StrategyDecision uses asdict-based serialization; reconstruct
            # by unpacking and restoring enum values.
            sd = data["strategy_decision"].copy()
            from optimization_copilot.core.models import (
                Phase,
                RiskPosture,
                StabilizeSpec,
            )

            sd["risk_posture"] = RiskPosture(sd["risk_posture"])
            sd["phase"] = Phase(sd["phase"])
            sd["stabilize_spec"] = StabilizeSpec(**sd["stabilize_spec"])
            data["strategy_decision"] = StrategyDecision(**sd)
        else:
            data["strategy_decision"] = None
        return cls(**data)
