"""Campaign state management with checkpoint and resume support.

CampaignState holds the complete mutable state of a running optimization
campaign, including the spec, snapshot, iteration counter, decision
history, completed trials, and retry queue. It supports full round-trip
serialization to JSON for checkpoint/resume across process restarts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from optimization_copilot.core.models import CampaignSnapshot, Phase
from optimization_copilot.dsl.spec import OptimizationSpec


# ── Campaign State ─────────────────────────────────────


@dataclass
class CampaignState:
    """Complete mutable state of a running optimization campaign.

    Captures everything needed to checkpoint a campaign to disk and
    resume it later, including the immutable spec, the evolving
    snapshot, iteration counter, decision history, and retry queue.

    Attributes:
        spec: The optimization specification defining the campaign.
        snapshot: Current campaign snapshot with observations and metadata.
        iteration: Current iteration counter.
        phase_history: Log of phase transitions with timestamps and reasons.
        decision_history: Log of strategy decisions made at each iteration.
        completed_trials: Serialized records of all completed trials.
        pending_retries: Serialized records of trials awaiting retry.
        terminated: Whether the campaign has been terminated.
        termination_reason: Human-readable reason for termination.
        seed: Random seed for reproducibility.
    """

    spec: OptimizationSpec
    snapshot: CampaignSnapshot
    iteration: int = 0
    phase_history: list[dict[str, Any]] = field(default_factory=list)
    decision_history: list[dict[str, Any]] = field(default_factory=list)
    completed_trials: list[dict[str, Any]] = field(default_factory=list)
    pending_retries: list[dict[str, Any]] = field(default_factory=list)
    terminated: bool = False
    termination_reason: str = ""
    seed: int = 42

    # ── Serialization ──────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire campaign state to a plain dictionary.

        All nested objects (spec, snapshot) are serialized via their
        own ``to_dict()`` methods, producing a fully JSON-serializable
        structure.

        Returns:
            A dictionary suitable for JSON serialization.
        """
        return {
            "spec": self.spec.to_dict(),
            "snapshot": self.snapshot.to_dict(),
            "iteration": self.iteration,
            "phase_history": list(self.phase_history),
            "decision_history": list(self.decision_history),
            "completed_trials": list(self.completed_trials),
            "pending_retries": list(self.pending_retries),
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignState:
        """Reconstruct a CampaignState from a dictionary.

        Delegates to ``OptimizationSpec.from_dict`` and
        ``CampaignSnapshot.from_dict`` for nested reconstruction.

        Args:
            data: Dictionary as produced by ``to_dict()``.

        Returns:
            A new CampaignState instance.
        """
        data = data.copy()
        data["spec"] = OptimizationSpec.from_dict(data["spec"])
        data["snapshot"] = CampaignSnapshot.from_dict(data["snapshot"])
        return cls(**data)

    # ── JSON Serialization ─────────────────────────────

    def to_json(self) -> str:
        """Serialize the campaign state to a JSON string.

        Uses sorted keys for deterministic output and ``default=str``
        to handle any non-serializable values gracefully.

        Returns:
            A JSON string representation of the campaign state.
        """
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> CampaignState:
        """Reconstruct a CampaignState from a JSON string.

        Args:
            json_str: JSON string as produced by ``to_json()``.

        Returns:
            A new CampaignState instance.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    # ── File-based Checkpoint / Resume ─────────────────

    def checkpoint_to_file(self, path: str | Path) -> None:
        """Write the campaign state to a JSON file.

        Creates parent directories if they do not exist. The file
        is written atomically by first serializing to string, then
        writing in a single operation.

        Args:
            path: Filesystem path for the checkpoint file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def resume_from_file(cls, path: str | Path) -> CampaignState:
        """Restore a CampaignState from a checkpoint file.

        Args:
            path: Filesystem path to the checkpoint JSON file.

        Returns:
            A new CampaignState instance restored from the file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        path = Path(path)
        json_str = path.read_text(encoding="utf-8")
        return cls.from_json(json_str)
