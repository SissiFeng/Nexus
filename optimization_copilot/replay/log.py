"""Decision log recording for deterministic replay.

Every optimization iteration produces a DecisionLogEntry capturing the full
pipeline state: snapshot hash, diagnostics, fingerprint, decision, candidates,
and ingested results.  A DecisionLog aggregates entries for an entire campaign
and supports round-trip JSON serialization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from optimization_copilot.core.hashing import (
    diagnostics_hash,
    decision_hash,
    snapshot_hash,
)


# ---------------------------------------------------------------------------
# DecisionLogEntry
# ---------------------------------------------------------------------------


@dataclass
class DecisionLogEntry:
    """Single iteration record in the decision log.

    Captures everything needed to verify deterministic replay:
    hashes of the snapshot, diagnostics, and decision; the full
    diagnostics vector; the problem fingerprint; the strategy
    decision; suggested candidates; ingested results; and metadata
    such as phase, backend, reason codes, and seed.
    """

    iteration: int
    timestamp: float
    snapshot_hash: str
    diagnostics_hash: str
    diagnostics: dict[str, float]
    fingerprint: dict[str, str]
    decision: dict[str, Any]
    decision_hash: str
    suggested_candidates: list[dict[str, Any]]
    ingested_results: list[dict[str, Any]]
    phase: str
    backend_name: str
    reason_codes: list[str]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "snapshot_hash": self.snapshot_hash,
            "diagnostics_hash": self.diagnostics_hash,
            "diagnostics": dict(self.diagnostics),
            "fingerprint": dict(self.fingerprint),
            "decision": dict(self.decision),
            "decision_hash": self.decision_hash,
            "suggested_candidates": [dict(c) for c in self.suggested_candidates],
            "ingested_results": [dict(r) for r in self.ingested_results],
            "phase": self.phase,
            "backend_name": self.backend_name,
            "reason_codes": list(self.reason_codes),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionLogEntry:
        """Deserialize from a plain dict."""
        return cls(
            iteration=data["iteration"],
            timestamp=data["timestamp"],
            snapshot_hash=data["snapshot_hash"],
            diagnostics_hash=data["diagnostics_hash"],
            diagnostics=data["diagnostics"],
            fingerprint=data["fingerprint"],
            decision=data["decision"],
            decision_hash=data["decision_hash"],
            suggested_candidates=data["suggested_candidates"],
            ingested_results=data["ingested_results"],
            phase=data["phase"],
            backend_name=data["backend_name"],
            reason_codes=data["reason_codes"],
            seed=data["seed"],
        )


# ---------------------------------------------------------------------------
# DecisionLog
# ---------------------------------------------------------------------------


@dataclass
class DecisionLog:
    """Complete decision log for an optimization campaign.

    Aggregates DecisionLogEntry instances for every iteration and
    provides lookup, serialization, and analysis utilities.
    """

    campaign_id: str
    spec: dict[str, Any]
    base_seed: int
    entries: list[DecisionLogEntry] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- mutation -----------------------------------------------------------

    def append(self, entry: DecisionLogEntry) -> None:
        """Add an entry to the log."""
        self.entries.append(entry)

    # -- queries ------------------------------------------------------------

    def get_entry(self, iteration: int) -> DecisionLogEntry | None:
        """Lookup a log entry by iteration number.

        Returns None if no entry exists for the given iteration.
        """
        for entry in self.entries:
            if entry.iteration == iteration:
                return entry
        return None

    @property
    def n_entries(self) -> int:
        """Number of recorded iterations."""
        return len(self.entries)

    @property
    def phase_transitions(self) -> list[tuple[int, str, str]]:
        """Detect phase transitions across the log.

        Returns a list of ``(iteration, from_phase, to_phase)`` tuples
        for every iteration where the phase changed from the previous
        entry.
        """
        transitions: list[tuple[int, str, str]] = []
        for i in range(1, len(self.entries)):
            prev_phase = self.entries[i - 1].phase
            curr_phase = self.entries[i].phase
            if curr_phase != prev_phase:
                transitions.append((self.entries[i].iteration, prev_phase, curr_phase))
        return transitions

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full log to a plain dict."""
        return {
            "campaign_id": self.campaign_id,
            "spec": self.spec,
            "base_seed": self.base_seed,
            "entries": [e.to_dict() for e in self.entries],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionLog:
        """Deserialize from a plain dict."""
        return cls(
            campaign_id=data["campaign_id"],
            spec=data["spec"],
            base_seed=data["base_seed"],
            entries=[DecisionLogEntry.from_dict(e) for e in data.get("entries", [])],
            metadata=data.get("metadata", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, default=str, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> DecisionLog:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(text))

    def save(self, path: str | Path) -> None:
        """Write the log to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> DecisionLog:
        """Load a log from a JSON file."""
        path = Path(path)
        return cls.from_json(path.read_text(encoding="utf-8"))
