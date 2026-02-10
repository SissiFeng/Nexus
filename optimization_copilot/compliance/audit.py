"""Audit trail with hash-chained entries for regulatory compliance.

Provides ``AuditEntry`` (extends ``DecisionLogEntry`` with chain hashing),
``AuditLog`` (aggregates entries with chain integrity), ``ChainVerification``
(result of chain validation), and ``verify_chain`` (the validation function).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from optimization_copilot.core.hashing import _sha256, _stable_serialize
from optimization_copilot.replay.log import DecisionLogEntry


# ---------------------------------------------------------------------------
# Module-local helpers
# ---------------------------------------------------------------------------


def _compute_content_hash(entry_dict: dict[str, Any]) -> str:
    """Hash all fields of an entry dict except ``chain_hash``."""
    content = {k: v for k, v in entry_dict.items() if k != "chain_hash"}
    return _sha256(_stable_serialize(content))


def _compute_chain_hash(previous_chain_hash: str, content_hash: str) -> str:
    """Compute the next chain hash from the previous link and content hash."""
    if not previous_chain_hash:
        return _sha256(content_hash)
    return _sha256(previous_chain_hash + content_hash)


# ---------------------------------------------------------------------------
# ChainVerification
# ---------------------------------------------------------------------------


@dataclass
class ChainVerification:
    """Result of verifying an audit log's hash chain integrity.

    Attributes
    ----------
    valid : bool
        True when every entry's ``chain_hash`` matches the expected value.
    n_entries : int
        Total number of entries examined.
    n_broken_links : int
        Number of entries whose chain hash did not match.
    first_broken_link : int or None
        Iteration number of the first broken link, or None if the chain
        is intact.
    broken_links : list[int]
        Iteration numbers of all broken links.
    details : list[dict[str, Any]]
        Per-entry detail dicts (populated only for broken links).
    """

    valid: bool
    n_entries: int
    n_broken_links: int
    first_broken_link: int | None
    broken_links: list[int]
    details: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary of the verification result."""
        if self.valid:
            return (
                f"Chain verification PASSED: {self.n_entries} entries, "
                f"all hashes valid."
            )
        return (
            f"Chain verification FAILED: {self.n_broken_links}/{self.n_entries} "
            f"broken links. First broken at iteration {self.first_broken_link}."
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "valid": self.valid,
            "n_entries": self.n_entries,
            "n_broken_links": self.n_broken_links,
            "first_broken_link": self.first_broken_link,
            "broken_links": list(self.broken_links),
            "details": list(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChainVerification:
        """Deserialize from a plain dict."""
        return cls(
            valid=data["valid"],
            n_entries=data["n_entries"],
            n_broken_links=data["n_broken_links"],
            first_broken_link=data["first_broken_link"],
            broken_links=data["broken_links"],
            details=data.get("details", []),
        )


# ---------------------------------------------------------------------------
# AuditEntry
# ---------------------------------------------------------------------------


@dataclass
class AuditEntry:
    """A single hash-chained audit entry extending ``DecisionLogEntry``.

    Contains all fields from ``DecisionLogEntry`` plus ``chain_hash``,
    ``signer_id``, and ``entry_version`` for regulatory traceability.
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
    chain_hash: str = ""
    signer_id: str = ""
    entry_version: str = "1.0.0"

    @classmethod
    def from_log_entry(
        cls,
        entry: DecisionLogEntry,
        chain_hash: str,
        signer_id: str = "",
        entry_version: str = "1.0.0",
    ) -> AuditEntry:
        """Create an ``AuditEntry`` from a ``DecisionLogEntry``.

        Copies all fields from *entry* and attaches the chain hash,
        signer identifier, and entry version.
        """
        return cls(
            iteration=entry.iteration,
            timestamp=entry.timestamp,
            snapshot_hash=entry.snapshot_hash,
            diagnostics_hash=entry.diagnostics_hash,
            diagnostics=dict(entry.diagnostics),
            fingerprint=dict(entry.fingerprint),
            decision=dict(entry.decision),
            decision_hash=entry.decision_hash,
            suggested_candidates=[dict(c) for c in entry.suggested_candidates],
            ingested_results=[dict(r) for r in entry.ingested_results],
            phase=entry.phase,
            backend_name=entry.backend_name,
            reason_codes=list(entry.reason_codes),
            seed=entry.seed,
            chain_hash=chain_hash,
            signer_id=signer_id,
            entry_version=entry_version,
        )

    def content_hash(self) -> str:
        """Compute the content hash (all fields except ``chain_hash``)."""
        return _compute_content_hash(self.to_dict())

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
            "chain_hash": self.chain_hash,
            "signer_id": self.signer_id,
            "entry_version": self.entry_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditEntry:
        """Deserialize from a plain dict."""
        return cls(**data)


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------


@dataclass
class AuditLog:
    """Hash-chained audit log for a complete optimization campaign.

    Aggregates ``AuditEntry`` instances and provides chain integrity
    verification, JSON serialization, and file persistence.

    Attributes
    ----------
    campaign_id : str
        Unique identifier for the campaign.
    spec : dict[str, Any]
        Campaign specification (parameter specs, objectives, etc.).
    base_seed : int
        Base random seed for the campaign.
    entries : list[AuditEntry]
        Ordered sequence of audit entries.
    metadata : dict[str, Any]
        Arbitrary metadata about the audit.
    signer_id : str
        Identifier of the signing entity.
    """

    campaign_id: str
    spec: dict[str, Any]
    base_seed: int
    entries: list[AuditEntry] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    signer_id: str = ""

    # -- mutation -----------------------------------------------------------

    def append(self, entry: AuditEntry) -> None:
        """Add an audit entry to the log."""
        self.entries.append(entry)

    # -- queries ------------------------------------------------------------

    def get_entry(self, iteration: int) -> AuditEntry | None:
        """Look up an entry by iteration number.

        Returns ``None`` if no entry exists for the given iteration.
        """
        for entry in self.entries:
            if entry.iteration == iteration:
                return entry
        return None

    @property
    def n_entries(self) -> int:
        """Number of recorded entries."""
        return len(self.entries)

    @property
    def chain_intact(self) -> bool:
        """Return ``True`` if the hash chain is fully intact."""
        return verify_chain(self).valid

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full audit log to a plain dict."""
        return {
            "campaign_id": self.campaign_id,
            "spec": self.spec,
            "base_seed": self.base_seed,
            "entries": [e.to_dict() for e in self.entries],
            "metadata": dict(self.metadata),
            "signer_id": self.signer_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditLog:
        """Deserialize from a plain dict."""
        return cls(
            campaign_id=data["campaign_id"],
            spec=data["spec"],
            base_seed=data["base_seed"],
            entries=[AuditEntry.from_dict(e) for e in data.get("entries", [])],
            metadata=data.get("metadata", {}),
            signer_id=data.get("signer_id", ""),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, text: str) -> AuditLog:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(text))

    def save(self, path: str | Path) -> None:
        """Write the audit log to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> AuditLog:
        """Load an audit log from a JSON file."""
        path = Path(path)
        return cls.from_json(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Chain verification
# ---------------------------------------------------------------------------


def verify_chain(audit_log: AuditLog) -> ChainVerification:
    """Verify the hash chain integrity of an ``AuditLog``.

    Walks through every entry and recomputes the expected ``chain_hash``
    from the previous entry's chain hash and the current entry's content
    hash.  Returns a ``ChainVerification`` summarising the result.

    Parameters
    ----------
    audit_log : AuditLog
        The audit log to verify.

    Returns
    -------
    ChainVerification
    """
    entries = audit_log.entries
    n = len(entries)

    if n == 0:
        return ChainVerification(
            valid=True,
            n_entries=0,
            n_broken_links=0,
            first_broken_link=None,
            broken_links=[],
            details=[],
        )

    broken_links: list[int] = []
    details: list[dict[str, Any]] = []
    first_broken: int | None = None

    for i, entry in enumerate(entries):
        prev_chain = entries[i - 1].chain_hash if i > 0 else ""
        c_hash = entry.content_hash()
        expected = _compute_chain_hash(prev_chain, c_hash)

        if entry.chain_hash != expected:
            broken_links.append(entry.iteration)
            if first_broken is None:
                first_broken = entry.iteration
            details.append({
                "index": i,
                "iteration": entry.iteration,
                "expected_chain_hash": expected,
                "actual_chain_hash": entry.chain_hash,
                "content_hash": c_hash,
            })

    return ChainVerification(
        valid=len(broken_links) == 0,
        n_entries=n,
        n_broken_links=len(broken_links),
        first_broken_link=first_broken,
        broken_links=broken_links,
        details=details,
    )
