"""Data provenance tracking for optimization campaigns.

Provides an append-only chain of provenance records that captures
the full lineage of data transformations, decisions, and observations
throughout an optimization campaign lifecycle.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator


# ── Valid actions ──────────────────────────────────────

_VALID_ACTIONS = frozenset({
    "import",
    "transform",
    "suggest",
    "observe",
    "decision",
    "export",
})


# ── Dataclasses ────────────────────────────────────────


@dataclass
class ProvenanceRecord:
    """A single provenance record in the chain.

    Parameters
    ----------
    record_id:
        Unique identifier for this record (typically a UUID).
    timestamp:
        Unix timestamp when the record was created.
    source:
        Identifier for the data source or system component.
    action:
        The type of action recorded. Must be one of:
        ``"import"``, ``"transform"``, ``"suggest"``, ``"observe"``,
        ``"decision"``, ``"export"``.
    agent:
        The agent or system that performed the action.
    parent_ids:
        Record IDs of parent records this record derives from.
    metadata:
        Additional key-value metadata for this record.

    Raises
    ------
    ValueError
        If ``action`` is not one of the valid action types.
    """

    record_id: str
    timestamp: float
    source: str
    action: str
    agent: str
    parent_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in _VALID_ACTIONS:
            raise ValueError(
                f"Invalid action '{self.action}'. "
                f"Must be one of: {', '.join(sorted(_VALID_ACTIONS))}"
            )


# ── ProvenanceChain ────────────────────────────────────


class ProvenanceChain:
    """An append-only chain of provenance records.

    Supports lineage queries, child lookups, and iteration.
    Records cannot be removed once appended.
    """

    def __init__(self) -> None:
        self._records: list[ProvenanceRecord] = []
        self._index: dict[str, int] = {}

    def append(self, record: ProvenanceRecord) -> None:
        """Append a record to the chain.

        Parameters
        ----------
        record:
            The provenance record to add.

        Raises
        ------
        ValueError
            If a record with the same ``record_id`` already exists.
        """
        if record.record_id in self._index:
            raise ValueError(
                f"Record with id '{record.record_id}' already exists in the chain."
            )
        self._index[record.record_id] = len(self._records)
        self._records.append(record)

    def get_record(self, record_id: str) -> ProvenanceRecord | None:
        """Retrieve a record by its ID.

        Parameters
        ----------
        record_id:
            The unique identifier of the record.

        Returns
        -------
        ProvenanceRecord | None
            The matching record, or ``None`` if not found.
        """
        idx = self._index.get(record_id)
        if idx is None:
            return None
        return self._records[idx]

    def get_children(self, record_id: str) -> list[ProvenanceRecord]:
        """Find all records that list the given record as a parent.

        Parameters
        ----------
        record_id:
            The parent record ID to search for.

        Returns
        -------
        list[ProvenanceRecord]
            Records whose ``parent_ids`` contain ``record_id``.
        """
        return [
            rec for rec in self._records
            if record_id in rec.parent_ids
        ]

    def get_lineage(self, record_id: str) -> list[ProvenanceRecord]:
        """Walk the parent chain recursively to build full lineage.

        Returns ancestors in breadth-first order, starting with the
        immediate parents.

        Parameters
        ----------
        record_id:
            The record whose lineage to trace.

        Returns
        -------
        list[ProvenanceRecord]
            All ancestor records, deduplicated and in traversal order.
        """
        lineage: list[ProvenanceRecord] = []
        visited: set[str] = set()
        queue: list[str] = []

        # Seed the queue with the record's own parent_ids.
        record = self.get_record(record_id)
        if record is None:
            return lineage

        queue.extend(record.parent_ids)

        while queue:
            pid = queue.pop(0)
            if pid in visited:
                continue
            visited.add(pid)
            parent = self.get_record(pid)
            if parent is not None:
                lineage.append(parent)
                queue.extend(parent.parent_ids)

        return lineage

    def __len__(self) -> int:
        """Return the number of records in the chain."""
        return len(self._records)

    def __iter__(self) -> Iterator[ProvenanceRecord]:
        """Iterate over all records in insertion order."""
        return iter(self._records)


# ── ProvenanceTracker ──────────────────────────────────


class ProvenanceTracker:
    """High-level provenance tracking with automatic ID and timestamp generation.

    Wraps a ``ProvenanceChain`` and provides convenience methods for
    recording actions and exporting the full chain.
    """

    def __init__(self) -> None:
        self._chain = ProvenanceChain()

    def record(
        self,
        action: str,
        source: str,
        agent: str = "system",
        parent_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProvenanceRecord:
        """Create and append a new provenance record.

        Parameters
        ----------
        action:
            The type of action. Must be one of the valid action types.
        source:
            Identifier for the data source or system component.
        agent:
            The agent or system that performed the action.
        parent_ids:
            Optional list of parent record IDs.
        metadata:
            Optional additional metadata.

        Returns
        -------
        ProvenanceRecord
            The newly created and appended record.
        """
        rec = ProvenanceRecord(
            record_id=str(uuid.uuid4()),
            timestamp=time.time(),
            source=source,
            action=action,
            agent=agent,
            parent_ids=parent_ids or [],
            metadata=metadata or {},
        )
        self._chain.append(rec)
        return rec

    def get_chain(self) -> ProvenanceChain:
        """Return the underlying provenance chain.

        Returns
        -------
        ProvenanceChain
            The full chain of recorded provenance events.
        """
        return self._chain

    def export_jsonld(self) -> str:
        """Serialize the full provenance chain as JSON-LD.

        Returns
        -------
        str
            A JSON-LD string representing the provenance chain with
            W3C PROV-O context.
        """
        records_data = []
        for rec in self._chain:
            records_data.append({
                "@type": "prov:Activity",
                "record_id": rec.record_id,
                "timestamp": rec.timestamp,
                "source": rec.source,
                "action": rec.action,
                "agent": rec.agent,
                "parent_ids": rec.parent_ids,
                "metadata": rec.metadata,
            })

        payload = {
            "@context": {
                "prov": "http://www.w3.org/ns/prov#",
                "record_id": "prov:identifier",
                "timestamp": "prov:atTime",
                "source": "prov:wasAssociatedWith",
                "action": "prov:type",
                "agent": "prov:wasAttributedTo",
                "parent_ids": "prov:wasDerivedFrom",
            },
            "@type": "prov:Bundle",
            "records": records_data,
        }
        return json.dumps(payload, indent=2, default=str)
