"""Campaign event logging for reproducibility tracking.

Provides ``EventType`` (classification enum), ``CampaignEvent`` (individual
logged event with UUID and timestamp), and ``CampaignLogger`` (append-only
event log with JSONL persistence).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# EventType
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    """Classification of campaign lifecycle events."""

    INIT = "init"
    SUGGEST = "suggest"
    OBSERVE = "observe"
    DECISION = "decision"
    SWITCH = "switch"
    DIAGNOSTIC = "diagnostic"
    ERROR = "error"
    COMPLETE = "complete"


# ---------------------------------------------------------------------------
# CampaignEvent
# ---------------------------------------------------------------------------


@dataclass
class CampaignEvent:
    """A single logged event in an optimization campaign.

    Attributes
    ----------
    event_id : str
        Unique identifier (UUID4) for this event.
    event_type : EventType
        The classification of this event.
    timestamp : float
        Unix timestamp when the event was recorded.
    iteration : int
        Campaign iteration at which the event occurred.
    data : dict[str, Any]
        Arbitrary payload associated with the event.
    random_seed : int or None
        Random seed active at the time of the event, if applicable.
    """

    event_id: str
    event_type: EventType
    timestamp: float
    iteration: int
    data: dict[str, Any] = field(default_factory=dict)
    random_seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        The ``event_type`` is stored as its string value for JSON
        compatibility.
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "iteration": self.iteration,
            "data": dict(self.data),
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CampaignEvent:
        """Deserialize from a plain dict."""
        return cls(
            event_id=d["event_id"],
            event_type=EventType(d["event_type"]),
            timestamp=d["timestamp"],
            iteration=d["iteration"],
            data=d.get("data", {}),
            random_seed=d.get("random_seed"),
        )


# ---------------------------------------------------------------------------
# CampaignLogger
# ---------------------------------------------------------------------------


class CampaignLogger:
    """Append-only event logger for an optimization campaign.

    Parameters
    ----------
    campaign_id : str
        Identifier for the campaign this logger tracks.
    """

    def __init__(self, campaign_id: str = "") -> None:
        self.campaign_id = campaign_id
        self._events: list[CampaignEvent] = []

    # -- logging ------------------------------------------------------------

    def log(
        self,
        event_type: EventType,
        data: dict[str, Any] | None = None,
        seed: int | None = None,
        iteration: int = 0,
    ) -> CampaignEvent:
        """Record a new campaign event.

        Automatically generates an ``event_id`` (UUID4) and ``timestamp``
        (``time.time()``), appends the event to the internal list, and
        returns it.

        Parameters
        ----------
        event_type : EventType
            Classification of the event.
        data : dict or None
            Arbitrary event payload.
        seed : int or None
            Random seed active at the time of the event.
        iteration : int
            Campaign iteration number.

        Returns
        -------
        CampaignEvent
            The newly created event.
        """
        event = CampaignEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=time.time(),
            iteration=iteration,
            data=data if data is not None else {},
            random_seed=seed,
        )
        self._events.append(event)
        return event

    # -- queries ------------------------------------------------------------

    def get_events(self) -> list[CampaignEvent]:
        """Return a copy of all logged events."""
        return list(self._events)

    def get_events_by_type(self, event_type: EventType) -> list[CampaignEvent]:
        """Return events filtered by type.

        Parameters
        ----------
        event_type : EventType
            The event type to filter on.

        Returns
        -------
        list[CampaignEvent]
        """
        return [e for e in self._events if e.event_type == event_type]

    # -- persistence --------------------------------------------------------

    def export_jsonl(self, path: str) -> None:
        """Write all events to a JSONL file (one JSON object per line).

        Parameters
        ----------
        path : str
            Destination file path.
        """
        with open(path, "w", encoding="utf-8") as fh:
            for event in self._events:
                fh.write(json.dumps(event.to_dict(), default=str) + "\n")

    @classmethod
    def from_jsonl(cls, path: str) -> CampaignLogger:
        """Reconstruct a ``CampaignLogger`` from a JSONL file.

        Parameters
        ----------
        path : str
            Source JSONL file path.

        Returns
        -------
        CampaignLogger
        """
        logger = cls()
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                event = CampaignEvent.from_dict(json.loads(line))
                logger._events.append(event)
        return logger

    # -- dunder protocols ---------------------------------------------------

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self):  # noqa: ANN204
        return iter(self._events)
