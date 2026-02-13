"""Campaign replay engine for reproducibility verification.

Provides ``ReplayResult`` (comparison summary) and ``CampaignReplayer``
(replays a sequence of events and checks deterministic reproducibility).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from optimization_copilot.reproducibility.logger import CampaignEvent


# ---------------------------------------------------------------------------
# ReplayResult
# ---------------------------------------------------------------------------


@dataclass
class ReplayResult:
    """Summary of a campaign replay comparison.

    Attributes
    ----------
    original_events : list[CampaignEvent]
        The original event sequence.
    replayed_events : list[CampaignEvent]
        The replayed event sequence.
    matches : int
        Number of events that matched between original and replay.
    mismatches : int
        Number of events that did not match.
    reproducibility_score : float
        Fraction of matching events (``matches / total``), or 1.0 if
        both sequences are empty.
    mismatch_details : list[dict[str, Any]]
        Per-mismatch detail records with index, field, and differing values.
    """

    original_events: list[CampaignEvent]
    replayed_events: list[CampaignEvent]
    matches: int
    mismatches: int
    reproducibility_score: float
    mismatch_details: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CampaignReplayer
# ---------------------------------------------------------------------------


class CampaignReplayer:
    """Replays campaign events and compares against the originals.

    Used to verify that a campaign can be deterministically reproduced
    given the same inputs and random seeds.
    """

    @staticmethod
    def compare_events(e1: CampaignEvent, e2: CampaignEvent) -> bool:
        """Compare two events by type, iteration, and data.

        Parameters
        ----------
        e1 : CampaignEvent
            First event.
        e2 : CampaignEvent
            Second event.

        Returns
        -------
        bool
            ``True`` if event_type, iteration, and data all match.
        """
        return (
            e1.event_type == e2.event_type
            and e1.iteration == e2.iteration
            and e1.data == e2.data
        )

    def replay(
        self,
        events: list[CampaignEvent],
        replay_fn: Callable[[CampaignEvent], CampaignEvent] | None = None,
    ) -> ReplayResult:
        """Replay a sequence of events and compare to originals.

        Parameters
        ----------
        events : list[CampaignEvent]
            The original event sequence to replay.
        replay_fn : callable or None
            A function that takes an original ``CampaignEvent`` and returns
            a replayed ``CampaignEvent``.  If ``None``, a default identity
            replay is used (copies the event), which models a perfectly
            deterministic campaign.

        Returns
        -------
        ReplayResult
            Summary of the comparison including per-mismatch details.
        """
        if replay_fn is None:
            replay_fn = self._default_replay_fn

        replayed: list[CampaignEvent] = []
        matches = 0
        mismatches = 0
        mismatch_details: list[dict[str, Any]] = []

        for idx, original in enumerate(events):
            replayed_event = replay_fn(original)
            replayed.append(replayed_event)

            if self.compare_events(original, replayed_event):
                matches += 1
            else:
                mismatches += 1
                details = self._collect_mismatch_details(idx, original, replayed_event)
                mismatch_details.extend(details)

        total = len(events)
        score = matches / total if total > 0 else 1.0

        return ReplayResult(
            original_events=list(events),
            replayed_events=replayed,
            matches=matches,
            mismatches=mismatches,
            reproducibility_score=score,
            mismatch_details=mismatch_details,
        )

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _default_replay_fn(event: CampaignEvent) -> CampaignEvent:
        """Default replay function: copy the event (perfect determinism)."""
        return CampaignEvent(
            event_id=event.event_id,
            event_type=event.event_type,
            timestamp=event.timestamp,
            iteration=event.iteration,
            data=dict(event.data),
            random_seed=event.random_seed,
        )

    @staticmethod
    def _collect_mismatch_details(
        index: int,
        original: CampaignEvent,
        replayed: CampaignEvent,
    ) -> list[dict[str, Any]]:
        """Collect per-field mismatch details between two events."""
        details: list[dict[str, Any]] = []

        if original.event_type != replayed.event_type:
            details.append({
                "index": index,
                "original_type": original.event_type.value,
                "replayed_type": replayed.event_type.value,
                "field": "event_type",
                "original_value": original.event_type.value,
                "replayed_value": replayed.event_type.value,
            })

        if original.iteration != replayed.iteration:
            details.append({
                "index": index,
                "original_type": original.event_type.value,
                "replayed_type": replayed.event_type.value,
                "field": "iteration",
                "original_value": original.iteration,
                "replayed_value": replayed.iteration,
            })

        if original.data != replayed.data:
            details.append({
                "index": index,
                "original_type": original.event_type.value,
                "replayed_type": replayed.event_type.value,
                "field": "data",
                "original_value": original.data,
                "replayed_value": replayed.data,
            })

        # If no specific field mismatch was identified but compare_events
        # returned False, record a generic mismatch entry.
        if not details:
            details.append({
                "index": index,
                "original_type": original.event_type.value,
                "replayed_type": replayed.event_type.value,
                "field": "unknown",
                "original_value": None,
                "replayed_value": None,
            })

        return details
