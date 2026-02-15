"""Event system for the optimization engine lifecycle.

Provides a lightweight publish-subscribe mechanism for engine events.
Components can register callbacks for specific event types and the
engine emits events as trials complete, batches finish, phases change,
and termination occurs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


# ── Enums ──────────────────────────────────────────────


class EngineEvent(str, Enum):
    TRIAL_COMPLETE = "trial_complete"
    TRIAL_FAILED = "trial_failed"
    BATCH_COMPLETE = "batch_complete"
    BATCH_FAILED = "batch_failed"
    PHASE_CHANGE = "phase_change"
    ITERATION_COMPLETE = "iteration_complete"
    CHECKPOINT_SAVED = "checkpoint_saved"
    TERMINATION = "termination"


# ── Event Payload ──────────────────────────────────────


@dataclass
class EventPayload:
    """Data payload accompanying an engine event.

    Attributes:
        event: The type of event that occurred.
        iteration: The iteration number when the event was emitted.
        data: Arbitrary event-specific data.
    """

    event: EngineEvent
    iteration: int
    data: dict[str, Any] = field(default_factory=dict)


# ── Type Alias ─────────────────────────────────────────


EventCallback = Callable[[EventPayload], None]


# ── Event Hook ─────────────────────────────────────────


class EventHook:
    """Publish-subscribe event dispatcher for engine lifecycle events.

    Handlers are registered for specific ``EngineEvent`` types.  When
    an ``EventPayload`` is emitted, all handlers registered for that
    event type are invoked synchronously in registration order.

    Example::

        hook = EventHook()
        hook.on(EngineEvent.TRIAL_COMPLETE, lambda p: print(p))
        hook.emit(EventPayload(
            event=EngineEvent.TRIAL_COMPLETE,
            iteration=1,
            data={"trial_id": "t-001"},
        ))
    """

    def __init__(self) -> None:
        self._handlers: dict[EngineEvent, list[EventCallback]] = {}

    def on(self, event: EngineEvent, callback: EventCallback) -> None:
        """Register a handler for a specific event type.

        Args:
            event: The event type to listen for.
            callback: A callable that accepts an ``EventPayload``.
        """
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(callback)

    def emit(self, payload: EventPayload) -> None:
        """Dispatch an event to all registered handlers.

        Handlers are called synchronously in the order they were
        registered. If a handler raises an exception, subsequent
        handlers for the same event are still invoked and the first
        exception is re-raised after all handlers have been called.

        Args:
            payload: The event payload to dispatch.
        """
        handlers = self._handlers.get(payload.event, [])
        first_error: Exception | None = None
        for handler in handlers:
            try:
                handler(payload)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def clear(self) -> None:
        """Remove all registered handlers for all event types."""
        self._handlers.clear()
