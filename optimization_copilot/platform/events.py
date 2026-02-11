"""Async event bus bridging synchronous engine events to async subscribers."""

from __future__ import annotations

import asyncio
import json
import logging
from time import time
from typing import Any

logger = logging.getLogger(__name__)


class AsyncEventBus:
    """Thread-safe event bus that bridges sync engine events to async consumers.

    Usage:
        bus = AsyncEventBus()

        # Async consumer (WebSocket handler)
        queue = bus.subscribe(campaign_id="abc")
        event = await queue.get()

        # Sync producer (engine thread) — thread-safe
        bus.publish("abc", "iteration_complete", {"iteration": 5})
    """

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop
        # campaign_id -> list of subscriber queues
        self._subscribers: dict[str | None, list[asyncio.Queue[dict[str, Any]]]] = {}

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        return self._loop

    @loop.setter
    def loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(
        self, campaign_id: str | None = None, maxsize: int = 100
    ) -> asyncio.Queue[dict[str, Any]]:
        """Subscribe to events. campaign_id=None subscribes to all campaigns."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        if campaign_id not in self._subscribers:
            self._subscribers[campaign_id] = []
        self._subscribers[campaign_id].append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove a subscriber queue."""
        for campaign_id, queues in self._subscribers.items():
            if queue in queues:
                queues.remove(queue)
                if not queues:
                    del self._subscribers[campaign_id]
                return

    def publish(self, campaign_id: str, event: str, data: dict[str, Any]) -> None:
        """Publish an event (thread-safe — can be called from engine thread)."""
        message = {
            "campaign_id": campaign_id,
            "event": event,
            "data": data,
            "timestamp": time(),
        }

        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._dispatch, campaign_id, message)
        else:
            # Fallback for when no event loop is set (e.g., testing)
            self._dispatch(campaign_id, message)

    def _dispatch(self, campaign_id: str, message: dict[str, Any]) -> None:
        """Dispatch message to matching subscribers."""
        # Campaign-specific subscribers
        for queue in self._subscribers.get(campaign_id, []):
            self._put_nowait(queue, message)

        # Global subscribers (campaign_id=None)
        for queue in self._subscribers.get(None, []):
            self._put_nowait(queue, message)

    @staticmethod
    def _put_nowait(
        queue: asyncio.Queue[dict[str, Any]], message: dict[str, Any]
    ) -> None:
        """Put message into queue, dropping oldest if full."""
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()  # Drop oldest
            except asyncio.QueueEmpty:
                pass
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                logger.warning("Event queue still full after drop, skipping event")

    def subscriber_count(self, campaign_id: str | None = None) -> int:
        """Count subscribers for a campaign (or global if None)."""
        return len(self._subscribers.get(campaign_id, []))

    def clear(self) -> None:
        """Remove all subscribers."""
        self._subscribers.clear()

    @staticmethod
    def format_sse(message: dict[str, Any]) -> str:
        """Format message as Server-Sent Event string."""
        return f"data: {json.dumps(message)}\n\n"
