"""WebSocket routes for real-time campaign events."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from optimization_copilot.api.deps import get_event_bus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])


async def _pump_events(
    websocket: WebSocket,
    queue: "asyncio.Queue[dict]",
    label: str,
) -> None:
    """Read events from *queue* and forward them to the WebSocket client.

    Also listens for incoming client messages so that we can detect
    disconnects promptly (and handle malformed JSON gracefully).
    """
    send_task: asyncio.Task | None = None
    recv_task: asyncio.Task | None = None

    try:
        while True:
            # Wait for either a new event to send or a client message/disconnect.
            if send_task is None or send_task.done():
                send_task = asyncio.ensure_future(queue.get())
            if recv_task is None or recv_task.done():
                recv_task = asyncio.ensure_future(websocket.receive_text())

            done, _ = await asyncio.wait(
                {send_task, recv_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                if task is send_task:
                    message = task.result()
                    await websocket.send_json(message)
                elif task is recv_task:
                    raw = task.result()
                    # Best-effort: parse incoming message but don't require it.
                    try:
                        json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(
                            "WS %s: received non-JSON message, ignoring",
                            label,
                        )
    except WebSocketDisconnect:
        logger.debug("WS %s: client disconnected", label)
    except asyncio.CancelledError:
        logger.debug("WS %s: task cancelled", label)
    except Exception:
        # Catch-all for unexpected connection errors (e.g. broken pipe,
        # connection reset) so the server never crashes on a single socket.
        logger.exception("WS %s: unexpected error", label)
    finally:
        # Cancel any in-flight tasks to avoid dangling coroutines.
        for t in (send_task, recv_task):
            if t is not None and not t.done():
                t.cancel()


@router.websocket("/{campaign_id}")
async def campaign_events(websocket: WebSocket, campaign_id: str) -> None:
    """WebSocket endpoint for real-time events from a specific campaign."""
    try:
        await websocket.accept()
    except Exception:
        logger.exception("WS campaign/%s: failed to accept connection", campaign_id)
        return

    event_bus = get_event_bus()
    queue = event_bus.subscribe(campaign_id=campaign_id)
    try:
        await _pump_events(websocket, queue, label=f"campaign/{campaign_id}")
    finally:
        event_bus.unsubscribe(queue)
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except Exception:
                pass


@router.websocket("")
async def all_events(websocket: WebSocket) -> None:
    """WebSocket endpoint for events from all campaigns."""
    try:
        await websocket.accept()
    except Exception:
        logger.exception("WS all: failed to accept connection")
        return

    event_bus = get_event_bus()
    queue = event_bus.subscribe(campaign_id=None)
    try:
        await _pump_events(websocket, queue, label="all")
    finally:
        event_bus.unsubscribe(queue)
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except Exception:
                pass
