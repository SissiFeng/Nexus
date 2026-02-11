"""WebSocket routes for real-time campaign events."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from optimization_copilot.api.deps import get_event_bus

router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/{campaign_id}")
async def campaign_events(websocket: WebSocket, campaign_id: str) -> None:
    """WebSocket endpoint for real-time events from a specific campaign."""
    await websocket.accept()
    event_bus = get_event_bus()
    queue = event_bus.subscribe(campaign_id=campaign_id)

    try:
        while True:
            message = await queue.get()
            await websocket.send_json(message)
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        pass
    finally:
        event_bus.unsubscribe(queue)


@router.websocket("")
async def all_events(websocket: WebSocket) -> None:
    """WebSocket endpoint for events from all campaigns."""
    await websocket.accept()
    event_bus = get_event_bus()
    queue = event_bus.subscribe(campaign_id=None)

    try:
        while True:
            message = await queue.get()
            await websocket.send_json(message)
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        pass
    finally:
        event_bus.unsubscribe(queue)
