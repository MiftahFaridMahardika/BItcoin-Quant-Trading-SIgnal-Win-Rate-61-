import asyncio
import json
import logging
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)
router = APIRouter()

# Active WebSocket connections
active_connections: Set[WebSocket] = set()


async def broadcast(message: dict):
    """Broadcast message to all connected clients."""
    if not active_connections:
        return
    data = json.dumps(message)
    dead = set()
    for ws in active_connections:
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.send_text(data)
        except Exception:
            dead.add(ws)
    active_connections.difference_update(dead)


@router.websocket("/feed")
async def websocket_feed(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    logger.info(f"Client connected. Total: {len(active_connections)}")

    try:
        # Send welcome message
        await websocket.send_json({"type": "connected", "message": "BTC Signal Feed connected"})

        while True:
            # Keep alive with heartbeat
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        logger.info("Client disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_connections.discard(websocket)
        logger.info(f"Client removed. Total: {len(active_connections)}")
