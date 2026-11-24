"""WebSocket routes for the phone and dashboard (design §8.2)."""

import logging
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from server.common.schemas import PingMessage
from server.events import MESSAGES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/ws")


async def serve_client(websocket: WebSocket, clients: set[WebSocket]) -> None:
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            try:
                message = MESSAGES.validate_json(await websocket.receive_text())
            except (ValidationError, KeyError):  # KeyError: a binary message has no "text"
                # Ignored, not disconnected, so one bad message can't silence the client's alerts.
                logger.warning("Ignored invalid WebSocket message")
                continue
            # hello needs no handling: the session and its language are already in the database.
            if isinstance(message, PingMessage):
                await websocket.send_json({"type": "pong"})
    except (WebSocketDisconnect, RuntimeError):  # RuntimeError: the hub already closed this socket
        pass
    finally:
        clients.discard(websocket)


@router.websocket("/phone")
async def phone(websocket: WebSocket, session_id: UUID) -> None:
    # shortcut: session_id is only checked as a UUID and every phone gets every alert, which is
    # fine with one phone; filter by session if a second phone is ever used.
    await serve_client(websocket, websocket.app.state.hub.phones)


@router.websocket("/dashboard")
async def dashboard(websocket: WebSocket) -> None:
    await serve_client(websocket, websocket.app.state.hub.dashboards)
