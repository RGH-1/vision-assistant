"""Event hub: how events reach the phone and dashboard WebSockets (design §5.3, §8.2)."""

import asyncio
import contextlib
import logging
from typing import Any

from fastapi import WebSocket
from pydantic import TypeAdapter, ValidationError

from server.common.schemas import WebSocketMessage

logger = logging.getLogger(__name__)

MESSAGES = TypeAdapter(WebSocketMessage)
PHONE_MESSAGE_TYPES = {"alert", "status", "ping"}  # the dashboard gets everything


class EventHub:
    def __init__(self, loop: asyncio.AbstractEventLoop, send_timeout_s: float) -> None:
        self._loop = loop
        self._send_timeout_s = send_timeout_s
        self._events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.phones: set[WebSocket] = set()
        self.dashboards: set[WebSocket] = set()

    def publish(self, event: dict[str, Any]) -> None:
        """Safe to call from any thread: the event is handed over to the loop's own thread."""
        self._loop.call_soon_threadsafe(self._events.put_nowait, event)

    async def run(self) -> None:
        """Send events one at a time, so every client gets them in the same order."""
        while True:
            await self._broadcast(await self._events.get())

    async def _broadcast(self, event: dict[str, Any]) -> None:
        try:
            message = MESSAGES.validate_python(event)
        except ValidationError as error:
            # Only field names and error types, so the rejected content never reaches the logs.
            logger.error(
                "Dropped invalid event: %s", [(e["loc"], e["type"]) for e in error.errors()]
            )
            return

        clients = self.dashboards
        if message.type in PHONE_MESSAGE_TYPES:
            clients = self.dashboards | self.phones
        body = message.model_dump(mode="json")
        # Sent together, so a stalled client can't hold this event back from the others.
        await asyncio.gather(*(self._send(client, body) for client in clients))

    async def _send(self, client: WebSocket, body: dict[str, Any]) -> None:
        try:
            await asyncio.wait_for(client.send_json(body), self._send_timeout_s)
        except Exception as error:  # whatever one client does must not stop the broadcast
            logger.info("Dropping WebSocket client: %r", error)
            self.phones.discard(client)
            self.dashboards.discard(client)
            # Closed so a client that is still alive notices and reconnects (§8.2).
            with contextlib.suppress(Exception):
                await asyncio.wait_for(client.close(), self._send_timeout_s)
