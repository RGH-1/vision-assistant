"""Event hub: how the ingest thread reaches the asyncio loop (design §5.3)."""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class EventHub:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, event: dict[str, Any]) -> None:
        """Safe to call from any thread: the event is handed over to the loop's own thread."""
        self._loop.call_soon_threadsafe(self._deliver, event)

    def _deliver(self, event: dict[str, Any]) -> None:
        # shortcut: only logs for now; VA-61 sends the event to the connected WebSockets.
        logger.debug("event: %s", event.get("type"))
