"""WebSocket connection manager: fan out live envelopes to dashboard clients."""

from __future__ import annotations

import asyncio
import logging

from fastapi import WebSocket

log = logging.getLogger("bump.api.ws")


class ConnectionManager:
    def __init__(self) -> None:
        self._clients: dict[WebSocket, str | None] = {}
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket, session_id: str | None) -> None:
        await ws.accept()
        async with self._lock:
            self._clients[ws] = session_id
        log.info("WS connected (session filter=%s, total=%d)", session_id, len(self._clients))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.pop(ws, None)

    async def broadcast(self, message: dict) -> None:
        """Send ``message`` to every client whose session filter matches (or is
        unset). Drops clients that error."""
        target_session = message.get("session_id")
        async with self._lock:
            items = list(self._clients.items())
        dead: list[WebSocket] = []
        for ws, filt in items:
            if filt is not None and target_session is not None and filt != target_session:
                continue
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.pop(ws, None)

    @property
    def count(self) -> int:
        return len(self._clients)
