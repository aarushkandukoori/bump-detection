"""FastAPI app: live WebSocket stream + REST historical queries.

Startup wires the DB pool (with retry) and the Redis consumer that persists
readings/alerts and fans them out to WebSocket clients.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from bump_common.constants import API_METRICS_PORT, LATENCY_BUDGET_MS
from bump_common.metrics import maybe_start_metrics_server
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .db import Database
from .redis_consumer import RedisConsumer
from .ws import ConnectionManager

log = logging.getLogger("bump.api")

db = Database()
manager = ConnectionManager()
consumer = RedisConsumer(db, manager)


def _parse_ts(v: str | None) -> datetime | None:
    if not v:
        return None
    try:
        # Accept ISO-8601 or epoch-ms.
        if v.isdigit():
            return datetime.fromtimestamp(int(v) / 1000.0, tz=UTC)
        return datetime.fromisoformat(v)
    except ValueError as exc:
        raise HTTPException(400, f"bad timestamp: {v}") from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    maybe_start_metrics_server(API_METRICS_PORT)
    await db.connect()
    await consumer.start()
    log.info("API started (latency budget %d ms)", int(LATENCY_BUDGET_MS))
    yield
    await consumer.stop()
    await db.close()


app = FastAPI(title="BUMP API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dashboard dev origin; tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "ws_clients": manager.count, "budget_ms": LATENCY_BUDGET_MS}


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    session_id = ws.query_params.get("session_id")
    await manager.connect(ws, session_id)
    await ws.send_json({"type": "hello", "budget_ms": LATENCY_BUDGET_MS})
    try:
        while True:
            # We don't expect client messages; this keeps the socket open and
            # detects disconnects.
            await ws.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(ws)
    except Exception:
        await manager.disconnect(ws)


@app.get("/api/sessions")
async def list_sessions() -> list[dict]:
    return await db.list_sessions()


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    row = await db.get_session(session_id)
    if row is None:
        raise HTTPException(404, "session not found")
    return row


@app.get("/api/sessions/{session_id}/readings")
async def get_readings(
    session_id: str,
    from_: str | None = Query(None, alias="from"),
    to: str | None = None,
    limit: int = 5000,
) -> list[dict]:
    return await db.get_readings(
        session_id, _parse_ts(from_), _parse_ts(to), min(limit, 50000)
    )


@app.get("/api/sessions/{session_id}/alerts")
async def get_alerts(
    session_id: str,
    from_: str | None = Query(None, alias="from"),
    to: str | None = None,
) -> list[dict]:
    return await db.get_alerts(session_id, _parse_ts(from_), _parse_ts(to))
