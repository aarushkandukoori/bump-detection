"""API wiring tests: REST endpoints, WS envelope broadcast, and startup.

These stub the database (TimescaleDB is not required to run them) so they verify
the FastAPI layer — routing, serialization, the WebSocket connection manager —
in isolation from storage.
"""

from __future__ import annotations

import httpx
import pytest


@pytest.fixture
async def client(monkeypatch):
    import bump_api.main as m

    # Stub DB + Redis consumer so lifespan starts without real backends.
    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(m.db, "connect", _noop)
    monkeypatch.setattr(m.db, "close", _noop)
    monkeypatch.setattr(m.consumer, "start", _noop)
    monkeypatch.setattr(m.consumer, "stop", _noop)
    monkeypatch.setattr(m, "maybe_start_metrics_server", lambda *_a, **_k: None)

    sessions = [
        {"session_id": "demo-100", "record": "100", "source": "mitbih", "fs": 360,
         "started_at": "2026-01-01T00:00:00Z", "ended_at": None, "reading_count": 3},
    ]
    monkeypatch.setattr(m.db, "list_sessions", lambda: _async(sessions))
    monkeypatch.setattr(
        m.db, "get_session",
        lambda sid: _async({**sessions[0], "summary": {"readings": 3, "alert_count": 1}})
        if sid == "demo-100" else _async(None),
    )
    monkeypatch.setattr(m.db, "get_readings", lambda *a, **k: _async([]))
    monkeypatch.setattr(m.db, "get_alerts", lambda *a, **k: _async([]))

    transport = httpx.ASGITransport(app=m.app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as c,
        m.app.router.lifespan_context(m.app),
    ):
        yield c


async def _async(value):
    return value


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["budget_ms"] == 250


async def test_list_sessions(client):
    r = await client.get("/api/sessions")
    assert r.status_code == 200
    assert r.json()[0]["session_id"] == "demo-100"


async def test_get_session_404(client):
    r = await client.get("/api/sessions/does-not-exist")
    assert r.status_code == 404


async def test_connection_manager_filters_by_session():
    from bump_api.ws import ConnectionManager

    class FakeWS:
        def __init__(self):
            self.sent = []

        async def accept(self):
            pass

        async def send_json(self, m):
            self.sent.append(m)

    mgr = ConnectionManager()
    all_clients, filtered = FakeWS(), FakeWS()
    await mgr.connect(all_clients, None)
    await mgr.connect(filtered, "demo-100")

    await mgr.broadcast({"type": "hr", "session_id": "demo-100", "hr": 44})
    await mgr.broadcast({"type": "hr", "session_id": "other", "hr": 80})

    # Unfiltered client sees both; filtered client sees only its session.
    assert len(all_clients.sent) == 2
    assert len(filtered.sent) == 1
    assert filtered.sent[0]["session_id"] == "demo-100"
