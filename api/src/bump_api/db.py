"""Async TimescaleDB access (asyncpg). The API is the sole writer to the DB.

Connection is established lazily with retry/backoff so the service can start
before the database is ready (compose ordering is best-effort).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime

import asyncpg
from bump_common.constants import DATABASE_URL, session_meta_key
from bump_common.schemas import AlertEvent, HREvent

log = logging.getLogger("bump.api.db")


def _ts(ms: float) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC)


class Database:
    def __init__(self, dsn: str = DATABASE_URL) -> None:
        self.dsn = dsn
        self.pool: asyncpg.Pool | None = None
        self._known_sessions: set[str] = set()

    async def connect(self, retries: int = 30, delay: float = 2.0) -> None:
        last: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                self.pool = await asyncpg.create_pool(
                    self.dsn, min_size=1, max_size=8, command_timeout=30
                )
                log.info("Connected to TimescaleDB")
                return
            except Exception as exc:  # DB not ready yet
                last = exc
                log.warning("DB connect attempt %d/%d failed: %s", attempt, retries, exc)
                await asyncio.sleep(delay)
        raise RuntimeError(f"Could not connect to database: {last}")

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    # --- writes ------------------------------------------------------------
    async def register_session_if_new(self, redis_client, session_id: str) -> None:
        """Upsert a sessions row the first time we see a session_id, pulling
        record/source/fs from the Redis metadata hash ingestion wrote."""
        if session_id in self._known_sessions or self.pool is None:
            return
        meta: dict = {}
        try:
            raw = await redis_client.hgetall(session_meta_key(session_id))
            meta = {
                (k.decode() if isinstance(k, bytes) else k):
                (v.decode() if isinstance(v, bytes) else v)
                for k, v in (raw or {}).items()
            }
        except Exception:
            log.debug("No session meta for %s", session_id)
        started_at = None
        if meta.get("started_at"):
            try:
                started_at = _ts(float(meta["started_at"]) * 1000.0)
            except ValueError:
                started_at = None
        await self.pool.execute(
            """
            INSERT INTO sessions (session_id, record, source, fs, started_at, meta)
            VALUES ($1, $2, $3, $4, COALESCE($5, now()), $6::jsonb)
            ON CONFLICT (session_id) DO NOTHING
            """,
            session_id,
            meta.get("record"),
            meta.get("source", "unknown"),
            int(meta.get("fs", 360)),
            started_at,
            json.dumps(meta),
        )
        self._known_sessions.add(session_id)

    async def insert_hr_reading(self, evt: HREvent) -> None:
        if self.pool is None:
            return
        await self.pool.execute(
            """
            INSERT INTO hr_readings
              (time, session_id, beat_seq, instantaneous_hr, rr_ms,
               class_label, bradycardia, class_probs, latency_ms)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9)
            """,
            _ts(evt.t_infer_ms),
            evt.session_id,
            evt.beat_seq,
            None if evt.instantaneous_hr != evt.instantaneous_hr else evt.instantaneous_hr,
            None if evt.rr_ms != evt.rr_ms else evt.rr_ms,  # NaN guard
            evt.class_label,
            evt.bradycardia,
            json.dumps(evt.class_probs),
            evt.latency_ms,
        )

    async def insert_alert(self, evt: AlertEvent) -> None:
        if self.pool is None:
            return
        await self.pool.execute(
            """
            INSERT INTO alerts (time, session_id, type, hr, severity, message, latency_ms)
            VALUES ($1,$2,$3,$4,$5,$6,$7)
            """,
            _ts(evt.t_alert_ms),
            evt.session_id,
            evt.type,
            evt.hr,
            evt.severity,
            evt.message,
            evt.latency_ms,
        )

    # --- reads -------------------------------------------------------------
    async def list_sessions(self) -> list[dict]:
        rows = await self.pool.fetch(
            """
            SELECT s.session_id, s.record, s.source, s.fs, s.started_at, s.ended_at,
                   COUNT(h.*) AS reading_count
            FROM sessions s
            LEFT JOIN hr_readings h ON h.session_id = s.session_id
            GROUP BY s.session_id
            ORDER BY s.started_at DESC
            """
        )
        return [dict(r) for r in rows]

    async def get_session(self, session_id: str) -> dict | None:
        srow = await self.pool.fetchrow(
            "SELECT * FROM sessions WHERE session_id = $1", session_id
        )
        if srow is None:
            return None
        stats = await self.pool.fetchrow(
            """
            SELECT COUNT(*) AS readings,
                   MIN(instantaneous_hr) AS min_hr,
                   MAX(instantaneous_hr) AS max_hr,
                   AVG(instantaneous_hr) AS avg_hr,
                   COUNT(*) FILTER (WHERE bradycardia) AS brady_beats
            FROM hr_readings WHERE session_id = $1
            """,
            session_id,
        )
        alert_count = await self.pool.fetchval(
            "SELECT COUNT(*) FROM alerts WHERE session_id = $1", session_id
        )
        out = dict(srow)
        out["summary"] = {**dict(stats), "alert_count": alert_count}
        return out

    async def get_readings(
        self, session_id: str, from_ts: datetime | None,
        to_ts: datetime | None, limit: int = 5000,
    ) -> list[dict]:
        rows = await self.pool.fetch(
            """
            SELECT time, beat_seq, instantaneous_hr, rr_ms, class_label,
                   bradycardia, latency_ms
            FROM hr_readings
            WHERE session_id = $1
              AND ($2::timestamptz IS NULL OR time >= $2)
              AND ($3::timestamptz IS NULL OR time <= $3)
            ORDER BY time ASC
            LIMIT $4
            """,
            session_id, from_ts, to_ts, limit,
        )
        return [dict(r) for r in rows]

    async def get_alerts(
        self, session_id: str, from_ts: datetime | None, to_ts: datetime | None
    ) -> list[dict]:
        rows = await self.pool.fetch(
            """
            SELECT time, type, hr, severity, message, latency_ms
            FROM alerts
            WHERE session_id = $1
              AND ($2::timestamptz IS NULL OR time >= $2)
              AND ($3::timestamptz IS NULL OR time <= $3)
            ORDER BY time ASC
            """,
            session_id, from_ts, to_ts,
        )
        return [dict(r) for r in rows]
