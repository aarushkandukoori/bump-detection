"""Async Redis consumer: hr:events + alerts:events -> persist + broadcast, and
ecg:raw -> coalesced waveform broadcast for the live dashboard.
"""

from __future__ import annotations

import asyncio
import logging

import redis.asyncio as aioredis
from bump_common.constants import (
    GROUP_API_EVENTS,
    GROUP_API_WAVE,
    REDIS_URL,
    STREAM_ALERTS,
    STREAM_ECG_RAW,
    STREAM_HR_EVENTS,
)
from bump_common.redis_streams import decode_fields
from bump_common.schemas import AlertEvent, HREvent, RawFrame

from .db import Database
from .ws import ConnectionManager

log = logging.getLogger("bump.api.consumer")

# Coalesce this many raw frames into one waveform WS message and downsample by 2.
WAVE_COALESCE_FRAMES = 6
WAVE_DOWNSAMPLE = 2


async def _ensure_group(r: aioredis.Redis, stream: str, group: str) -> None:
    try:
        await r.xgroup_create(stream, group, id="$", mkstream=True)
    except Exception as exc:
        if "BUSYGROUP" not in str(exc):
            raise


class RedisConsumer:
    def __init__(self, db: Database, manager: ConnectionManager, redis_url: str = REDIS_URL):
        self.db = db
        self.manager = manager
        self.redis = aioredis.from_url(redis_url)
        self._tasks: list[asyncio.Task] = []
        self._wave_buf: dict[str, list[float]] = {}
        self._wave_meta: dict[str, tuple[float, int, int]] = {}

    async def start(self) -> None:
        await _ensure_group(self.redis, STREAM_HR_EVENTS, GROUP_API_EVENTS)
        await _ensure_group(self.redis, STREAM_ALERTS, GROUP_API_EVENTS)
        await _ensure_group(self.redis, STREAM_ECG_RAW, GROUP_API_WAVE)
        self._tasks = [
            asyncio.create_task(self._consume_events()),
            asyncio.create_task(self._consume_waveform()),
        ]
        log.info("Redis consumer started")

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await self.redis.aclose()

    # ------------------------------------------------------------------
    async def _consume_events(self) -> None:
        while True:
            try:
                resp = await self.redis.xreadgroup(
                    GROUP_API_EVENTS, "api",
                    {STREAM_HR_EVENTS: ">", STREAM_ALERTS: ">"},
                    count=128, block=1000,
                )
                for stream, messages in resp or []:
                    sname = stream.decode() if isinstance(stream, bytes) else stream
                    for msg_id, raw in messages:
                        await self._handle_event(sname, decode_fields(raw))
                        await self.redis.xack(sname, GROUP_API_EVENTS, msg_id)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("event consumer error")
                await asyncio.sleep(1)

    async def _handle_event(self, stream: str, fields: dict[str, str]) -> None:
        session_id = fields.get("session_id", "")
        await self.db.register_session_if_new(self.redis, session_id)
        if stream == STREAM_HR_EVENTS:
            evt = HREvent.from_fields(fields)
            await self.db.insert_hr_reading(evt)
            await self.manager.broadcast({
                "type": "hr",
                "session_id": evt.session_id,
                "t": evt.t_infer_ms,
                "hr": None if evt.instantaneous_hr != evt.instantaneous_hr else evt.instantaneous_hr,
                "rr_ms": None if evt.rr_ms != evt.rr_ms else evt.rr_ms,
                "class_label": evt.class_label,
                "bradycardia": evt.bradycardia,
                "latency_ms": evt.latency_ms,
            })
        else:  # alerts
            evt = AlertEvent.from_fields(fields)
            await self.db.insert_alert(evt)
            await self.manager.broadcast({
                "type": "alert",
                "session_id": evt.session_id,
                "t": evt.t_alert_ms,
                "alert_type": evt.type,
                "hr": evt.hr,
                "severity": evt.severity,
                "message": evt.message,
                "latency_ms": evt.latency_ms,
            })

    # ------------------------------------------------------------------
    async def _consume_waveform(self) -> None:
        while True:
            try:
                resp = await self.redis.xreadgroup(
                    GROUP_API_WAVE, "api", {STREAM_ECG_RAW: ">"}, count=128, block=1000,
                )
                for stream, messages in resp or []:
                    sname = stream.decode() if isinstance(stream, bytes) else stream
                    for msg_id, raw in messages:
                        await self._handle_waveform(decode_fields(raw))
                        await self.redis.xack(sname, GROUP_API_WAVE, msg_id)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("waveform consumer error")
                await asyncio.sleep(1)

    async def _handle_waveform(self, fields: dict[str, str]) -> None:
        frame = RawFrame.from_fields(fields)
        sid = frame.session_id
        buf = self._wave_buf.setdefault(sid, [])
        buf.extend(frame.samples)
        self._wave_meta.setdefault(sid, (frame.t_emit_ms, frame.fs, 0))
        count = self._wave_meta[sid][2] + 1
        self._wave_meta[sid] = (frame.t_emit_ms, frame.fs, count)
        if count >= WAVE_COALESCE_FRAMES:
            downsampled = buf[::WAVE_DOWNSAMPLE]
            await self.manager.broadcast({
                "type": "waveform",
                "session_id": sid,
                "t": frame.t_emit_ms,
                "fs": frame.fs / WAVE_DOWNSAMPLE,
                "samples": [round(x, 4) for x in downsampled],
            })
            self._wave_buf[sid] = []
            self._wave_meta[sid] = (frame.t_emit_ms, frame.fs, 0)
