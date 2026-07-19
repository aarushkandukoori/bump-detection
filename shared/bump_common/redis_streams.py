"""Thin, client-agnostic helpers for the Redis stream wire format.

Encoding/decoding lives here so services share one implementation. The sync
convenience helpers take a ``redis.Redis`` client; the API service uses its own
async client but reuses the same ``*_fields`` encoders on the schema models.
"""

from __future__ import annotations

from typing import Any

from .schemas import AlertEvent, HREvent, RawFrame


def add_raw_frame(client: Any, stream: str, frame: RawFrame, maxlen: int = 20000) -> str:
    return client.xadd(stream, frame.to_fields(), maxlen=maxlen, approximate=True)


def add_hr_event(client: Any, stream: str, evt: HREvent, maxlen: int = 50000) -> str:
    return client.xadd(stream, evt.to_fields(), maxlen=maxlen, approximate=True)


def add_alert_event(client: Any, stream: str, evt: AlertEvent, maxlen: int = 10000) -> str:
    return client.xadd(stream, evt.to_fields(), maxlen=maxlen, approximate=True)


def ensure_group(client: Any, stream: str, group: str) -> None:
    """Create a consumer group at the stream tail, ignoring 'already exists'."""
    try:
        client.xgroup_create(stream, group, id="$", mkstream=True)
    except Exception as exc:  # redis.exceptions.ResponseError: BUSYGROUP
        if "BUSYGROUP" not in str(exc):
            raise


def decode_fields(raw: dict[bytes | str, bytes | str]) -> dict[str, str]:
    """Normalise a raw redis field mapping to ``dict[str, str]``."""
    out: dict[str, str] = {}
    for k, v in raw.items():
        ks = k.decode() if isinstance(k, bytes) else k
        vs = v.decode() if isinstance(v, bytes) else v
        out[ks] = vs
    return out
