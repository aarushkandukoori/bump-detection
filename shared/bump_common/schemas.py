"""Pydantic message contracts for the Redis streams.

Redis stream field values are strings, so every model provides ``to_fields()``
(model -> ``dict[str, str]``) and ``from_fields()`` (``dict`` -> model). Complex
values are JSON-encoded. Keep these in lock-step across ingestion / inference /
api — they are the wire format.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field


def _dumps(v: object) -> str:
    return json.dumps(v, separators=(",", ":"))


class RawFrame(BaseModel):
    """A small frame of raw ECG samples emitted by the ingestion service."""

    session_id: str
    frame_seq: int
    # Wall-clock (unix ms) when the frame was emitted — the "sensor event" time
    # that the end-to-end latency budget is measured from.
    t_emit_ms: float
    fs: int
    # Absolute index of samples[0] within the session's sample stream.
    start_index: int
    samples: list[float]

    def to_fields(self) -> dict[str, str]:
        return {
            "session_id": self.session_id,
            "frame_seq": str(self.frame_seq),
            "t_emit_ms": repr(self.t_emit_ms),
            "fs": str(self.fs),
            "start_index": str(self.start_index),
            "samples": _dumps(self.samples),
        }

    @classmethod
    def from_fields(cls, f: dict[str, str]) -> RawFrame:
        return cls(
            session_id=f["session_id"],
            frame_seq=int(f["frame_seq"]),
            t_emit_ms=float(f["t_emit_ms"]),
            fs=int(f["fs"]),
            start_index=int(f["start_index"]),
            samples=json.loads(f["samples"]),
        )


class HREvent(BaseModel):
    """Per-beat inference output: HR, morphology class, and alert decision."""

    session_id: str
    beat_seq: int
    r_peak_index: int  # absolute sample index of the R-peak
    t_emit_ms: float  # originating frame emit time (sensor event)
    t_infer_ms: float  # when inference produced this event
    rr_ms: float
    instantaneous_hr: float
    class_label: str
    class_probs: dict[str, float] = Field(default_factory=dict)
    bradycardia: bool = False
    alert: bool = False
    alert_reason: str | None = None
    latency_ms: float = 0.0

    def to_fields(self) -> dict[str, str]:
        return {
            "session_id": self.session_id,
            "beat_seq": str(self.beat_seq),
            "r_peak_index": str(self.r_peak_index),
            "t_emit_ms": repr(self.t_emit_ms),
            "t_infer_ms": repr(self.t_infer_ms),
            "rr_ms": repr(self.rr_ms),
            "instantaneous_hr": repr(self.instantaneous_hr),
            "class_label": self.class_label,
            "class_probs": _dumps(self.class_probs),
            "bradycardia": "1" if self.bradycardia else "0",
            "alert": "1" if self.alert else "0",
            "alert_reason": self.alert_reason or "",
            "latency_ms": repr(self.latency_ms),
        }

    @classmethod
    def from_fields(cls, f: dict[str, str]) -> HREvent:
        return cls(
            session_id=f["session_id"],
            beat_seq=int(f["beat_seq"]),
            r_peak_index=int(f["r_peak_index"]),
            t_emit_ms=float(f["t_emit_ms"]),
            t_infer_ms=float(f["t_infer_ms"]),
            rr_ms=float(f["rr_ms"]),
            instantaneous_hr=float(f["instantaneous_hr"]),
            class_label=f["class_label"],
            class_probs=json.loads(f.get("class_probs", "{}")),
            bradycardia=f.get("bradycardia", "0") == "1",
            alert=f.get("alert", "0") == "1",
            alert_reason=f.get("alert_reason") or None,
            latency_ms=float(f.get("latency_ms", "0")),
        )


class AlertEvent(BaseModel):
    """A fired alert (bradycardia or arrhythmia). Only emitted on transitions."""

    session_id: str
    t_sensor_ms: float  # originating sensor-event time
    t_alert_ms: float  # when the alert was published
    type: str  # "bradycardia" | "arrhythmia"
    hr: float
    severity: str = "warning"  # "warning" | "critical"
    message: str = ""
    latency_ms: float = 0.0

    def to_fields(self) -> dict[str, str]:
        return {
            "session_id": self.session_id,
            "t_sensor_ms": repr(self.t_sensor_ms),
            "t_alert_ms": repr(self.t_alert_ms),
            "type": self.type,
            "hr": repr(self.hr),
            "severity": self.severity,
            "message": self.message,
            "latency_ms": repr(self.latency_ms),
        }

    @classmethod
    def from_fields(cls, f: dict[str, str]) -> AlertEvent:
        return cls(
            session_id=f["session_id"],
            t_sensor_ms=float(f["t_sensor_ms"]),
            t_alert_ms=float(f["t_alert_ms"]),
            type=f["type"],
            hr=float(f["hr"]),
            severity=f.get("severity", "warning"),
            message=f.get("message", ""),
            latency_ms=float(f.get("latency_ms", "0")),
        )
