"""bump_common: shared contracts and signal core for the BUMP pipeline.

Import ``redis_streams`` and ``metrics`` explicitly where needed so that pure
signal/schema use (e.g. unit tests) does not pull in redis/prometheus.
"""

from __future__ import annotations

from . import beats, constants, schemas, signal
from .constants import (
    BRADY_BPM_THRESHOLD,
    BRADY_MIN_BEATS,
    BRADY_SUSTAIN_SEC,
    CLASS_LABELS,
    CLASS_TO_IDX,
    DEFAULT_SAMPLE_RATE_HZ,
    IDX_TO_CLASS,
    LATENCY_BUDGET_MS,
    NUM_CLASSES,
    STREAM_ALERTS,
    STREAM_ECG_RAW,
    STREAM_HR_EVENTS,
)
from .schemas import AlertEvent, HREvent, RawFrame
from .signal import (
    BradycardiaMonitor,
    StreamingRPeakDetector,
    hr_from_peaks,
    instantaneous_hr,
    pan_tompkins,
    rr_intervals_ms,
)

__all__ = [
    "beats",
    "constants",
    "schemas",
    "signal",
    "AlertEvent",
    "HREvent",
    "RawFrame",
    "BradycardiaMonitor",
    "StreamingRPeakDetector",
    "pan_tompkins",
    "rr_intervals_ms",
    "instantaneous_hr",
    "hr_from_peaks",
    "CLASS_LABELS",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "NUM_CLASSES",
    "DEFAULT_SAMPLE_RATE_HZ",
    "BRADY_BPM_THRESHOLD",
    "BRADY_MIN_BEATS",
    "BRADY_SUSTAIN_SEC",
    "LATENCY_BUDGET_MS",
    "STREAM_ECG_RAW",
    "STREAM_HR_EVENTS",
    "STREAM_ALERTS",
]
