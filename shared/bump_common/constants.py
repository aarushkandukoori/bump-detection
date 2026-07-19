"""Shared constants and system-wide contracts for the BUMP pipeline.

This module is the single source of truth for values that MUST agree across
every service (ingestion, inference, api) and the tests. Do not fork these.
"""

from __future__ import annotations

import os

# --- Signal parameters -------------------------------------------------------
# MIT-BIH Arrhythmia Database native sampling rate.
DEFAULT_SAMPLE_RATE_HZ: int = int(os.getenv("SAMPLE_RATE_HZ", "360"))

# --- Classification labels ---------------------------------------------------
# System-level classes. NOTE: "Bradycardia" is fundamentally a *rate* condition
# (sustained HR < 60 bpm), not a beat morphology. The CNN emits all three
# classes so the ONNX artifact matches the spec, but at inference time the
# deterministic rate rule below is the *authoritative* bradycardia trigger.
CLASS_LABELS: list[str] = ["Normal", "Bradycardia", "Other"]
CLASS_TO_IDX: dict[str, int] = {label: i for i, label in enumerate(CLASS_LABELS)}
IDX_TO_CLASS: dict[int, str] = dict(enumerate(CLASS_LABELS))
NUM_CLASSES: int = len(CLASS_LABELS)

# AAMI-style grouping of MIT-BIH beat annotation symbols into a coarse
# "Normal vs Other (arrhythmia)" morphology target. Bradycardia is layered on
# top of this via rate, so it is intentionally absent here.
# Reference: ANSI/AAMI EC57 recommended beat groupings.
MITBIH_NORMAL_SYMBOLS: frozenset[str] = frozenset({"N", "L", "R", "e", "j"})
MITBIH_ARRHYTHMIA_SYMBOLS: frozenset[str] = frozenset(
    {"A", "a", "J", "S", "V", "E", "F", "!", "[", "]", "P", "f", "Q", "/"}
)

# --- Bradycardia decision rule (clinical) ------------------------------------
# Adult resting bradycardia is HR < 60 bpm. We require the low rate to be
# *sustained* to avoid single-beat artifacts triggering a (simulated) dose.
BRADY_BPM_THRESHOLD: float = float(os.getenv("BRADY_BPM_THRESHOLD", "60"))
# Number of consecutive qualifying beats OR seconds the low rate must persist.
BRADY_SUSTAIN_SEC: float = float(os.getenv("BRADY_SUSTAIN_SEC", "5.0"))
BRADY_MIN_BEATS: int = int(os.getenv("BRADY_MIN_BEATS", "4"))

# Physiologically plausible HR bounds; anything outside is treated as noise and
# not used to drive alerts (defensive against detector artifacts).
HR_MIN_PLAUSIBLE_BPM: float = 20.0
HR_MAX_PLAUSIBLE_BPM: float = 300.0

# --- Beat window for the CNN -------------------------------------------------
# Symmetric window (in samples) extracted around each R-peak at 360 Hz.
# 0.972 s total -> captures P-QRS-T for one beat.
BEAT_WINDOW_PRE_SAMPLES: int = 90
BEAT_WINDOW_POST_SAMPLES: int = 260
BEAT_WINDOW_LEN: int = BEAT_WINDOW_PRE_SAMPLES + BEAT_WINDOW_POST_SAMPLES  # 350

# --- ONNX model I/O contract -------------------------------------------------
# The trained model takes a flat vector: [normalised waveform (BEAT_WINDOW_LEN)]
# followed by [preceding-RR feature (1)]. Training and inference MUST agree on
# these names/shapes exactly.
MODEL_INPUT_DIM: int = BEAT_WINDOW_LEN + 1  # 351
ONNX_INPUT_NAME: str = "beat_input"
ONNX_OUTPUT_NAME: str = "logits"  # raw logits; wrapper applies softmax
DEFAULT_MODEL_PATH: str = os.getenv("MODEL_PATH", "/models/beat_cnn.onnx")

# --- Redis Stream names ------------------------------------------------------
STREAM_ECG_RAW: str = os.getenv("STREAM_ECG_RAW", "ecg:raw")
STREAM_HR_EVENTS: str = os.getenv("STREAM_HR_EVENTS", "hr:events")
STREAM_ALERTS: str = os.getenv("STREAM_ALERTS", "alerts:events")

# Consumer groups (must match the service that creates them).
GROUP_INFERENCE: str = "inference"
GROUP_API_EVENTS: str = "api-events"  # API reads hr:events + alerts:events
GROUP_API_WAVE: str = "api-wave"  # API reads ecg:raw for live waveform

# --- Latency budget ----------------------------------------------------------
# Sensor emit -> alert published. This is the server-measurable slice of the
# end-to-end budget and is what the integration test asserts against.
LATENCY_BUDGET_MS: float = float(os.getenv("LATENCY_BUDGET_MS", "250"))

# --- Ingestion framing -------------------------------------------------------
# Samples per Redis stream frame. 8 samples @360Hz ~= 22ms of signal, a good
# trade between Redis throughput and latency granularity.
FRAME_SAMPLES: int = int(os.getenv("FRAME_SAMPLES", "8"))

# --- Connection defaults -----------------------------------------------------
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "postgresql://bump:bump@localhost:5432/bump"
)

# --- Session metadata handoff ------------------------------------------------
# Ingestion HSETs {record, source, fs, started_at} here at startup; the API
# reads it when first registering a session, so DB writes stay solely in the API
# and ingestion needs no database dependency.
def session_meta_key(session_id: str) -> str:
    return f"session:{session_id}:meta"


# --- Metrics scrape ports (Prometheus) ---------------------------------------
INGESTION_METRICS_PORT: int = int(os.getenv("INGESTION_METRICS_PORT", "9101"))
INFERENCE_METRICS_PORT: int = int(os.getenv("INFERENCE_METRICS_PORT", "9102"))
API_METRICS_PORT: int = int(os.getenv("API_METRICS_PORT", "9103"))
