"""End-to-end pipeline test: a known bradycardia episode must raise an alert
within the latency budget.

Latency semantics
-----------------
Frames are stamped with ``t_emit_ms = real wall-clock at feed time`` and fed as
fast as possible. The *sustain* logic runs on signal time (sample indices), so a
6 s bradycardia episode is recognised even though the frames are fed in
milliseconds. The alert's ``latency_ms`` is therefore the true processing
reaction time (now - the emit time of the frame that triggered it), which is
what the 250 ms budget governs. Bradycardia recall is the primary assertion —
missing the episode is the dangerous failure mode.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from bump_common.constants import FRAME_SAMPLES, LATENCY_BUDGET_MS
from bump_common.schemas import RawFrame


def _brady_scenario(fs: int) -> np.ndarray:
    """Normal 78 bpm -> sustained 44 bpm (>6 s) -> recovery."""
    import neurokit2 as nk

    def seg(hr, dur, seed):
        return np.asarray(
            nk.ecg_simulate(duration=dur, sampling_rate=fs, heart_rate=hr,
                            random_state=seed, method="ecgsyn"),
            dtype=np.float32,
        )

    return np.concatenate([seg(78, 8, 1), seg(44, 12, 2), seg(78, 4, 3)])


@pytest.mark.integration
def test_bradycardia_alert_fires_within_budget(fs, onnx_model_path):
    from bump_inference.classifier import BeatClassifier
    from bump_inference.pipeline import SessionPipeline

    signal = _brady_scenario(fs)
    pipe = SessionPipeline(
        session_id="itest", fs=fs, classifier=BeatClassifier(onnx_model_path)
    )

    alerts = []
    hr_events = []
    worst_alert_latency = 0.0
    for i in range(0, len(signal), FRAME_SAMPLES):
        chunk = signal[i : i + FRAME_SAMPLES]
        frame = RawFrame(
            session_id="itest", frame_seq=i // FRAME_SAMPLES,
            t_emit_ms=time.time() * 1000.0, fs=fs, start_index=i,
            samples=chunk.tolist(),
        )
        hrs, als = pipe.ingest_frame(frame)
        hr_events.extend(hrs)
        for a in als:
            if a.type == "bradycardia":
                worst_alert_latency = max(worst_alert_latency, a.latency_ms)
                alerts.append(a)

    # 1) The dangerous failure mode: a sustained bradycardia MUST be flagged.
    assert alerts, "no bradycardia alert fired for a sustained 44 bpm episode"

    # 2) Reaction latency must be within budget.
    assert worst_alert_latency < LATENCY_BUDGET_MS, (
        f"alert latency {worst_alert_latency:.1f}ms exceeds budget {LATENCY_BUDGET_MS}ms"
    )

    # Sanity: HR series spans the episode (tracked the slow segment).
    hrs = [h.instantaneous_hr for h in hr_events if not np.isnan(h.instantaneous_hr)]
    assert min(hrs) < 55, "detector never saw the bradycardic rate"


@pytest.mark.integration
@pytest.mark.redis
def test_pipeline_over_redis(fs, onnx_model_path):
    """Full wire path over a real Redis if one is reachable (else skip):
    ingestion frames -> stream -> inference consumer -> hr/alert streams."""
    import redis as redis_lib
    from bump_common.constants import (
        GROUP_INFERENCE,
        STREAM_ALERTS,
        STREAM_ECG_RAW,
        STREAM_HR_EVENTS,
    )
    from bump_common.redis_streams import (
        add_alert_event,
        add_raw_frame,
        decode_fields,
        ensure_group,
    )
    from bump_common.schemas import AlertEvent
    from bump_inference.classifier import BeatClassifier
    from bump_inference.pipeline import SessionPipeline

    url = __import__("os").getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        r = redis_lib.Redis.from_url(url)
        r.ping()
    except Exception as exc:
        pytest.skip(f"Redis unreachable: {exc}")

    # Isolated per-run stream + group so the test never races leftover data.
    sid = f"redis-itest-{int(time.time()*1000)}"
    raw_stream = f"{STREAM_ECG_RAW}:test:{sid}"
    signal = _brady_scenario(fs)

    # Create the consumer group at the (empty) tail BEFORE producing — this is
    # the real service ordering; otherwise '>' would deliver nothing.
    ensure_group(r, raw_stream, GROUP_INFERENCE)
    pipe = SessionPipeline(session_id=sid, fs=fs, classifier=BeatClassifier(onnx_model_path))
    fired: list[AlertEvent] = []

    # Interleave produce + consume (each frame is consumed right after it is
    # emitted, as in the real-time system) so latency reflects true processing
    # time, not queue-wait from batching all frames up front.
    for i in range(0, len(signal), FRAME_SAMPLES):
        chunk = signal[i : i + FRAME_SAMPLES]
        add_raw_frame(r, raw_stream, RawFrame(
            session_id=sid, frame_seq=i // FRAME_SAMPLES, t_emit_ms=time.time() * 1000.0,
            fs=fs, start_index=i, samples=chunk.tolist(),
        ))
        resp = r.xreadgroup(GROUP_INFERENCE, "itest", {raw_stream: ">"}, count=64, block=5)
        for _s, msgs in resp or []:
            for _mid, raw in msgs:
                frame = RawFrame.from_fields(decode_fields(raw))
                _hrs, als = pipe.ingest_frame(frame)
                for a in als:
                    add_alert_event(r, STREAM_ALERTS, a)
                    if a.type == "bradycardia":
                        fired.append(a)

    r.delete(raw_stream)  # cleanup
    assert fired, "no bradycardia alert produced over the Redis path"
    assert min(a.latency_ms for a in fired) < LATENCY_BUDGET_MS
