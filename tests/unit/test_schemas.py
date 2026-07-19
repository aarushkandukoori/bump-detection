"""Wire-format round-trips for the Redis stream message models."""

from __future__ import annotations

from bump_common.schemas import AlertEvent, HREvent, RawFrame


def test_raw_frame_roundtrip():
    f = RawFrame(
        session_id="s1", frame_seq=3, t_emit_ms=1234.5, fs=360,
        start_index=24, samples=[0.1, -0.2, 0.33],
    )
    back = RawFrame.from_fields(f.to_fields())
    assert back == f


def test_hr_event_roundtrip():
    e = HREvent(
        session_id="s1", beat_seq=7, r_peak_index=901, t_emit_ms=1000.0,
        t_infer_ms=1005.0, rr_ms=800.0, instantaneous_hr=75.0,
        class_label="Normal", class_probs={"Normal": 0.9, "Bradycardia": 0.05, "Other": 0.05},
        bradycardia=False, alert=False, alert_reason=None, latency_ms=5.0,
    )
    back = HREvent.from_fields(e.to_fields())
    assert back == e


def test_alert_event_roundtrip():
    a = AlertEvent(
        session_id="s1", t_sensor_ms=1000.0, t_alert_ms=1010.0, type="bradycardia",
        hr=45.0, severity="critical", message="Sustained bradycardia", latency_ms=10.0,
    )
    back = AlertEvent.from_fields(a.to_fields())
    assert back == a
