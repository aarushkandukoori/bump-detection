"""SessionPipeline: the pure, Redis-free core of the inference service.

Given a stream of :class:`RawFrame` s for one session it detects R-peaks online,
computes RR/HR, classifies beat morphology via the ONNX model, applies the
deterministic sustained-bradycardia rule, and returns HR/alert events. The
service loop (``service.py``) wires this to Redis; the integration test drives it
directly. Keeping it side-effect-free is what makes the latency test
deterministic.

Two clocks, deliberately separate
---------------------------------
* **Signal time** (``peak_index / fs``) drives the *sustain* logic: bradycardia
  means HR < 60 bpm for several *seconds of ECG*, independent of how fast frames
  are fed.
* **Wall-clock** (``time.time()``) drives the *latency* metric: for every event
  produced while ingesting a frame, ``latency_ms = now - frame.t_emit_ms``. In
  real-time streaming ``t_emit_ms`` is ~now, so this measures the pipeline's
  reaction latency (the 250 ms budget). The inherent QRS-detection delay
  (~edge-guard) is a separate, documented sensing delay, not this budget.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from bump_common.beats import extract_beat
from bump_common.constants import (
    BEAT_WINDOW_POST_SAMPLES,
    BEAT_WINDOW_PRE_SAMPLES,
    BRADY_BPM_THRESHOLD,
    BRADY_MIN_BEATS,
    BRADY_SUSTAIN_SEC,
    CLASS_LABELS,
    LATENCY_BUDGET_MS,
)
from bump_common.metrics import (
    ALERTS_TOTAL,
    BEATS_TOTAL,
    BUDGET_VIOLATIONS,
    SENSOR_TO_ALERT_LATENCY,
    SENSOR_TO_INFER_LATENCY,
)
from bump_common.schemas import AlertEvent, HREvent, RawFrame
from bump_common.signal import (
    BradycardiaMonitor,
    StreamingRPeakDetector,
    instantaneous_hr,
    is_plausible_hr,
)

from .classifier import BeatClassifier

# Number of consecutive CNN "Other" beats required before an arrhythmia alert.
ARRHYTHMIA_RUN = 3


def _now_ms() -> float:
    return time.time() * 1000.0


@dataclass
class _PendingBeat:
    peak_abs: int
    rr_ms: float | None
    hr: float
    bradycardia_at_detect: bool
    frame_t_emit_ms: float  # emit time of the frame that confirmed this peak


@dataclass
class SessionPipeline:
    """Stateful per-session detector + classifier + alert logic."""

    session_id: str
    fs: float
    classifier: BeatClassifier | None = None
    brady_threshold_bpm: float = BRADY_BPM_THRESHOLD
    brady_min_beats: int = BRADY_MIN_BEATS
    brady_sustain_sec: float = BRADY_SUSTAIN_SEC
    buffer_sec: float = 8.0

    # --- internal state ---
    _detector: StreamingRPeakDetector = field(init=False)
    _monitor: BradycardiaMonitor = field(init=False)
    _buf: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _buf_start: int = 0  # absolute index of _buf[0]
    _prev_peak_abs: int | None = None
    _beat_seq: int = 0
    _pending: deque[_PendingBeat] = field(default_factory=deque)
    _brady_active: bool = False
    _other_run: int = 0
    _arrhythmia_active: bool = False

    def __post_init__(self) -> None:
        self._detector = StreamingRPeakDetector(fs=self.fs)
        self._monitor = BradycardiaMonitor(
            threshold_bpm=self.brady_threshold_bpm,
            min_beats=self.brady_min_beats,
            sustain_sec=self.brady_sustain_sec,
        )

    # ------------------------------------------------------------------
    def ingest_frame(self, frame: RawFrame) -> tuple[list[HREvent], list[AlertEvent]]:
        """Process one frame; return (HR events, alert events) produced by it."""
        samples = np.asarray(frame.samples, dtype=np.float32)
        self._append(samples)

        hr_events: list[HREvent] = []
        alerts: list[AlertEvent] = []

        # 1) Detect newly-confirmed R-peaks and run the fast (rate-based) path:
        #    RR/HR, bradycardia rule, and bradycardia alerting. This path is NOT
        #    blocked by classification, so alerts stay within the budget.
        for peak_abs in self._detector.update(samples):
            rr_ms: float | None = None
            hr = float("nan")
            if self._prev_peak_abs is not None:
                rr_ms = (peak_abs - self._prev_peak_abs) / self.fs * 1000.0
                hr = instantaneous_hr(rr_ms)
            self._prev_peak_abs = peak_abs

            # Sustain logic on *signal* time.
            t_signal_ms = peak_abs / self.fs * 1000.0
            was_active = self._brady_active
            if not np.isnan(hr):
                self._brady_active = self._monitor.update(hr, t_signal_ms)

            # Rising edge -> emit a bradycardia alert (latency-critical path).
            if self._brady_active and not was_active:
                alerts.append(
                    self._make_alert(
                        frame, "bradycardia", hr, "critical",
                        f"Sustained bradycardia: HR ~{hr:.0f} bpm "
                        f"(< {self.brady_threshold_bpm:.0f} for "
                        f"{self.brady_sustain_sec:.0f}s)",
                    )
                )

            self._pending.append(
                _PendingBeat(peak_abs, rr_ms, hr, self._brady_active, frame.t_emit_ms)
            )

        # 2) Classify any pending beats whose full window is now buffered.
        hr_events.extend(self._drain_pending(frame, alerts))
        return hr_events, alerts

    # ------------------------------------------------------------------
    def _drain_pending(
        self, frame: RawFrame, alerts: list[AlertEvent]
    ) -> list[HREvent]:
        out: list[HREvent] = []
        buf_end_abs = self._buf_start + len(self._buf)
        while self._pending:
            beat = self._pending[0]
            if beat.peak_abs + BEAT_WINDOW_POST_SAMPLES >= buf_end_abs:
                break  # window not fully buffered yet
            self._pending.popleft()

            rel = beat.peak_abs - self._buf_start
            window = extract_beat(
                self._buf, rel, BEAT_WINDOW_PRE_SAMPLES, BEAT_WINDOW_POST_SAMPLES
            )
            label, probs = self._classify(window, beat.rr_ms, beat.hr)
            # Rate rule is authoritative for Bradycardia — keep the live label
            # consistent with the safety gate (CNN Brady class is morphology-
            # ambiguous; sinus brady looks like Normal).
            if beat.bradycardia_at_detect:
                label = "Bradycardia"

            # Conservative arrhythmia alerting from morphology (not latency
            # critical): fire on a short run of "Other" beats.
            if label == "Other":
                self._other_run += 1
            else:
                self._other_run = 0
                self._arrhythmia_active = False
            if self._other_run >= ARRHYTHMIA_RUN and not self._arrhythmia_active:
                self._arrhythmia_active = True
                alerts.append(
                    self._make_alert(
                        frame, "arrhythmia", beat.hr, "warning",
                        f"Arrhythmic morphology on {self._other_run} consecutive beats",
                    )
                )

            self._beat_seq += 1
            now = _now_ms()
            latency = now - beat.frame_t_emit_ms
            SENSOR_TO_INFER_LATENCY.observe(max(0.0, latency))
            BEATS_TOTAL.labels(self.session_id, label).inc()
            if latency > LATENCY_BUDGET_MS:
                BUDGET_VIOLATIONS.labels("infer").inc()

            out.append(
                HREvent(
                    session_id=self.session_id,
                    beat_seq=self._beat_seq,
                    r_peak_index=int(beat.peak_abs),
                    t_emit_ms=beat.frame_t_emit_ms,
                    t_infer_ms=now,
                    rr_ms=float(beat.rr_ms) if beat.rr_ms is not None else float("nan"),
                    instantaneous_hr=float(beat.hr),
                    class_label=label,
                    class_probs=probs,
                    bradycardia=beat.bradycardia_at_detect,
                    alert=beat.bradycardia_at_detect or self._arrhythmia_active,
                    alert_reason=("bradycardia" if beat.bradycardia_at_detect else None),
                    latency_ms=latency,
                )
            )
        return out

    # ------------------------------------------------------------------
    def _classify(
        self, window: np.ndarray, rr_ms: float | None, hr: float
    ) -> tuple[str, dict[str, float]]:
        if self.classifier is not None:
            res = self.classifier.classify_beat(window, rr_ms)
            return res.label, res.probs
        # Fallback when no model is loaded: rate-based label so the pipeline is
        # still useful (and testable) without the ONNX artifact.
        label = (
            "Bradycardia"
            if is_plausible_hr(hr) and hr < self.brady_threshold_bpm
            else "Normal"
        )
        return label, {c: (1.0 if c == label else 0.0) for c in CLASS_LABELS}

    def _make_alert(
        self, frame: RawFrame, atype: str, hr: float, severity: str, message: str
    ) -> AlertEvent:
        now = _now_ms()
        latency = now - frame.t_emit_ms
        SENSOR_TO_ALERT_LATENCY.observe(max(0.0, latency))
        ALERTS_TOTAL.labels(self.session_id, atype, severity).inc()
        if latency > LATENCY_BUDGET_MS:
            BUDGET_VIOLATIONS.labels("alert").inc()
        return AlertEvent(
            session_id=self.session_id,
            t_sensor_ms=frame.t_emit_ms,
            t_alert_ms=now,
            type=atype,
            hr=float(hr) if not np.isnan(hr) else 0.0,
            severity=severity,
            message=message,
            latency_ms=latency,
        )

    def _append(self, samples: np.ndarray) -> None:
        if samples.size:
            self._buf = np.concatenate([self._buf, samples])
        max_len = int(self.buffer_sec * self.fs)
        if len(self._buf) > max_len:
            drop = len(self._buf) - max_len
            self._buf = self._buf[drop:]
            self._buf_start += drop
