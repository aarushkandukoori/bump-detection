"""ECG signal processing: Pan-Tompkins R-peak detection, RR/HR math, and the
deterministic sustained-bradycardia rule.

This is the clinically-critical core of BUMP. Both the ``inference`` service and
the unit tests import the SAME functions here, so what CI validates against
MIT-BIH annotations is exactly what runs in production.

References
----------
Pan J, Tompkins WJ. "A Real-Time QRS Detection Algorithm." IEEE Trans Biomed
Eng. 1985;32(3):230-236.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from .constants import (
    BRADY_BPM_THRESHOLD,
    BRADY_MIN_BEATS,
    BRADY_SUSTAIN_SEC,
    HR_MAX_PLAUSIBLE_BPM,
    HR_MIN_PLAUSIBLE_BPM,
)

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def bandpass_filter(
    sig: np.ndarray, fs: float, low: float = 5.0, high: float = 15.0, order: int = 2
) -> np.ndarray:
    """Zero-phase Butterworth bandpass (approx. Pan-Tompkins 5-15 Hz passband).

    Zero-phase (``filtfilt``) keeps R-peak locations aligned with the raw signal,
    which matters when scoring against annotations placed on the R-peak.
    """
    sig = np.asarray(sig, dtype=float)
    nyq = 0.5 * fs
    high = min(high, nyq * 0.99)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    # filtfilt needs padlen < len(sig); guard very short segments.
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(sig) <= padlen:
        return sig - np.mean(sig)
    return filtfilt(b, a, sig)


def _five_point_derivative(sig: np.ndarray, fs: float) -> np.ndarray:
    """Classic Pan-Tompkins 5-point derivative: emphasises QRS slope."""
    d = np.zeros_like(sig)
    # y(n) = (1/8)(2x(n) + x(n-1) - x(n-3) - 2x(n-4)) * fs
    d[4:] = (
        2 * sig[4:] + sig[3:-1] - sig[1:-3] - 2 * sig[:-4]
    ) * (fs / 8.0)
    return d


# ---------------------------------------------------------------------------
# Batch Pan-Tompkins
# ---------------------------------------------------------------------------


def pan_tompkins(sig: np.ndarray, fs: float) -> np.ndarray:
    """Detect R-peaks in an ECG segment. Returns integer sample indices.

    This is the reference (batch) implementation. ``StreamingRPeakDetector``
    wraps it over a rolling buffer for the online path so both share one
    algorithm.
    """
    sig = np.asarray(sig, dtype=float)
    if len(sig) < int(0.4 * fs):
        return np.array([], dtype=int)

    filtered = bandpass_filter(sig, fs)
    deriv = _five_point_derivative(filtered, fs)
    squared = deriv ** 2

    # Moving-window integration (~150 ms).
    win = max(1, int(round(0.150 * fs)))
    mwi = np.convolve(squared, np.ones(win) / win, mode="same")

    integ_peaks = _adaptive_threshold(mwi, fs)
    r_peaks = _localize_r_peaks(filtered, integ_peaks, fs)

    # Drop peaks inside the filter-transient region at each boundary (~100 ms),
    # where band-pass edge effects produce spurious deflections.
    guard = int(0.1 * fs)
    if r_peaks.size:
        r_peaks = r_peaks[(r_peaks >= guard) & (r_peaks < len(sig) - guard)]
    return r_peaks


def _adaptive_threshold(mwi: np.ndarray, fs: float) -> np.ndarray:
    """Pan-Tompkins adaptive dual-threshold with RR-based search-back."""
    refractory = int(0.2 * fs)  # 200 ms physiological refractory period
    cand, _ = find_peaks(mwi, distance=refractory)
    if len(cand) == 0:
        return np.array([], dtype=int)

    # Learning phase: initialise running signal/noise level estimates.
    learn = min(len(mwi), int(2 * fs))
    spki = float(np.max(mwi[:learn])) * 0.25
    npki = float(np.mean(mwi[:learn])) * 0.5
    if spki <= npki:
        spki = npki * 2.0 + 1e-12

    signal_peaks: list[int] = []
    rr_recent: list[int] = []

    def thr1() -> float:
        return npki + 0.25 * (spki - npki)

    for pk in cand:
        val = float(mwi[pk])
        t1 = thr1()
        t2 = 0.5 * t1

        accepted = False
        if val >= t1:
            accepted = True
        elif signal_peaks:
            # Search-back: if the gap since the last accepted QRS is too long,
            # re-examine candidates against the lower threshold T2.
            rr_mean = np.mean(rr_recent) if rr_recent else refractory * 2
            if (pk - signal_peaks[-1]) > 1.66 * rr_mean and val >= t2:
                accepted = True

        if accepted:
            if signal_peaks:
                rr = pk - signal_peaks[-1]
                if refractory <= rr:  # ignore sub-refractory doubles
                    rr_recent.append(rr)
                    if len(rr_recent) > 8:
                        rr_recent.pop(0)
            signal_peaks.append(pk)
            spki = 0.125 * val + 0.875 * spki
        else:
            npki = 0.125 * val + 0.875 * npki

    return np.array(signal_peaks, dtype=int)


def _localize_r_peaks(
    filtered: np.ndarray, integ_peaks: np.ndarray, fs: float
) -> np.ndarray:
    """Snap each integration-window peak to the nearest local extremum of the
    band-pass signal (the true R-peak fiducial)."""
    if len(integ_peaks) == 0:
        return integ_peaks
    half = int(0.05 * fs)  # +-50 ms search window
    out: list[int] = []
    n = len(filtered)
    for p in integ_peaks:
        lo, hi = max(0, p - 2 * half), min(n, p + half)
        if hi <= lo:
            out.append(int(p))
            continue
        seg = filtered[lo:hi]
        # R-peak is the dominant deflection; use max |amplitude|.
        idx = lo + int(np.argmax(np.abs(seg - np.mean(seg))))
        out.append(idx)
    # De-duplicate after snapping (two integ peaks may map to one R-peak).
    out_sorted = sorted(set(out))
    deduped: list[int] = []
    min_gap = int(0.2 * fs)
    for idx in out_sorted:
        if not deduped or (idx - deduped[-1]) >= min_gap:
            deduped.append(idx)
    return np.array(deduped, dtype=int)


# ---------------------------------------------------------------------------
# Streaming wrapper
# ---------------------------------------------------------------------------


@dataclass
class StreamingRPeakDetector:
    """Online R-peak detector over a bounded rolling buffer.

    Feed chunks via :meth:`update`; it returns *absolute* sample indices of
    newly-confirmed R-peaks. A peak is only emitted once it sits at least
    ``edge_guard_sec`` from the buffer's right edge, guaranteeing enough
    right-context that its location is stable (won't shift as more data
    arrives). This keeps detection latency bounded (~edge_guard) while reusing
    the exact batch algorithm the tests validate.
    """

    fs: float
    buffer_sec: float = 8.0
    edge_guard_sec: float = 0.25

    _buf: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    _buf_start: int = 0  # absolute index of _buf[0]
    _last_emitted_abs: int = -1

    def update(self, samples: np.ndarray) -> list[int]:
        samples = np.asarray(samples, dtype=float)
        if samples.size:
            self._buf = np.concatenate([self._buf, samples])

        # Trim old data we can no longer need, keeping buffer bounded.
        max_len = int(self.buffer_sec * self.fs)
        if len(self._buf) > max_len:
            drop = len(self._buf) - max_len
            # Never drop past the last emitted peak's context.
            self._buf = self._buf[drop:]
            self._buf_start += drop

        peaks = pan_tompkins(self._buf, self.fs)
        edge_guard = int(self.edge_guard_sec * self.fs)
        confirm_before = len(self._buf) - edge_guard

        new_abs: list[int] = []
        for p in peaks:
            if p >= confirm_before:
                continue  # too close to the edge; wait for more data
            abs_idx = self._buf_start + int(p)
            if abs_idx > self._last_emitted_abs:
                new_abs.append(abs_idx)
        if new_abs:
            self._last_emitted_abs = new_abs[-1]
        return new_abs


# ---------------------------------------------------------------------------
# RR / HR math
# ---------------------------------------------------------------------------


def rr_intervals_ms(peaks: np.ndarray, fs: float) -> np.ndarray:
    """Convert R-peak sample indices to RR-intervals in milliseconds."""
    peaks = np.asarray(peaks)
    if len(peaks) < 2:
        return np.array([], dtype=float)
    return np.diff(peaks) / fs * 1000.0


def instantaneous_hr(rr_ms: float) -> float:
    """Beat-to-beat heart rate (bpm) from a single RR-interval in ms."""
    if rr_ms <= 0:
        return float("nan")
    return 60000.0 / rr_ms


def hr_from_peaks(peaks: np.ndarray, fs: float) -> np.ndarray:
    """Per-beat instantaneous HR series from R-peak indices."""
    rr = rr_intervals_ms(peaks, fs)
    if rr.size == 0:
        return np.array([], dtype=float)
    return 60000.0 / rr


def is_plausible_hr(hr: float) -> bool:
    return HR_MIN_PLAUSIBLE_BPM <= hr <= HR_MAX_PLAUSIBLE_BPM


# ---------------------------------------------------------------------------
# Sustained bradycardia rule
# ---------------------------------------------------------------------------


@dataclass
class BradycardiaMonitor:
    """Stateful, deterministic sustained-bradycardia detector.

    Bradycardia fires only when HR stays below ``threshold_bpm`` for at least
    ``min_beats`` consecutive qualifying beats AND ``sustain_sec`` of elapsed
    time. Requiring sustain avoids a single artifactual long RR (a missed beat
    doubles the apparent RR and halves HR) triggering a dose. False negatives
    are the dangerous mode, but a single dropped beat is not bradycardia.
    """

    threshold_bpm: float = BRADY_BPM_THRESHOLD
    min_beats: int = BRADY_MIN_BEATS
    sustain_sec: float = BRADY_SUSTAIN_SEC

    _low_run_beats: int = 0
    _low_run_start_ms: float | None = None
    _active: bool = False

    def update(self, hr_bpm: float, t_beat_ms: float) -> bool:
        """Feed one beat's HR and timestamp (ms). Returns True while an alert
        condition is currently active."""
        if not is_plausible_hr(hr_bpm):
            # Implausible reading: hold state, don't reset, don't count.
            return self._active

        if hr_bpm < self.threshold_bpm:
            if self._low_run_beats == 0:
                self._low_run_start_ms = t_beat_ms
            self._low_run_beats += 1
            elapsed = t_beat_ms - (self._low_run_start_ms or t_beat_ms)
            if (
                self._low_run_beats >= self.min_beats
                and elapsed >= self.sustain_sec * 1000.0
            ):
                self._active = True
        else:
            self._low_run_beats = 0
            self._low_run_start_ms = None
            self._active = False
        return self._active

    @property
    def active(self) -> bool:
        return self._active
