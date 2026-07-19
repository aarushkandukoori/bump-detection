"""Build the BeatCNN training set from MIT-BIH annotations.

For every annotated beat we produce the exact model input the inference service
uses (``make_model_input`` = normalised waveform + preceding-RR feature) and a
3-class label:

* **Normal** vs **Other** from the beat's annotation symbol (AAMI-style grouping).
* **Bradycardia** *overrides* the morphology label when the local heart rate is
  *sustained* below the threshold — computed from the annotation RR intervals,
  mirroring the deterministic rate rule used at inference time so train and
  inference agree on what "bradycardia" means.

Splits are by *record* (inter-patient, de Chazal DS1/DS2) so no patient leaks
between train and validation.
"""

from __future__ import annotations

import logging

import numpy as np
from bump_common.beats import make_model_input
from bump_common.constants import (
    BRADY_BPM_THRESHOLD,
    BRADY_MIN_BEATS,
    CLASS_TO_IDX,
    MITBIH_ARRHYTHMIA_SYMBOLS,
    MITBIH_NORMAL_SYMBOLS,
)

log = logging.getLogger("bump.training.dataset")

# de Chazal inter-patient split (paced records 102/104/107/217 excluded).
DS1_TRAIN = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
]
DS2_VAL = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
]

_BEAT_SYMBOLS = MITBIH_NORMAL_SYMBOLS | MITBIH_ARRHYTHMIA_SYMBOLS


def _morphology_label(symbol: str) -> str:
    return "Normal" if symbol in MITBIH_NORMAL_SYMBOLS else "Other"


def build_record(record: str, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (X[n,351] float32, y[n] int64) for one MIT-BIH record."""
    import wfdb

    rec = wfdb.rdrecord(record, pn_dir="mitdb")
    ann = wfdb.rdann(record, "atr", pn_dir="mitdb")
    sig = np.asarray(rec.p_signal[:, 0], dtype=np.float32)
    fs = int(rec.fs)

    # Keep only beat annotations, in order.
    beats = [
        (int(s), sym)
        for s, sym in zip(ann.sample, ann.symbol, strict=False)
        if sym in _BEAT_SYMBOLS
    ]
    if limit:
        beats = beats[:limit]

    xs: list[np.ndarray] = []
    ys: list[int] = []
    prev_sample: int | None = None
    recent_hr: list[float] = []

    for sample, symbol in beats:
        rr_ms = None
        if prev_sample is not None:
            rr_ms = (sample - prev_sample) / fs * 1000.0
        prev_sample = sample

        # Track a short trailing window of instantaneous HR for the sustain test.
        if rr_ms and rr_ms > 0:
            recent_hr.append(60000.0 / rr_ms)
            if len(recent_hr) > BRADY_MIN_BEATS:
                recent_hr.pop(0)

        # Label: sustained low rate -> Bradycardia, else morphology.
        sustained_brady = (
            len(recent_hr) >= BRADY_MIN_BEATS
            and all(hr < BRADY_BPM_THRESHOLD for hr in recent_hr)
        )
        label = "Bradycardia" if sustained_brady else _morphology_label(symbol)

        from bump_common.beats import extract_beat

        window = extract_beat(sig, sample)
        xs.append(make_model_input(window, rr_ms))
        ys.append(CLASS_TO_IDX[label])

    if not xs:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def build_dataset(
    records: list[str], limit_per_record: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate multiple records into one (X, y)."""
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for rec in records:
        try:
            x, y = build_record(rec, limit_per_record)
        except Exception as exc:
            log.warning("Skipping record %s: %s", rec, exc)
            continue
        if len(x):
            all_x.append(x)
            all_y.append(y)
            log.info("record %s: %d beats", rec, len(y))
    if not all_x:
        raise RuntimeError("No data built — is PhysioNet reachable?")
    return np.concatenate(all_x), np.concatenate(all_y)


def get_splits(smoke: bool = False) -> tuple[list[str], list[str]]:
    if smoke:
        return ["101"], ["100"]
    return DS1_TRAIN, DS2_VAL


# --- Synthetic rate augmentation --------------------------------------------
# MIT-BIH is not bradycardia-rich (records sit mostly 60-90 bpm), so the
# Bradycardia class would have thin support. We augment TRAIN with synthetic
# beats spanning a range of rates, labeled by the SAME sustained-rate rule.
# Validation stays pure MIT-BIH so reported metrics are not inflated.
# Crucially we include synthetic beats at NORMAL rates too, so the model
# cannot shortcut "synthetic morphology == bradycardia" — the only signal that
# separates the synthetic classes is the RR feature. This teaches the CNN to
# use rate context; the deterministic rate rule remains authoritative at
# inference (and overrides the live class label when BradycardiaMonitor fires).
_BRADY_RATES = [38.0, 42.0, 47.0, 52.0, 55.0, 57.0]
_NORMAL_RATES = [65.0, 68.0, 75.0, 78.0, 85.0, 92.0]


def build_synthetic_rate_beats(
    fs: int, seed: int, seconds_per_rate: float = 20.0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate rate-labeled synthetic beats (brady + normal) via neurokit2."""
    import neurokit2 as nk
    from bump_common.beats import extract_beat
    from bump_common.signal import pan_tompkins

    xs: list[np.ndarray] = []
    ys: list[int] = []
    for k, hr in enumerate(_BRADY_RATES + _NORMAL_RATES):
        sig = np.asarray(
            nk.ecg_simulate(
                duration=seconds_per_rate, sampling_rate=fs, heart_rate=hr,
                random_state=seed + k, method="ecgsyn",
            ),
            dtype=np.float32,
        )
        peaks = pan_tompkins(sig, fs)
        label = CLASS_TO_IDX["Bradycardia" if hr < BRADY_BPM_THRESHOLD else "Normal"]
        for i in range(1, len(peaks)):
            rr_ms = (peaks[i] - peaks[i - 1]) / fs * 1000.0
            window = extract_beat(sig, int(peaks[i]))
            xs.append(make_model_input(window, rr_ms))
            ys.append(label)
    if not xs:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)
