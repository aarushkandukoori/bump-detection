"""ECG signal sources for replay: real MIT-BIH records and a synthetic generator.

Every source returns a 1-D float32 numpy array of samples plus the sample rate.
The replayer paces emission to real time; sources just produce the waveform.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("bump.ingestion")


def mitbih_source(record: str, channel: int = 0) -> tuple[np.ndarray, int]:
    """Load a MIT-BIH Arrhythmia Database record from PhysioNet via wfdb.

    Downloads (and caches) the record on first use. Returns channel-0 signal in
    millivolts and its native sample rate (360 Hz for mitdb).
    """
    import wfdb  # imported lazily so the module imports without the dep present

    log.info("Loading MIT-BIH record %s from PhysioNet (mitdb)…", record)
    rec = wfdb.rdrecord(record, pn_dir="mitdb")
    sig = np.asarray(rec.p_signal[:, channel], dtype=np.float32)
    fs = int(rec.fs)
    log.info("Loaded record %s: %d samples @ %d Hz", record, len(sig), fs)
    return sig, fs


def synthetic_source(
    hr_bpm: float, duration_sec: float, fs: int, seed: int = 42
) -> tuple[np.ndarray, int]:
    """Generate a clean synthetic ECG at a fixed heart rate (neurokit2)."""
    import neurokit2 as nk

    sig = nk.ecg_simulate(
        duration=float(duration_sec),
        sampling_rate=fs,
        heart_rate=float(hr_bpm),
        random_state=seed,
        method="ecgsyn",
    )
    return np.asarray(sig, dtype=np.float32), fs


def build_bradycardia_demo(
    fs: int, duration_sec: float = 60.0
) -> tuple[np.ndarray, int, dict[str, float]]:
    """Synthesise a demo ECG with a *sustained* bradycardia episode.

    MIT-BIH is not bradycardia-rich, so this gives the pipeline a known,
    reproducible low-rate event to detect and alert on. Layout (repeated to fill
    ``duration_sec``): ~12 s normal (75 bpm) -> ~14 s bradycardia (45 bpm) ->
    ~8 s recovery (78 bpm). Returns the signal plus the (approximate) seconds at
    which the bradycardia segment starts/ends in the first cycle.
    """
    import neurokit2 as nk

    def seg(hr: float, dur: float, seed: int) -> np.ndarray:
        s = nk.ecg_simulate(
            duration=dur, sampling_rate=fs, heart_rate=hr, random_state=seed,
            method="ecgsyn",
        )
        return np.asarray(s, dtype=np.float32)

    cycle = [
        ("normal", 75.0, 12.0),
        ("bradycardia", 45.0, 14.0),
        ("recovery", 78.0, 8.0),
    ]
    parts: list[np.ndarray] = []
    t = 0.0
    brady_start = brady_end = 0.0
    seed = 1
    total = 0.0
    while total < duration_sec:
        for name, hr, dur in cycle:
            parts.append(seg(hr, dur, seed))
            seed += 1
            if name == "bradycardia" and brady_start == 0.0:
                brady_start, brady_end = t, t + dur
            t += dur
            total += dur
            if total >= duration_sec:
                break
    sig = np.concatenate(parts)
    return sig, fs, {"brady_start_sec": brady_start, "brady_end_sec": brady_end}
