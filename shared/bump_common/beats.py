"""Beat-window extraction and normalisation.

Training and inference MUST preprocess beats identically or the ONNX model sees
a different distribution at runtime than it trained on. Both import these
functions so preprocessing is defined exactly once.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    BEAT_WINDOW_LEN,
    BEAT_WINDOW_POST_SAMPLES,
    BEAT_WINDOW_PRE_SAMPLES,
)


def extract_beat(
    sig: np.ndarray,
    peak_idx: int,
    pre: int = BEAT_WINDOW_PRE_SAMPLES,
    post: int = BEAT_WINDOW_POST_SAMPLES,
) -> np.ndarray:
    """Extract a fixed-length window centred on ``peak_idx``.

    Edges are zero-padded so the returned window is always ``pre + post``
    samples, letting beats near the start/end of a buffer be classified.
    """
    sig = np.asarray(sig, dtype=np.float32)
    n = len(sig)
    lo, hi = peak_idx - pre, peak_idx + post
    window = np.zeros(pre + post, dtype=np.float32)
    src_lo, src_hi = max(0, lo), min(n, hi)
    if src_hi > src_lo:
        dst_lo = src_lo - lo
        window[dst_lo : dst_lo + (src_hi - src_lo)] = sig[src_lo:src_hi]
    return window


def normalize_beat(window: np.ndarray) -> np.ndarray:
    """Per-beat z-score normalisation (zero mean, unit std).

    Removes baseline wander and amplitude scaling differences between records
    and (in deployment) between sensors, so morphology is what the CNN keys on.
    """
    window = np.asarray(window, dtype=np.float32)
    mean = float(window.mean())
    std = float(window.std())
    if std < 1e-6:
        return window - mean
    return (window - mean) / std


def rr_feature(preceding_rr_ms: float | None) -> float:
    """Normalised preceding-RR auxiliary feature fed alongside the waveform.

    Scaled so a normal ~800 ms RR maps near 1.0. Gives the CNN rate context
    (relevant to the Bradycardia class) without it having to infer rate from a
    single-beat window. Missing RR -> 0.0 (neutral).
    """
    if preceding_rr_ms is None or preceding_rr_ms <= 0:
        return 0.0
    return float(preceding_rr_ms / 800.0)


def make_model_input(
    window: np.ndarray, preceding_rr_ms: float | None
) -> np.ndarray:
    """Assemble the full CNN input vector: normalised waveform + RR feature.

    Returns shape ``(BEAT_WINDOW_LEN + 1,)`` float32. The model reshapes the
    waveform to ``(1, BEAT_WINDOW_LEN)`` for the conv stack and concatenates the
    scalar RR feature before the dense head.
    """
    norm = normalize_beat(window)
    if norm.shape[0] != BEAT_WINDOW_LEN:
        fixed = np.zeros(BEAT_WINDOW_LEN, dtype=np.float32)
        m = min(BEAT_WINDOW_LEN, norm.shape[0])
        fixed[:m] = norm[:m]
        norm = fixed
    return np.concatenate([norm, [rr_feature(preceding_rr_ms)]]).astype(np.float32)
