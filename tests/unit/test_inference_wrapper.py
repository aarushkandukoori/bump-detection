"""ONNX beat-classifier wrapper and beat preprocessing."""

from __future__ import annotations

import numpy as np
from bump_common.beats import make_model_input, normalize_beat
from bump_common.constants import BEAT_WINDOW_LEN, CLASS_LABELS, MODEL_INPUT_DIM


def test_make_model_input_shape():
    window = np.random.randn(BEAT_WINDOW_LEN).astype(np.float32)
    vec = make_model_input(window, preceding_rr_ms=800.0)
    assert vec.shape == (MODEL_INPUT_DIM,)
    assert vec.dtype == np.float32


def test_normalize_beat_zero_mean_unit_std():
    w = np.random.randn(BEAT_WINDOW_LEN).astype(np.float32) * 5 + 3
    n = normalize_beat(w)
    assert abs(float(n.mean())) < 1e-4
    assert abs(float(n.std()) - 1.0) < 1e-3


def test_classifier_outputs_valid_distribution(onnx_model_path):
    from bump_inference.classifier import BeatClassifier

    clf = BeatClassifier(onnx_model_path)
    window = np.random.randn(BEAT_WINDOW_LEN).astype(np.float32)
    res = clf.classify_beat(window, preceding_rr_ms=800.0)

    assert res.label in CLASS_LABELS
    assert abs(sum(res.probs.values()) - 1.0) < 1e-4
    assert set(res.probs) == set(CLASS_LABELS)
    assert 0.0 <= res.confidence <= 1.0


def test_classifier_on_real_beat(onnx_model_path, synthetic_ecg):
    from bump_common.beats import extract_beat
    from bump_common.signal import pan_tompkins
    from bump_inference.classifier import BeatClassifier

    signal, fs, _ = synthetic_ecg
    peaks = pan_tompkins(signal, fs)
    window = extract_beat(signal, int(peaks[5]))
    clf = BeatClassifier(onnx_model_path)
    res = clf.classify_beat(window, preceding_rr_ms=830.0)
    assert res.label in CLASS_LABELS
