"""Shared pytest fixtures. Import paths come from ``pyproject.toml`` pythonpath."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "inference" / "models" / "beat_cnn.onnx"


@pytest.fixture(scope="session")
def fs() -> int:
    return 360


@pytest.fixture(scope="session")
def synthetic_ecg(fs):
    """Deterministic synthetic ECG + ground-truth R-peaks (neurokit2)."""
    import neurokit2 as nk

    signal = nk.ecg_simulate(
        duration=30, sampling_rate=fs, heart_rate=72, random_state=42
    )
    signal = np.asarray(signal, dtype=np.float64)
    _, info = nk.ecg_peaks(nk.ecg_clean(signal, sampling_rate=fs), sampling_rate=fs)
    return signal, fs, np.asarray(info["ECG_R_Peaks"], dtype=int)


@pytest.fixture(scope="session")
def onnx_model_path() -> str:
    """Ensure a shape-correct ONNX model exists (build an untrained one if not)."""
    if not MODEL_PATH.exists():
        from model import build_untrained_and_export

        build_untrained_and_export(MODEL_PATH)
    return str(MODEL_PATH)
