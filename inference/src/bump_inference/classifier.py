"""ONNX beat-classifier wrapper.

Loads the exported BeatCNN and runs it via onnxruntime, simulating the
edge/MCU-constrained inference path. Both the inference service and the unit
tests use this wrapper, so the tested behaviour is the deployed behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from bump_common.beats import make_model_input
from bump_common.constants import (
    IDX_TO_CLASS,
    MODEL_INPUT_DIM,
    ONNX_INPUT_NAME,
    ONNX_OUTPUT_NAME,
)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


@dataclass
class Classification:
    label: str
    index: int
    probs: dict[str, float]

    @property
    def confidence(self) -> float:
        return self.probs.get(self.label, 0.0)


class BeatClassifier:
    """Thin onnxruntime wrapper around the exported BeatCNN."""

    def __init__(self, model_path: str | Path):
        import onnxruntime as ort  # imported lazily so tests can skip if absent

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1  # single-thread: edge-like, deterministic
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name
        # Validate the artifact honours the shared I/O contract.
        in_shape = self.session.get_inputs()[0].shape
        if in_shape[-1] not in (MODEL_INPUT_DIM, "input_dim", None):
            raise ValueError(
                f"Model input dim {in_shape[-1]} != expected {MODEL_INPUT_DIM}"
            )

    def infer_vector(self, vec: np.ndarray) -> Classification:
        """Classify a pre-assembled model input vector of shape (MODEL_INPUT_DIM,)."""
        vec = np.asarray(vec, dtype=np.float32).reshape(1, MODEL_INPUT_DIM)
        logits = self.session.run([self._output_name], {self._input_name: vec})[0]
        probs = _softmax(logits)[0]
        idx = int(np.argmax(probs))
        return Classification(
            label=IDX_TO_CLASS[idx],
            index=idx,
            probs={IDX_TO_CLASS[i]: float(p) for i, p in enumerate(probs)},
        )

    def classify_beat(
        self, window: np.ndarray, preceding_rr_ms: float | None
    ) -> Classification:
        """Classify a raw beat window + preceding RR (does normalisation)."""
        return self.infer_vector(make_model_input(window, preceding_rr_ms))


# Expose the contract names for callers that build inputs manually.
INPUT_NAME = ONNX_INPUT_NAME
OUTPUT_NAME = ONNX_OUTPUT_NAME
