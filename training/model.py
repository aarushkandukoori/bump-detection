"""BeatCNN: lightweight 1D-CNN beat classifier and ONNX export.

Design goals
------------
* Small enough to be plausible for edge/MCU-constrained inference (a few tens of
  thousands of parameters, no recurrent layers).
* Single flat input vector so the ONNX I/O contract is trivial and matches
  ``bump_common.beats.make_model_input`` exactly: ``[waveform(350), rr(1)]``.
* Emits all three system classes (Normal / Bradycardia / Other) so the exported
  artifact matches the BUMP spec; the deterministic rate rule in the inference
  service remains the authoritative bradycardia trigger.

This module is the single source of truth for the architecture. ``train.py``
trains it; ``export_onnx`` writes the artifact that the inference service and
tests load. Nothing else should define the layer stack.
"""

from __future__ import annotations

from pathlib import Path

import torch
from bump_common.constants import (
    BEAT_WINDOW_LEN,
    MODEL_INPUT_DIM,
    NUM_CLASSES,
    ONNX_INPUT_NAME,
    ONNX_OUTPUT_NAME,
)
from torch import nn


class BeatCNN(nn.Module):
    """1D-CNN over a single-beat window plus a scalar preceding-RR feature."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 350 -> 175
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 175 -> 87
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # -> (B, 64, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 1, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, MODEL_INPUT_DIM). Split waveform and the trailing RR scalar.
        waveform = x[:, :BEAT_WINDOW_LEN].unsqueeze(1)  # (B, 1, 350)
        rr = x[:, BEAT_WINDOW_LEN:]  # (B, 1)
        feat = self.features(waveform).squeeze(-1)  # (B, 64)
        combined = torch.cat([feat, rr], dim=1)  # (B, 65)
        return self.head(combined)  # (B, num_classes) raw logits


def export_onnx(model: nn.Module, path: str | Path, opset: int = 18) -> None:
    """Export a trained (or freshly-initialised) BeatCNN to ONNX.

    Uses a dynamic batch axis and the names in the shared contract so the
    inference wrapper can bind by name. Prefer the legacy TorchScript exporter
    when available — the dynamo path is noisier across torch versions and is
    not required for this small static graph.
    """
    model = model.eval().cpu()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, MODEL_INPUT_DIM, dtype=torch.float32)
    export_kwargs = {
        "input_names": [ONNX_INPUT_NAME],
        "output_names": [ONNX_OUTPUT_NAME],
        "dynamic_axes": {
            ONNX_INPUT_NAME: {0: "batch"},
            ONNX_OUTPUT_NAME: {0: "batch"},
        },
        "opset_version": opset,
    }
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **export_kwargs)
    except TypeError:
        # Older torch without the dynamo kwarg.
        torch.onnx.export(model, dummy, str(path), **export_kwargs)


def build_untrained_and_export(path: str | Path) -> None:
    """Convenience: create a randomly-initialised model and export it.

    Lets the inference service / tests load a shape-correct artifact before a
    full training run has produced a real one.
    """
    torch.manual_seed(0)
    export_onnx(BeatCNN(), path)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Export an untrained BeatCNN to ONNX")
    ap.add_argument("--out", default="inference/models/beat_cnn.onnx")
    args = ap.parse_args()
    build_untrained_and_export(args.out)
    print(f"Wrote untrained BeatCNN to {args.out}")
