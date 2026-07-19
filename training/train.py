"""Train BeatCNN on MIT-BIH and export to ONNX.

Reports per-class precision/recall/F1 (bradycardia recall is the headline metric
— a missed bradycardia is the dangerous failure mode) and writes
``training/metrics.json`` for the README.

Usage:
    python training/train.py                 # full DS1/DS2 inter-patient split
    python training/train.py --smoke         # 1 train / 1 val record, 1 epoch
    python training/train.py --epochs 25 --out inference/models/beat_cnn.onnx
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from bump_common.constants import CLASS_LABELS, DEFAULT_SAMPLE_RATE_HZ, NUM_CLASSES
from dataset import build_dataset, build_synthetic_rate_beats, get_splits
from model import BeatCNN, export_onnx
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bump.training")


def _class_weights(y: np.ndarray, brady_boost: float = 2.5) -> torch.Tensor:
    counts = Counter(int(c) for c in y)
    total = len(y)
    # Inverse-frequency weights, upweighting minority classes (protects the
    # bradycardia/other recall that matters clinically). Extra boost on
    # Bradycardia because a false negative is the dangerous failure mode.
    brady_idx = CLASS_LABELS.index("Bradycardia")
    weights = [
        total / (NUM_CLASSES * max(1, counts.get(i, 0))) for i in range(NUM_CLASSES)
    ]
    weights[brady_idx] *= brady_boost
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[int] = []
    truth: list[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            preds.extend(logits.argmax(1).cpu().tolist())
            truth.extend(yb.tolist())
    return np.asarray(truth), np.asarray(preds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="inference/models/beat_cnn.onnx")
    ap.add_argument("--metrics", default="training/metrics.json")
    ap.add_argument("--smoke", action="store_true", help="tiny fast run")
    ap.add_argument("--limit-per-record", type=int, default=None)
    ap.add_argument("--no-synthetic", action="store_true",
                    help="disable synthetic rate augmentation (MIT-BIH only)")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 1
        args.limit_per_record = args.limit_per_record or 500

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_recs, val_recs = get_splits(smoke=args.smoke)
    log.info("Building datasets (train=%s val=%s)…", train_recs, val_recs)
    x_tr, y_tr = build_dataset(train_recs, args.limit_per_record)
    x_va, y_va = build_dataset(val_recs, args.limit_per_record)

    # Augment TRAIN only with rate-labeled synthetic beats. Validation stays
    # pure MIT-BIH so reported Brady recall is not inflated by synthetic data.
    # See dataset.build_synthetic_rate_beats.
    if not args.no_synthetic:
        sx_tr, sy_tr = build_synthetic_rate_beats(
            DEFAULT_SAMPLE_RATE_HZ, seed=1000, seconds_per_rate=40.0
        )
        x_tr, y_tr = np.concatenate([x_tr, sx_tr]), np.concatenate([y_tr, sy_tr])
        log.info("Added synthetic rate beats: +%d train (val stays MIT-BIH only)", len(sy_tr))

    log.info("Train %d beats %s | Val %d beats %s",
             len(y_tr), dict(Counter(y_tr.tolist())),
             len(y_va), dict(Counter(y_va.tolist())))

    train_ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    model = BeatCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(y_tr).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        log.info("epoch %d/%d  loss=%.4f", epoch, args.epochs, running / max(1, len(train_ds)))

    # --- evaluation ---
    truth, preds = evaluate(model, val_dl, device)
    report = classification_report(
        truth, preds, labels=list(range(NUM_CLASSES)),
        target_names=CLASS_LABELS, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(truth, preds, labels=list(range(NUM_CLASSES)))

    print("\n=== Per-class performance (validation, inter-patient DS2) ===")
    print(classification_report(
        truth, preds, labels=list(range(NUM_CLASSES)),
        target_names=CLASS_LABELS, zero_division=0,
    ))
    print("Confusion matrix (rows=true, cols=pred):")
    print("        " + "  ".join(f"{c[:5]:>6}" for c in CLASS_LABELS))
    for i, row in enumerate(cm):
        print(f"{CLASS_LABELS[i][:7]:>7} " + "  ".join(f"{v:>6}" for v in row))

    brady_recall = report.get("Bradycardia", {}).get("recall", 0.0)
    print(f"\n>>> BRADYCARDIA RECALL (the metric that matters most): {brady_recall:.3f}")
    print("    A false negative here = a missed dose in a real BUMP device.\n")

    metrics = {
        "per_class": {c: report[c] for c in CLASS_LABELS},
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
        "accuracy": report["accuracy"],
        "bradycardia_recall": brady_recall,
        "confusion_matrix": cm.tolist(),
        "class_labels": CLASS_LABELS,
        "train_counts": {CLASS_LABELS[k]: int(v) for k, v in Counter(y_tr.tolist()).items()},
        "val_counts": {CLASS_LABELS[k]: int(v) for k, v in Counter(y_va.tolist()).items()},
        "smoke": args.smoke,
    }
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2))
    log.info("Wrote metrics -> %s", args.metrics)

    export_onnx(model, args.out)
    log.info("Exported ONNX -> %s", args.out)


if __name__ == "__main__":
    main()
