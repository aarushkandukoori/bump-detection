"""Test helpers importable from any test module (``tests`` dir is on pythonpath)."""

from __future__ import annotations

import numpy as np


def score_peaks(true_peaks, det_peaks, tol) -> tuple[float, float, int, int, int]:
    """Sensitivity / PPV of detected peaks vs truth within ``tol`` samples."""
    true_peaks = np.asarray(sorted(true_peaks))
    det = np.asarray(sorted(det_peaks))
    matched: set[int] = set()
    tp = 0
    for t in true_peaks:
        if det.size == 0:
            break
        j = int(np.argmin(np.abs(det - t)))
        if abs(det[j] - t) <= tol and j not in matched:
            matched.add(j)
            tp += 1
    fn = len(true_peaks) - tp
    fp = len(det) - len(matched)
    se = tp / (tp + fn) if (tp + fn) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    return se, ppv, tp, fp, fn
