"""R-peak detection accuracy — synthetic ground truth and real MIT-BIH annotations.

These pin the clinically-critical detector: missing beats corrupts RR/HR and
could hide a bradycardia (the dangerous failure mode).
"""

from __future__ import annotations

import numpy as np
import pytest
from _helpers import score_peaks
from bump_common.signal import pan_tompkins

MITBIH_BEAT_SYMBOLS = set("NLRejAaJSVEF/fQ")


@pytest.mark.parametrize("hr", [45, 60, 72, 90])
def test_synthetic_accuracy(fs, hr):
    """High sensitivity + PPV across rates, including bradycardic 45 bpm."""
    import neurokit2 as nk

    ecg = nk.ecg_simulate(duration=30, sampling_rate=fs, heart_rate=hr, random_state=7)
    ecg = np.asarray(ecg)
    _, info = nk.ecg_peaks(nk.ecg_clean(ecg, sampling_rate=fs), sampling_rate=fs)
    truth = info["ECG_R_Peaks"]

    det = pan_tompkins(ecg, fs)
    se, ppv, tp, fp, fn = score_peaks(truth, det, tol=int(0.15 * fs))
    assert se >= 0.99, f"sensitivity {se:.3f} too low (fn={fn})"
    assert ppv >= 0.98, f"PPV {ppv:.3f} too low (fp={fp})"


@pytest.mark.network
@pytest.mark.parametrize("record", ["100", "234"])
def test_mitbih_accuracy(fs, record):
    """Validate against real MIT-BIH annotations (downloads from PhysioNet)."""
    try:
        import wfdb

        rec = wfdb.rdrecord(record, pn_dir="mitdb")
        ann = wfdb.rdann(record, "atr", pn_dir="mitdb")
    except Exception as exc:  # network/PhysioNet unavailable
        pytest.skip(f"PhysioNet unreachable: {exc}")

    sig = np.asarray(rec.p_signal[:, 0], dtype=float)
    truth = np.array(
        [
            s
            for s, sym in zip(ann.sample, ann.symbol, strict=False)
            if sym in MITBIH_BEAT_SYMBOLS
        ]
    )
    det = pan_tompkins(sig, rec.fs)
    se, ppv, tp, fp, fn = score_peaks(truth, det, tol=int(0.15 * rec.fs))
    assert se >= 0.99, f"record {record}: sensitivity {se:.4f} (fn={fn})"
    assert ppv >= 0.99, f"record {record}: PPV {ppv:.4f} (fp={fp})"
