"""RR-interval / heart-rate math and the sustained-bradycardia rule."""

from __future__ import annotations

import numpy as np
from bump_common.signal import (
    BradycardiaMonitor,
    hr_from_peaks,
    instantaneous_hr,
    is_plausible_hr,
    rr_intervals_ms,
)


def test_rr_intervals_ms():
    fs = 360
    peaks = np.array([0, 360, 720, 1080])  # exactly 1 s apart
    rr = rr_intervals_ms(peaks, fs)
    assert np.allclose(rr, 1000.0)


def test_instantaneous_hr():
    assert instantaneous_hr(1000.0) == 60.0  # 1 s -> 60 bpm
    assert instantaneous_hr(800.0) == 75.0  # 800 ms -> 75 bpm
    assert np.isnan(instantaneous_hr(0.0))


def test_hr_from_peaks():
    fs = 360
    peaks = np.array([0, int(fs * 0.8), int(fs * 1.6)])  # 75 bpm
    hr = hr_from_peaks(peaks, fs)
    assert np.allclose(hr, 75.0, atol=0.5)


def test_plausibility_bounds():
    assert is_plausible_hr(60)
    assert not is_plausible_hr(10)
    assert not is_plausible_hr(500)


def test_bradycardia_fires_on_sustained_low_rate():
    mon = BradycardiaMonitor(threshold_bpm=60, min_beats=4, sustain_sec=5)
    t = 0.0
    active = False
    for _ in range(12):
        t += 60000 / 45  # 45 bpm beat spacing (signal time)
        active = mon.update(45.0, t)
    assert active, "should fire on sustained 45 bpm"


def test_bradycardia_ignores_single_long_rr():
    """A single dropped beat (one artifactual slow reading) must NOT alert."""
    mon = BradycardiaMonitor(threshold_bpm=60, min_beats=4, sustain_sec=5)
    t = 0.0
    states = []
    for hr in [75, 75, 40, 75, 75, 75]:
        t += 60000 / max(hr, 1)
        states.append(mon.update(hr, t))
    assert not any(states)


def test_bradycardia_resets_on_recovery():
    mon = BradycardiaMonitor(threshold_bpm=60, min_beats=4, sustain_sec=5)
    t = 0.0
    for _ in range(12):
        t += 60000 / 45
        mon.update(45.0, t)
    assert mon.active
    # Recover to a normal rate -> clears.
    for _ in range(3):
        t += 60000 / 75
        mon.update(75.0, t)
    assert not mon.active
