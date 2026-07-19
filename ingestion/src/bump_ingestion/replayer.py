"""Replayer: emit ECG samples over the Redis stream paced to real time.

Batches ``FRAME_SAMPLES`` samples per :class:`RawFrame` and stamps each with the
wall-clock emit time (``t_emit_ms``) that the latency budget is measured from.
Pacing uses a monotonic reference clock so timing never accumulates drift.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from bump_common.constants import FRAME_SAMPLES, STREAM_ECG_RAW
from bump_common.redis_streams import add_raw_frame
from bump_common.schemas import RawFrame

log = logging.getLogger("bump.ingestion")


class Replayer:
    def __init__(
        self,
        redis_client: Any,
        session_id: str,
        fs: int,
        speed: float = 1.0,
        frame_samples: int = FRAME_SAMPLES,
        stream: str = STREAM_ECG_RAW,
    ) -> None:
        self.redis = redis_client
        self.session_id = session_id
        self.fs = fs
        self.speed = max(1e-6, speed)
        self.frame_samples = frame_samples
        self.stream = stream
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self, signal: np.ndarray, loop: bool = False) -> None:
        """Stream ``signal`` frame-by-frame at (real time / speed).

        ``speed`` > 1 emits faster than real time (useful for tests); ``speed``
        == 1 is real time. With ``loop`` the signal repeats until :meth:`stop`.
        """
        frame_period = self.frame_samples / self.fs / self.speed  # seconds/frame
        n = len(signal)
        frame_seq = 0
        abs_index = 0
        start_mono = time.monotonic()

        while not self._stop:
            for i in range(0, n, self.frame_samples):
                if self._stop:
                    break
                chunk = signal[i : i + self.frame_samples]
                frame = RawFrame(
                    session_id=self.session_id,
                    frame_seq=frame_seq,
                    t_emit_ms=time.time() * 1000.0,
                    fs=self.fs,
                    start_index=abs_index,
                    samples=[float(x) for x in chunk],
                )
                add_raw_frame(self.redis, self.stream, frame)
                frame_seq += 1
                abs_index += len(chunk)

                # Pace against a monotonic reference so drift can't accumulate.
                target = start_mono + frame_seq * frame_period
                sleep_for = target - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)

            if not loop:
                break

        log.info(
            "Replay finished: %d frames, %d samples (session=%s)",
            frame_seq,
            abs_index,
            self.session_id,
        )
