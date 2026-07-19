"""Ingestion entrypoint: ``python -m bump_ingestion``.

Replays a MIT-BIH record or a synthetic ECG over ``ecg:raw``. Publishes session
metadata to Redis (the API turns it into a ``sessions`` row) so ingestion needs
no database.

Env:
  SOURCE        mitbih | synthetic            (default: mitbih)
  RECORD        MIT-BIH record id             (default: 100)
  SESSION_ID    stream/session id             (default: derived from source)
  SPEED         real-time multiplier          (default: 1.0)
  SYNTHETIC_HR  fixed bpm for plain synthetic (default: use bradycardia demo)
  DURATION_SEC  synthetic length              (default: 120)
  LOOP          "1" to loop the record        (default: 1)
"""

from __future__ import annotations

import logging
import os
import signal as _signal
import sys
import time

import redis
from bump_common.constants import (
    DEFAULT_SAMPLE_RATE_HZ,
    INGESTION_METRICS_PORT,
    REDIS_URL,
    session_meta_key,
)
from bump_common.metrics import maybe_start_metrics_server

from .replayer import Replayer
from .sources import build_bradycardia_demo, mitbih_source, synthetic_source

log = logging.getLogger("bump.ingestion")


def _load_signal() -> tuple[object, int, str, str]:
    source = os.getenv("SOURCE", "mitbih").lower()
    fs = DEFAULT_SAMPLE_RATE_HZ
    if source == "mitbih":
        record = os.getenv("RECORD", "100")
        sig, fs = mitbih_source(record)
        return sig, fs, source, record
    # synthetic
    duration = float(os.getenv("DURATION_SEC", "120"))
    if os.getenv("SYNTHETIC_HR"):
        hr = float(os.getenv("SYNTHETIC_HR"))
        sig, fs = synthetic_source(hr, duration, fs)
        return sig, fs, source, f"synthetic-{hr:.0f}bpm"
    sig, fs, meta = build_bradycardia_demo(fs, duration)
    log.info("Bradycardia demo: brady segment ~%.0f-%.0f s",
             meta["brady_start_sec"], meta["brady_end_sec"])
    return sig, fs, source, "synthetic-brady-demo"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    sig, fs, source, record = _load_signal()

    session_id = os.getenv("SESSION_ID") or f"{source}-{record}".replace(" ", "-")
    speed = float(os.getenv("SPEED", "1.0"))
    loop = os.getenv("LOOP", "1") == "1"

    r = redis.Redis.from_url(REDIS_URL)
    r.hset(
        session_meta_key(session_id),
        mapping={
            "record": record,
            "source": source,
            "fs": str(fs),
            "started_at": str(time.time()),
        },
    )

    maybe_start_metrics_server(INGESTION_METRICS_PORT)
    replayer = Replayer(r, session_id, fs, speed=speed)

    def _shutdown(*_a: object) -> None:
        log.info("Stopping replay")
        replayer.stop()
        sys.exit(0)

    _signal.signal(_signal.SIGINT, _shutdown)
    _signal.signal(_signal.SIGTERM, _shutdown)

    log.info(
        "Streaming session=%s source=%s record=%s fs=%d speed=%.1f loop=%s",
        session_id, source, record, fs, speed, loop,
    )
    replayer.run(sig, loop=loop)


if __name__ == "__main__":
    main()
