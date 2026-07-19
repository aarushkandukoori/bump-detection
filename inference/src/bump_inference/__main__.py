"""Inference service entrypoint: ``python -m bump_inference``."""

from __future__ import annotations

import logging
import signal
import sys

from bump_common.constants import INFERENCE_METRICS_PORT
from bump_common.metrics import maybe_start_metrics_server

from .service import InferenceService


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log = logging.getLogger("bump.inference")

    def _shutdown(*_a: object) -> None:
        log.info("Shutting down inference service")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    maybe_start_metrics_server(INFERENCE_METRICS_PORT)
    log.info("Metrics on :%d/metrics", INFERENCE_METRICS_PORT)
    InferenceService().run()


if __name__ == "__main__":
    main()
