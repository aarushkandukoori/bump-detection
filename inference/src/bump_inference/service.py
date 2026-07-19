"""Redis consumer loop wrapping :class:`SessionPipeline`.

Reads raw ECG frames from ``ecg:raw`` (consumer group), runs each session's
pipeline, and publishes ``hr:events`` and ``alerts:events``.
"""

from __future__ import annotations

import logging
import os

import redis
from bump_common.constants import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SAMPLE_RATE_HZ,
    GROUP_INFERENCE,
    REDIS_URL,
    STREAM_ALERTS,
    STREAM_ECG_RAW,
    STREAM_HR_EVENTS,
)
from bump_common.redis_streams import (
    add_alert_event,
    add_hr_event,
    decode_fields,
    ensure_group,
)
from bump_common.schemas import RawFrame

from .classifier import BeatClassifier
from .pipeline import SessionPipeline

log = logging.getLogger("bump.inference")


class InferenceService:
    def __init__(
        self,
        redis_url: str = REDIS_URL,
        model_path: str = DEFAULT_MODEL_PATH,
        fs: int = DEFAULT_SAMPLE_RATE_HZ,
        consumer: str | None = None,
    ) -> None:
        self.redis = redis.Redis.from_url(redis_url)
        self.fs = fs
        self.consumer = consumer or f"inference-{os.getpid()}"
        self.classifier = self._load_classifier(model_path)
        self.pipelines: dict[str, SessionPipeline] = {}

    @staticmethod
    def _load_classifier(model_path: str) -> BeatClassifier | None:
        try:
            clf = BeatClassifier(model_path)
            log.info("Loaded ONNX beat classifier from %s", model_path)
            return clf
        except Exception as exc:  # missing model or onnxruntime unavailable
            log.warning(
                "No ONNX model at %s (%s); falling back to rate-only labels. "
                "Run `python training/train.py` (or `make model`) to produce one.",
                model_path,
                exc,
            )
            return None

    def _pipeline_for(self, session_id: str) -> SessionPipeline:
        pipe = self.pipelines.get(session_id)
        if pipe is None:
            pipe = SessionPipeline(
                session_id=session_id, fs=self.fs, classifier=self.classifier
            )
            self.pipelines[session_id] = pipe
            log.info("New session pipeline: %s", session_id)
        return pipe

    def run(self) -> None:
        ensure_group(self.redis, STREAM_ECG_RAW, GROUP_INFERENCE)
        log.info(
            "Inference consuming %s as group=%s consumer=%s",
            STREAM_ECG_RAW,
            GROUP_INFERENCE,
            self.consumer,
        )
        while True:
            resp = self.redis.xreadgroup(
                GROUP_INFERENCE,
                self.consumer,
                {STREAM_ECG_RAW: ">"},
                count=64,
                block=1000,
            )
            if not resp:
                continue
            for _stream, messages in resp:
                for msg_id, raw in messages:
                    try:
                        self._handle(decode_fields(raw))
                    except Exception:
                        log.exception("Failed to process frame %s", msg_id)
                    finally:
                        self.redis.xack(STREAM_ECG_RAW, GROUP_INFERENCE, msg_id)

    def _handle(self, fields: dict[str, str]) -> None:
        frame = RawFrame.from_fields(fields)
        pipe = self._pipeline_for(frame.session_id)
        hr_events, alerts = pipe.ingest_frame(frame)
        for evt in hr_events:
            add_hr_event(self.redis, STREAM_HR_EVENTS, evt)
        for alert in alerts:
            add_alert_event(self.redis, STREAM_ALERTS, alert)
            log.info(
                "ALERT %s session=%s hr=%.0f latency=%.1fms",
                alert.type,
                alert.session_id,
                alert.hr,
                alert.latency_ms,
            )
