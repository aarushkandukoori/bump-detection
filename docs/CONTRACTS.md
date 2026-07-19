# BUMP — Internal Contracts (single source of truth)

Every service is built against this document plus the code in `shared/bump_common/`.
Do **not** redefine constants, stream names, or schemas locally — import them from
`bump_common`. All Python services depend on the `bump-common` package
(`pip install -e ./shared`) and are built with the repo root as Docker build
context so they can copy `shared/` in.

## Sampling / signal
- Native sample rate: **360 Hz** (`DEFAULT_SAMPLE_RATE_HZ`).
- R-peak detection, RR/HR math, and the bradycardia rule live in
  `bump_common.signal` — **use them, do not reimplement**:
  - `pan_tompkins(sig, fs) -> np.ndarray[int]` (batch reference)
  - `StreamingRPeakDetector(fs, buffer_sec=8, edge_guard_sec=0.25).update(samples) -> list[int]`
    returns **absolute** sample indices of newly-confirmed R-peaks.
  - `rr_intervals_ms`, `instantaneous_hr(rr_ms)`, `hr_from_peaks`.
  - `BradycardiaMonitor(threshold_bpm=60, min_beats=4, sustain_sec=5).update(hr, t_ms) -> bool`.
- Beat preprocessing lives in `bump_common.beats` — use for train/inference parity:
  - `extract_beat(sig, peak_idx) -> np.ndarray[350]`
  - `make_model_input(window, preceding_rr_ms) -> np.ndarray[351] float32`

## Classes
`CLASS_LABELS = ["Normal", "Bradycardia", "Other"]` (index = list position).
Bradycardia is authoritatively decided by the **rate rule** (`BradycardiaMonitor`),
not by the CNN. The CNN still emits 3 classes so the ONNX artifact matches spec.

## Redis Streams (wire format = `bump_common.schemas`)
| Stream (`constants`) | Producer | Consumer(s) | Payload model |
|---|---|---|---|
| `STREAM_ECG_RAW` = `ecg:raw` | ingestion | inference, api(waveform) | `RawFrame` |
| `STREAM_HR_EVENTS` = `hr:events` | inference | api | `HREvent` |
| `STREAM_ALERTS` = `alerts:events` | inference | api | `AlertEvent` |

Each model has `.to_fields() -> dict[str,str]` and `.from_fields(dict) -> model`.
Produce with `xadd(stream, model.to_fields())`; consume via consumer groups
(`GROUP_INFERENCE`, `GROUP_API_EVENTS`, `GROUP_API_WAVE`). Helpers in
`bump_common.redis_streams` (`add_*`, `ensure_group`, `decode_fields`).

`RawFrame` fields: session_id, frame_seq, **t_emit_ms** (unix ms, the sensor-event
time the latency budget starts from), fs, start_index (abs index of samples[0]),
samples (list[float], length `FRAME_SAMPLES`=8).

`HREvent`: session_id, beat_seq, r_peak_index (abs), t_emit_ms (originating frame),
t_infer_ms, rr_ms, instantaneous_hr, class_label, class_probs, bradycardia,
alert, alert_reason, latency_ms.

`AlertEvent`: session_id, t_sensor_ms, t_alert_ms, type ("bradycardia"|"arrhythmia"),
hr, severity ("warning"|"critical"), message, latency_ms.

## Session metadata handoff
Ingestion, at startup, HSETs `session_meta_key(session_id)` (`session:{id}:meta`)
with `{record, source, fs, started_at}` (started_at = unix seconds). The API reads
this hash when it first sees a session_id, to upsert the `sessions` row. Ingestion
does **not** touch Postgres.

## Latency budget
`LATENCY_BUDGET_MS = 250`. The **server-measurable** slice is sensor frame emit
(`t_emit_ms`) → alert publish (`t_alert_ms`). This is what the integration test
asserts and what `bump_sensor_to_alert_latency_ms` (Prometheus histogram in
`bump_common.metrics`) records. Bradycardia alerts fire from the **rate** path so
they are fast; morphology classification may lag by up to one beat-window
(~0.72 s) and is not on the alert-critical path.

## ONNX model I/O contract
- Architecture + export: `training/model.py` (`BeatCNN`, `export_onnx`). Do not
  redefine the architecture elsewhere.
- Input: name `beat_input` (`ONNX_INPUT_NAME`), shape `(batch, 351)` float32
  = `make_model_input` output. Output: name `logits` (`ONNX_OUTPUT_NAME`),
  shape `(batch, 3)` raw logits.
- Runtime wrapper: `inference/src/bump_inference/classifier.py::BeatClassifier`
  (onnxruntime, single-thread). `.classify_beat(window, preceding_rr_ms)` and
  `.infer_vector(vec)` return a `Classification(label, index, probs)`.
- Default artifact path: `inference/models/beat_cnn.onnx` (mounted to `/models`
  in Docker; `DEFAULT_MODEL_PATH=/models/beat_cnn.onnx`).

## Database (TimescaleDB) — schema in `db/init.sql`
Tables: `sessions(session_id pk, record, source, fs, started_at, ended_at, meta)`,
`hr_readings(time, session_id, beat_seq, instantaneous_hr, rr_ms, class_label,
bradycardia, class_probs, latency_ms)` [hypertable on time],
`alerts(time, session_id, type, hr, severity, message, latency_ms)` [hypertable].
Only the **api** writes to the DB.

## Canonical entrypoints (for compose + integration test)
- ingestion: `python -m bump_ingestion` (env: `RECORD`, `SOURCE`, `SESSION_ID`,
  `SPEED`, `SYNTHETIC_HR`, `DURATION_SEC`).
- inference: `python -m bump_inference`.
- api: `uvicorn bump_api.main:app --host 0.0.0.0 --port 8000`. WS at `/ws/live`,
  REST under `/api/*`, health at `/health`.
- frontend: Vite dev on 5173; production build served by nginx on 80. Talks to the
  API base URL via `VITE_API_BASE` / `VITE_WS_BASE`.

## Env vars (see `.env.example`)
`REDIS_URL`, `DATABASE_URL`, `SAMPLE_RATE_HZ`, `BRADY_BPM_THRESHOLD`,
`BRADY_SUSTAIN_SEC`, `BRADY_MIN_BEATS`, `MODEL_PATH`, `FRAME_SAMPLES`,
`LATENCY_BUDGET_MS`, `*_METRICS_PORT`.
