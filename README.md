# BUMP — Bradycardia & arrhythmia Understanding / Monitoring Pipeline

**BUMP is a real-time ECG monitoring pipeline that detects sustained bradycardia and
abnormal beat morphology from a live sample stream, end-to-end, in well under a
quarter-second.** It replays the MIT-BIH Arrhythmia Database at its native 360 Hz,
runs a streaming Pan-Tompkins QRS detector to find R-peaks, computes beat-to-beat
heart rate, classifies each beat with a 1D-CNN exported to ONNX (run under
`onnxruntime` to simulate an edge/MCU inference path), and applies a **deterministic,
rate-based sustained-bradycardia rule** to decide when an alert should fire. Five
containerized services move data over Redis Streams into TimescaleDB and out to a
live web dashboard, with a server-measured sensor-to-alert latency budget of
**250 ms** enforced by an integration test and exported as a Prometheus histogram.

This is a portfolio project **and** the software prototype for a wearable closed-loop
device: **BUMP** — a hypothetical wrist/patch monitor that would trigger a micro-dose
of **atropine** (a heart-rate-raising drug) when it detects dangerous, sustained
bradycardia. The detection layer in this repo is exactly the logic that would decide
*when* such a device should act. See [**Why this matters for BUMP**](#why-this-matters-for-bump).

> ⚠️ **Not a medical device. Not for clinical use.** This is an engineering prototype
> built on a public research dataset. See [Design decisions & honest limitations](#design-decisions--honest-limitations).

---

## Table of contents

- [Architecture](#architecture)
- [How it works](#how-it-works)
- [Setup & run](#setup--run)
- [Model performance](#model-performance)
- [Testing](#testing)
- [Latency budget](#latency-budget)
- [Why this matters for BUMP](#why-this-matters-for-bump)
- [Design decisions & honest limitations](#design-decisions--honest-limitations)
- [Repository layout](#repository-layout)

---

## Architecture

Five services communicate exclusively through three Redis Streams and one
TimescaleDB instance. Every service is built against the single source of truth in
[`docs/CONTRACTS.md`](docs/CONTRACTS.md) and imports its constants, schemas, signal
processing, and metrics from the shared `bump_common` package — nothing is
reimplemented per-service.

```mermaid
graph LR
    subgraph edge["Sensor / replay"]
        ING["ingestion<br/><small>python -m bump_ingestion</small><br/>MIT-BIH @ 360 Hz or synthetic<br/>emits 8-sample frames (~22 ms)"]
    end

    subgraph streams["Redis Streams (bump_common.schemas wire format)"]
        RAW(["ecg:raw<br/><small>RawFrame</small>"])
        HR(["hr:events<br/><small>HREvent</small>"])
        ALERTS(["alerts:events<br/><small>AlertEvent</small>"])
    end

    subgraph compute["Detection"]
        INF["inference<br/><small>python -m bump_inference</small><br/>StreamingRPeakDetector → RR/HR<br/>BeatClassifier (ONNX) + BradycardiaMonitor"]
    end

    subgraph serve["Serving"]
        API["api<br/><small>uvicorn bump_api.main:app :8000</small><br/>REST /api/* · WS /ws/live · /health"]
        DB[("TimescaleDB<br/><small>sessions · hr_readings · alerts</small>")]
        FE["frontend<br/><small>Vite → nginx</small><br/>live dashboard"]
    end

    ING -- "t_emit_ms starts the clock" --> RAW
    RAW --> INF
    RAW -- "waveform" --> API
    INF --> HR
    INF -- "rate path (fast)" --> ALERTS
    HR --> API
    ALERTS --> API
    API -- "writes only" --> DB
    API -- "WebSocket push" --> FE

    ING -. "HSET session:{id}:meta" .-> META[["session meta<br/><small>Redis hash</small>"]]
    META -. "read once on new session" .-> API

    classDef budget stroke-dasharray: 4 3;
    class RAW,ALERTS budget;
```

**The 250 ms budget** is the *server-measurable* slice from the sensor frame emit
(`RawFrame.t_emit_ms`) to the alert publish (`AlertEvent.t_alert_ms`): the highlighted
`ecg:raw → inference → alerts:events` path. Bradycardia alerts ride the **rate path**,
which does not wait on the CNN, so they stay fast. The browser render hop is *on top
of* this budget and is not counted (see [Latency budget](#latency-budget)).

### Sensor-event → alert sequence

```mermaid
sequenceDiagram
    autonumber
    participant S as Sensor/replay<br/>(ingestion)
    participant R as Redis<br/>ecg:raw
    participant I as inference
    participant A as Redis<br/>alerts:events
    participant P as api + dashboard

    S->>R: XADD RawFrame(t_emit_ms, samples[8])
    Note over S,R: t_emit_ms = start of latency budget
    R->>I: consumer group reads frame
    I->>I: StreamingRPeakDetector.update() → new R-peak (abs idx)
    I->>I: rr_intervals_ms → instantaneous_hr
    I->>I: BradycardiaMonitor.update(hr, t_ms)
    Note over I: fires only after ≥4 beats AND ≥5 s sustained < 60 bpm
    I->>A: XADD AlertEvent(t_sensor_ms, t_alert_ms, latency_ms)
    Note over I,A: latency_ms = t_alert_ms − t_emit_ms<br/>observed into bump_sensor_to_alert_latency_ms
    A->>P: api consumes, writes DB, pushes over WS /ws/live
    Note over P: browser render hop is extra, off-budget
```

---

## How it works

The clinically-critical core lives in `shared/bump_common/` and is imported
identically by the inference service **and** the test suite, so what CI validates
against MIT-BIH annotations is exactly what runs in production.

### 1. MIT-BIH replay at 360 Hz
The **ingestion** service (`python -m bump_ingestion`) streams a record from the
[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/) at its native
**360 Hz**, or generates a synthetic ECG (used to demonstrate sustained bradycardia,
which MIT-BIH is sparse in). Samples are emitted as `RawFrame` messages of
`FRAME_SAMPLES = 8` samples each (~22 ms of signal) onto the `ecg:raw` stream. Each
frame carries `t_emit_ms`, the wall-clock sensor-event time that starts the latency
budget, plus `start_index` (the absolute sample index of `samples[0]`) so downstream
services can reconstruct absolute R-peak positions.

### 2. Pan-Tompkins R-peak detection
`bump_common.signal.StreamingRPeakDetector` wraps the batch `pan_tompkins()`
reference implementation over a bounded rolling buffer. It performs the classic
Pan-Tompkins pipeline — 5–15 Hz band-pass, 5-point derivative, squaring, 150 ms
moving-window integration, and an adaptive dual-threshold with RR-based search-back —
then snaps each detection to the true R-peak fiducial on the band-passed signal. A
peak is only emitted once it sits `edge_guard_sec = 0.25 s` from the buffer's right
edge, guaranteeing its location is stable before it is reported (bounded detection
latency, no retroactive corrections).

> **Verified accuracy:** on MIT-BIH **record 100**, scored against the reference
> annotations, the detector achieves **Sensitivity = 0.9996** and **PPV = 1.0**. The
> unit test (`tests/unit`) re-derives this from the annotation file, so any regression
> in the QRS core fails CI.

### 3. RR / HR computation
From consecutive R-peaks, `rr_intervals_ms(peaks, fs)` gives RR-intervals in
milliseconds and `instantaneous_hr(rr_ms)` gives beat-to-beat heart rate
(`60000 / rr_ms`). Readings outside `[20, 300]` bpm are treated as detector artifacts
and are not allowed to drive alerts.

### 4. 1D-CNN beat classifier → ONNX → onnxruntime (simulated edge inference)
Each confirmed beat is windowed to a fixed 350-sample window (90 pre + 260 post the
R-peak, ≈0.97 s), z-score normalised, and concatenated with a normalised
preceding-RR feature to form the **351-dim** model input — all via
`bump_common.beats.make_model_input`, so training and inference preprocess beats
*identically*. The model (`training/model.py::BeatCNN`) is a small 1D-CNN
(three conv blocks → adaptive pool → dense head, a few tens of thousands of
parameters, no recurrence — deliberately MCU-plausible) that emits raw `logits` over
the three system classes `["Normal", "Bradycardia", "Other"]`.

It is exported to **ONNX** (`export_onnx`, input `beat_input` `(batch, 351)`, output
`logits` `(batch, 3)`) and served through
`inference/src/bump_inference/classifier.py::BeatClassifier`, a single-threaded
`onnxruntime` wrapper (`intra_op_num_threads = 1`, CPU provider) that **simulates the
constrained, deterministic inference an edge MCU would run**. Measured cost is
**≈0.06 ms per beat** on CPU — orders of magnitude inside the per-beat budget, which
is the point: the morphology model is cheap enough to never be the bottleneck.

### 5. Deterministic sustained-bradycardia rule
Bradycardia is fundamentally a **rate** diagnosis, not a beat shape, so the
authoritative trigger is **not** the CNN. `bump_common.signal.BradycardiaMonitor`
fires only when HR stays below `BRADY_BPM_THRESHOLD = 60` bpm for at least
`BRADY_MIN_BEATS = 4` consecutive qualifying beats **and** `BRADY_SUSTAIN_SEC = 5 s`
of elapsed time. The dual "beats **and** seconds" condition is what prevents a single
dropped beat — which doubles the apparent RR and halves instantaneous HR — from
triggering a (simulated) atropine dose. The CNN still emits a `Bradycardia` class so
the ONNX artifact matches the full spec, but the rate rule is the one that decides
alerts.

---

## Setup & run

### Prerequisites
- **Docker** and **Docker Compose** (the only requirement for the full stack).
- For local (non-Docker) development: **Python 3.11**, **Node 20+** (frontend), a
  local **Redis**, and a **TimescaleDB/PostgreSQL** instance.

### Quick start (Docker)

```bash
# 1. Build/train the ONNX model artifact into inference/models/beat_cnn.onnx
make model

# 2. Bring up all five services + Redis + TimescaleDB
docker compose up --build
```

Then open the live dashboard at **<http://localhost:8080>**.

> `make model` runs training and writes `inference/models/beat_cnn.onnx`, which is
> bind-mounted into the inference container at `/models/beat_cnn.onnx`
> (`MODEL_PATH` / `DEFAULT_MODEL_PATH`). For a fast shape-correct artifact without a
> real training run, `python training/model.py --out inference/models/beat_cnn.onnx`
> exports a randomly-initialised BeatCNN.

**Ports**

| Service | In-container | Host | URL / purpose |
|---|---|---|---|
| frontend (nginx) | 80 | **8080** | dashboard — <http://localhost:8080> |
| api (uvicorn) | 8000 | 8000 | REST `/api/*`, WebSocket `/ws/live`, `/health` |
| ingestion metrics | 9101 | 9101 | Prometheus scrape |
| inference metrics | 9102 | 9102 | Prometheus scrape |
| api metrics | 9103 | 9103 | Prometheus scrape |
| redis | 6379 | 6379 | streams |
| timescaledb | 5432 | 5432 | storage |

### Switching the signal source / record

The ingestion service is configured entirely by environment variables (defaults in
`.env.example`). To replay a different MIT-BIH record or drive the synthetic
bradycardia demo, set them in your environment or `docker compose` override:

```bash
# Replay a specific MIT-BIH record
RECORD=100 SOURCE=mitbih docker compose up --build

# Synthetic source with a target heart rate (drives sustained-bradycardia demo)
SOURCE=synthetic SYNTHETIC_HR=45 docker compose up --build
```

| Env var | Meaning | Default |
|---|---|---|
| `RECORD` | MIT-BIH record id to replay | `100` |
| `SOURCE` | `mitbih` or `synthetic` | `mitbih` |
| `SESSION_ID` | session id (auto if unset) | generated |
| `SPEED` | replay speed multiplier (1.0 = real-time) | `1.0` |
| `SYNTHETIC_HR` | target bpm when `SOURCE=synthetic` (unset → sustained-brady demo) | — |
| `DURATION_SEC` | synthetic signal length (MIT-BIH plays/loops the full record) | `120` |

Other shared knobs (see `.env.example` and `bump_common.constants`):
`REDIS_URL`, `DATABASE_URL`, `SAMPLE_RATE_HZ` (360), `FRAME_SAMPLES` (8),
`BRADY_BPM_THRESHOLD` (60), `BRADY_SUSTAIN_SEC` (5), `BRADY_MIN_BEATS` (4),
`MODEL_PATH` (`/models/beat_cnn.onnx`), `LATENCY_BUDGET_MS` (250),
`INGESTION_METRICS_PORT` / `INFERENCE_METRICS_PORT` / `API_METRICS_PORT`, and the
frontend's `VITE_API_BASE` / `VITE_WS_BASE`.

### Running locally without Docker

All Python services depend on the local `bump-common` package, installed editable:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ./shared            # installs bump_common (numpy, scipy, pydantic)

# Start Redis and TimescaleDB yourself, then point the services at them:
export REDIS_URL=redis://localhost:6379/0
export DATABASE_URL=postgresql://bump:bump@localhost:5432/bump

# Each service runs from its own package (separate terminals):
python -m bump_ingestion                                   # ingestion
python -m bump_inference                                   # inference
uvicorn bump_api.main:app --host 0.0.0.0 --port 8000       # api

# Frontend dev server (Vite on :5173, proxies to the API):
cd frontend && npm install && npm run dev
```

Apply the database schema once from `db/init.sql` (Docker does this automatically via
`/docker-entrypoint-initdb.d`).

### Training the model

```bash
python training/train.py            # full training run → training/metrics.json + ONNX export
python training/train.py --smoke    # fast smoke run (tiny subset) for CI / sanity
```

`train.py` uses the frozen `BeatCNN` architecture from `training/model.py`, preprocesses
beats through `bump_common.beats` (train/inference parity), writes per-class metrics to
`training/metrics.json`, and exports `inference/models/beat_cnn.onnx`.

---

## Model performance

Per-class beat-classification metrics from the latest full training run
(`training/metrics.json`, inter-patient DS2 validation). The single most
important number is **Bradycardia recall** (see the callout below).

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.902 | 0.808 | 0.853 | 36181 |
| **Bradycardia** | 0.489 | **0.449** | 0.468 | 4557 |
| Other (arrhythmia) | 0.388 | 0.822 | 0.527 | 3703 |
| **macro avg** | 0.593 | 0.693 | 0.616 | 44441 |
| accuracy | — | — | 0.773 | 44441 |

> Regenerated by `python training/train.py` → `training/metrics.json`. Numbers above
> match the committed metrics file. After a retrain (especially with train-only
> synthetic augmentation), refresh this table from that file.

> ### 🚨 Bradycardia recall is the metric that matters most
> A **false negative — a missed bradycardia** — is the dangerous failure mode. In a
> real BUMP device it would mean a **missed atropine dose** while the patient's heart
> rate stays dangerously low. A false *positive* costs an unnecessary (and, for
> atropine, comparatively low-harm) dose; a false *negative* costs the patient. We
> therefore optimize and gate on **recall of the bradycardia condition**, and — because
> bradycardia is a rate diagnosis — the authoritative trigger is the deterministic
> `BradycardiaMonitor`, not the CNN. The CNN's Bradycardia class is morphology-
> ambiguous (sinus brady looks like Normal); its ~0.45 recall above is an honest
> reflection of that ill-posed 3-class task, **not** the safety-gate performance.
> Live alerts and the dashboard label fuse the rate rule on top of the CNN.

---

## Testing

```bash
# Unit tests (pure signal/schema core — no Redis, no DB, no network):
pytest tests/unit

# Integration test (end-to-end latency + alert path):
pytest tests/integration

# Everything:
pytest
```

**What the tests assert**

- **R-peak accuracy vs MIT-BIH annotations** (`tests/unit`): runs `pan_tompkins` /
  `StreamingRPeakDetector` against the reference annotations and asserts the
  Sensitivity / PPV verified above (Se = 0.9996, PPV = 1.0 on record 100), matching
  detections to annotations within a tolerance window. This guards the clinically
  critical QRS core.
- **RR / HR math** (`tests/unit`): asserts `rr_intervals_ms`, `instantaneous_hr`, and
  `hr_from_peaks` against known peak sequences, including the implausible-HR guard.
- **BradycardiaMonitor semantics** (`tests/unit`): asserts the rule fires only after
  the combined `min_beats` **and** `sustain_sec` conditions are met, and that a single
  long RR (dropped beat) does **not** trigger it.
- **ONNX wrapper** (`tests/unit`): loads `beat_cnn.onnx` through `BeatClassifier` and
  asserts the I/O contract — input `beat_input` `(batch, 351)`, output `logits`
  `(batch, 3)`, valid softmax probabilities summing to 1, and stable labels via
  `classify_beat` / `infer_vector`.
- **End-to-end latency & alert integration** (`tests/integration`): frames a synthetic
  bradycardic signal and drives it through the inference pipeline (`SessionPipeline`),
  asserting (a) a bradycardia `AlertEvent` is published — the false-negative guard — and
  (b) the server-measured `latency_ms` (sensor emit → alert publish) stays within
  `LATENCY_BUDGET_MS = 250`. A `@pytest.mark.redis` variant exercises the full
  `ecg:raw → consumer group → alerts:events` stream path against a live Redis.

**CI overview.** A GitHub Actions workflow runs on every push/PR: it installs
`bump-common` editable (`pip install -e ./shared`), exports a shape-correct ONNX
artifact (or runs `training/train.py --smoke`), then runs `pytest tests/unit` followed
by `pytest tests/integration` (with Redis/TimescaleDB as service containers). Because
the tests import the same `bump_common` functions the services run, a green CI run
means the deployed signal path is the validated one.

---

## Latency budget

**Definition.** The budget is the **server-measurable** slice of the end-to-end path:
from the sensor frame emit time `RawFrame.t_emit_ms` to the alert publish time
`AlertEvent.t_alert_ms`. This is the interval the pipeline actually controls (Redis
transport, R-peak confirmation, rate evaluation, alert publish) and it must stay
within:

```
LATENCY_BUDGET_MS = 250   # bump_common.constants
```

**What it does and does not include.** The budget measures **pipeline processing +
transport latency** for the beat that trips the alert — *not* the 5-second clinical
sustain window, which is an intentional part of the bradycardia definition, not
overhead. The rate path is fast by design: bradycardia alerts are published straight
from `BradycardiaMonitor` and do **not** wait on the CNN. Morphology classification
needs the full 260-sample post-R-peak window to fill (**≈0.72 s**), so it may lag by
up to one beat-window and is deliberately kept **off** the alert-critical path.

**How it's measured & exposed.** The inference service computes
`latency_ms = t_alert_ms − t_sensor_ms` and observes it into the Prometheus histogram
**`bump_sensor_to_alert_latency_ms`** (defined in `bump_common.metrics`, buckets
straddling the 250 ms line: `…, 200, 250, 300, …`). A companion
`bump_sensor_to_infer_latency_ms` tracks the inference hop, and
`bump_latency_budget_violations_total{stage}` counts any breach. Metrics are scraped
from `:9102` (inference), `:9101` (ingestion), and `:9103` (api). The integration
test asserts the same `latency_ms` value the histogram records.

> **The browser render hop is on top of this budget.** The 250 ms covers sensor →
> alert *published to Redis*; the additional WebSocket push (`api` → `/ws/live`) and
> browser paint are real user-perceived latency but are outside the server-measurable
> slice and are not counted against the budget.

---

## Why this matters for BUMP

BUMP's north star is a **wearable, closed-loop rescue device**: a patch or wristband
that continuously watches the heart and, on detecting dangerous **sustained
bradycardia**, delivers a micro-dose of **atropine** — a well-established
anticholinergic that raises heart rate — to buy time before the rate falls to a
syncopal or arrest-level low. The code in this repository is the **decision layer** of
that device: the software that would decide *when* to trigger a dose. Getting that
decision right is a safety problem, and the design reflects it:

- **Bradycardia is a rate diagnosis, so the trigger is a deterministic rate rule, not
  a black-box CNN.** Clinically, bradycardia *is* "sustained HR < 60 bpm" — it is
  defined by rate, not by the shape of any single beat. A dosing decision must be
  **auditable, explainable, and bounded**: `BradycardiaMonitor` is a handful of lines
  a clinician or regulator can read and reason about, with no failure mode where a
  model quietly drifts and starts (or stops) dosing. The CNN adds morphology context
  (is this "Other"/arrhythmic activity?) but never holds the dosing authority.

- **Sustained detection avoids dosing on a single dropped beat.** R-peak detectors
  miss beats, and a missed beat doubles the apparent RR interval and *halves* the
  instantaneous HR — instantly looking like profound bradycardia for one beat. Requiring
  the low rate to persist across **≥4 beats and ≥5 seconds** filters exactly this
  artifact class. In a device that actuates a drug, the cost of over-triggering is a
  real dose into a real bloodstream; the sustain requirement is a safety interlock, not
  a tuning nicety.

- **False negatives dominate the risk calculus.** The asymmetry is stark. A **missed**
  bradycardia (false negative) means the patient's heart rate stays dangerously low and
  the rescue dose never comes — the exact event the device exists to prevent. A
  **spurious** alert (false positive) means an unnecessary atropine micro-dose, whose
  side effects (transient tachycardia, dry mouth, etc.) are comparatively low-harm and
  self-limiting. So the system is tuned to **maximize recall of the bradycardia
  condition** and accept some false positives — which is why "Bradycardia recall" is
  the headline metric, and why the authoritative path is the conservative,
  high-recall rate rule rather than a probabilistic classifier operating near a
  decision boundary.

- **Safety & latency requirements for real hardware.** A server-measurable
  **250 ms** sensor-to-decision budget demonstrates that the detection logic keeps up
  with the signal in real time with wide margin — the morphology model at ~0.06 ms/beat
  is nowhere near the bottleneck. On real hardware the same architecture would run
  on-device: the streaming detector's bounded buffer and the small, non-recurrent CNN
  (run here through `onnxruntime` single-threaded precisely to mimic an MCU) are chosen
  to fit a power- and memory-constrained wearable and to run **deterministically**,
  which is a prerequisite for any device that closes a loop onto a drug. Real hardware
  would add the parts this prototype intentionally simulates — a sensor front-end, BLE
  transport, an actuation/interlock subsystem, lockout timers and maximum-dose limits,
  and regulatory-grade verification — but the timing, determinism, and false-negative
  discipline established here are the load-bearing requirements those layers build on.

---

## Design decisions & honest limitations

- **The bradycardia label is rate-derived, not a morphology ground truth.** MIT-BIH
  annotates beat *types*, not "bradycardia." We layer a rate-based label
  (`BradycardiaMonitor`) on top of the AAMI-style Normal/Other beat grouping, so the
  CNN's `Bradycardia` class exists mainly to keep the 3-class ONNX artifact faithful to
  the spec. The real bradycardia decision is always the deterministic rate rule.
- **MIT-BIH is not a bradycardia-rich dataset.** It is an arrhythmia database, not a
  brady corpus, so most records sit at normal-to-fast rates. Sustained bradycardia is
  demonstrated with the **synthetic** source (`SOURCE=synthetic`, `SYNTHETIC_HR`),
  which is honest about being a controlled demo rather than clinical evidence.
- **Latency is simulated, not sensor-real.** The "sensor" is a replay of digitized ECG,
  and the budget measures the *server* pipeline. A real device adds analog front-end,
  ADC, and **BLE** transport latency — plus the browser render hop that this project
  explicitly counts as extra — none of which are modeled here.
- **Edge inference is emulated on CPU.** The single-threaded `onnxruntime` path
  approximates MCU-constrained, deterministic inference; it is not a measurement on real
  embedded silicon.
- **Not a medical device.** No safety certification, no clinical validation, no
  actuation hardware, no dose-limiting interlocks. **This software must not be used to
  make any real clinical or dosing decision.** It is an engineering demonstration.

---

## Repository layout

```
BUMP-Project/
├── docs/CONTRACTS.md              # single source of truth for interfaces
├── shared/bump_common/            # constants, schemas, signal, beats, metrics, redis_streams
│   └── pyproject.toml             # → pip install -e ./shared  (package: bump-common)
├── ingestion/src/bump_ingestion/  # MIT-BIH / synthetic replay → ecg:raw   (python -m bump_ingestion)
├── inference/
│   ├── src/bump_inference/        # R-peak → HR → ONNX classify → alerts   (python -m bump_inference)
│   └── models/beat_cnn.onnx       # exported artifact (mounted to /models)
├── api/src/bump_api/              # FastAPI: REST /api/*, WS /ws/live, /health, DB writes
├── frontend/                      # Vite dev (:5173) → nginx (:80, host :8080)
├── db/init.sql                    # TimescaleDB schema (sessions, hr_readings, alerts)
├── training/                      # model.py (BeatCNN + export) · train.py (→ metrics.json)
└── tests/{unit,integration}/      # QRS/RR/HR/ONNX unit tests + e2e latency/alert test
```

The deeper interface reference lives in [`docs/CONTRACTS.md`](docs/CONTRACTS.md).
