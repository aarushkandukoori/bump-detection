-- BUMP TimescaleDB schema.
-- Applied automatically on first container start (mounted into
-- /docker-entrypoint-initdb.d). Idempotent so re-runs are safe.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- One row per replay/monitoring session.
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    record      TEXT,                       -- MIT-BIH record id or synthetic tag
    source      TEXT NOT NULL DEFAULT 'mitbih',  -- 'mitbih' | 'synthetic'
    fs          INTEGER NOT NULL DEFAULT 360,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at    TIMESTAMPTZ,
    meta        JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Per-beat HR readings (time-series hypertable).
CREATE TABLE IF NOT EXISTS hr_readings (
    time              TIMESTAMPTZ      NOT NULL,
    session_id        TEXT             NOT NULL,
    beat_seq          BIGINT,
    instantaneous_hr  DOUBLE PRECISION,
    rr_ms             DOUBLE PRECISION,
    class_label       TEXT,
    bradycardia       BOOLEAN          NOT NULL DEFAULT FALSE,
    class_probs       JSONB,
    latency_ms        DOUBLE PRECISION
);

SELECT create_hypertable('hr_readings', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_hr_session_time
    ON hr_readings (session_id, time DESC);

-- Fired alerts (time-series hypertable).
CREATE TABLE IF NOT EXISTS alerts (
    time         TIMESTAMPTZ      NOT NULL,
    session_id   TEXT             NOT NULL,
    type         TEXT             NOT NULL,   -- 'bradycardia' | 'arrhythmia'
    hr           DOUBLE PRECISION,
    severity     TEXT             NOT NULL DEFAULT 'warning',
    message      TEXT,
    latency_ms   DOUBLE PRECISION
);

SELECT create_hypertable('alerts', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_alerts_session_time
    ON alerts (session_id, time DESC);
