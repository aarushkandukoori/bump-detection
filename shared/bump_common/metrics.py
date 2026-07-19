"""Prometheus-style metrics with graceful degradation.

If ``prometheus_client`` is unavailable (e.g. in a minimal test env) every
metric becomes a no-op so importing this never breaks a service.
"""

from __future__ import annotations

from contextlib import suppress

from .constants import LATENCY_BUDGET_MS

try:  # pragma: no cover - exercised via presence/absence of the dependency
    from prometheus_client import Counter, Histogram, start_http_server

    _ENABLED = True
except Exception:  # pragma: no cover
    _ENABLED = False

    class _Noop:
        def labels(self, *a, **k):  # noqa: D401
            return self

        def observe(self, *a, **k):
            pass

        def inc(self, *a, **k):
            pass

    def start_http_server(*a, **k):  # type: ignore[misc]
        pass

    def Counter(*a, **k):  # type: ignore[misc]
        return _Noop()

    def Histogram(*a, **k):  # type: ignore[misc]
        return _Noop()


# Buckets straddling the 250 ms budget so the histogram shows how close we run.
_LATENCY_BUCKETS = (10, 25, 50, 75, 100, 150, 200, LATENCY_BUDGET_MS, 300, 500, 1000)

SENSOR_TO_INFER_LATENCY = Histogram(
    "bump_sensor_to_infer_latency_ms",
    "Latency from sensor frame emit to inference event (ms).",
    buckets=_LATENCY_BUCKETS,
)
SENSOR_TO_ALERT_LATENCY = Histogram(
    "bump_sensor_to_alert_latency_ms",
    "Latency from sensor frame emit to alert publish (ms).",
    buckets=_LATENCY_BUCKETS,
)
BEATS_TOTAL = Counter(
    "bump_beats_total", "Beats classified.", ["session_id", "class_label"]
)
ALERTS_TOTAL = Counter(
    "bump_alerts_total", "Alerts fired.", ["session_id", "type", "severity"]
)
BUDGET_VIOLATIONS = Counter(
    "bump_latency_budget_violations_total",
    "Events exceeding the latency budget.",
    ["stage"],
)

METRICS_ENABLED = _ENABLED


def maybe_start_metrics_server(port: int) -> None:
    """Start the Prometheus scrape endpoint if the client is installed.

    Swallows ``OSError`` (port already bound) so a second process / test
    fixture cannot crash the service lifespan.
    """
    if not (_ENABLED and port > 0):
        return
    with suppress(OSError):
        start_http_server(port)
