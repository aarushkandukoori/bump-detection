import { useEffect, useState } from "react";
import {
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api";
import type { AlertRow, Reading, Session } from "../types";

export function HistoricalView() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<Session | null>(null);
  const [readings, setReadings] = useState<Reading[]>([]);
  const [alerts, setAlerts] = useState<AlertRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listSessions().then(setSessions).catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!selected) return;
    setError(null);
    Promise.all([
      api.getSession(selected),
      api.getReadings(selected),
      api.getAlerts(selected),
    ])
      .then(([s, r, a]) => {
        setDetail(s);
        setReadings(r);
        setAlerts(a);
      })
      .catch((e) => setError(String(e)));
  }, [selected]);

  const chartData = readings
    .filter((r) => r.instantaneous_hr != null)
    .map((r, idx) => ({ idx, hr: r.instantaneous_hr as number }));

  return (
    <div className="history">
      <section className="panel">
        <h3>Past sessions</h3>
        {error && <p className="error">{error}</p>}
        {sessions.length === 0 && !error && <p className="muted">No sessions recorded yet.</p>}
        <div className="session-list">
          {sessions.map((s) => (
            <button
              key={s.session_id}
              className={`session-pill ${selected === s.session_id ? "active" : ""}`}
              onClick={() => setSelected(s.session_id)}
            >
              <strong>{s.session_id}</strong>
              <span className="muted">
                {s.source}
                {s.record ? ` · ${s.record}` : ""} · {s.reading_count ?? 0} beats
              </span>
            </button>
          ))}
        </div>
      </section>

      {detail && (
        <>
          <section className="panel">
            <h3>Summary — {detail.session_id}</h3>
            <div className="stat-row">
              <div className="stat">
                <div className="stat-label">Beats</div>
                <div className="stat-value">{detail.summary?.readings ?? 0}</div>
              </div>
              <div className="stat">
                <div className="stat-label">Avg HR</div>
                <div className="stat-value">
                  {detail.summary?.avg_hr != null ? Math.round(detail.summary.avg_hr) : "—"}
                </div>
              </div>
              <div className="stat">
                <div className="stat-label">Min / Max HR</div>
                <div className="stat-value">
                  {detail.summary?.min_hr != null ? Math.round(detail.summary.min_hr) : "—"} /{" "}
                  {detail.summary?.max_hr != null ? Math.round(detail.summary.max_hr) : "—"}
                </div>
              </div>
              <div className="stat bad">
                <div className="stat-label">Alerts</div>
                <div className="stat-value">{detail.summary?.alert_count ?? 0}</div>
              </div>
            </div>
          </section>

          <section className="panel">
            <h3>Heart-rate history</h3>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={chartData} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
                <XAxis dataKey="idx" hide />
                <YAxis domain={[30, 140]} width={40} stroke="#5b6b7a" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: "#10202c", border: "1px solid #21384a", borderRadius: 6 }}
                  formatter={(v: number) => [`${Math.round(v)} bpm`, "HR"]}
                />
                <ReferenceLine y={60} stroke="#e0576b" strokeDasharray="4 4" />
                <Line type="monotone" dataKey="hr" stroke="#4aa8ff" dot={false} strokeWidth={1.6} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </section>

          <section className="panel">
            <h3>Alerts</h3>
            {alerts.length === 0 ? (
              <p className="muted">No alerts in this session.</p>
            ) : (
              <table className="alert-table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Type</th>
                    <th>HR</th>
                    <th>Severity</th>
                    <th>Latency</th>
                    <th>Message</th>
                  </tr>
                </thead>
                <tbody>
                  {alerts.map((a, i) => (
                    <tr key={i} className={a.severity === "critical" ? "row-critical" : ""}>
                      <td>{new Date(a.time).toLocaleTimeString()}</td>
                      <td>{a.type}</td>
                      <td>{a.hr != null ? Math.round(a.hr) : "—"}</td>
                      <td>{a.severity}</td>
                      <td>{a.latency_ms != null ? `${a.latency_ms.toFixed(0)} ms` : "—"}</td>
                      <td>{a.message}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
        </>
      )}
    </div>
  );
}
