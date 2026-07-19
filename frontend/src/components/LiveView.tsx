import {
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { WsStatus } from "../useWebSocket";

export interface WavePoint {
  i: number;
  v: number;
}
export interface HrPoint {
  t: number;
  hr: number;
}
export interface LiveState {
  waveform: WavePoint[];
  hrSeries: HrPoint[];
  hr: number | null;
  classLabel: string;
  bradycardia: boolean;
  latencyMs: number | null;
  budgetMs: number;
}

interface Props {
  state: LiveState;
  status: WsStatus;
  sessionId: string;
}

function Stat({ label, value, unit, tone }: { label: string; value: string; unit?: string; tone?: string }) {
  return (
    <div className={`stat ${tone ?? ""}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">
        {value}
        {unit && <span className="stat-unit"> {unit}</span>}
      </div>
    </div>
  );
}

export function LiveView({ state, status, sessionId }: Props) {
  const { waveform, hrSeries, hr, classLabel, bradycardia, latencyMs, budgetMs } = state;
  const latencyTone =
    latencyMs == null ? "" : latencyMs < budgetMs ? "good" : "bad";
  const hrTone = bradycardia ? "bad" : hr != null && hr > 100 ? "warn" : "good";

  return (
    <div className="live">
      <div className="stat-row">
        <Stat label="Session" value={sessionId || "—"} />
        <Stat label="Heart rate" value={hr != null ? Math.round(hr).toString() : "—"} unit="bpm" tone={hrTone} />
        <Stat label="Classification" value={classLabel || "—"} tone={bradycardia ? "bad" : ""} />
        <Stat
          label="Reaction latency"
          value={latencyMs != null ? latencyMs.toFixed(0) : "—"}
          unit={`/ ${budgetMs}ms`}
          tone={latencyTone}
        />
        <Stat label="Stream" value={status} tone={status === "open" ? "good" : "warn"} />
      </div>

      <section className="panel">
        <h3>Live ECG waveform</h3>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={waveform} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
            <XAxis dataKey="i" hide />
            <YAxis domain={["auto", "auto"]} width={40} stroke="#5b6b7a" tick={{ fontSize: 11 }} />
            <Line
              type="monotone"
              dataKey="v"
              stroke="#38e0a0"
              dot={false}
              strokeWidth={1.4}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </section>

      <section className="panel">
        <h3>Heart-rate trend</h3>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={hrSeries} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
            <XAxis dataKey="t" hide />
            <YAxis domain={[30, 140]} width={40} stroke="#5b6b7a" tick={{ fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: "#10202c", border: "1px solid #21384a", borderRadius: 6 }}
              labelFormatter={() => ""}
              formatter={(v: number) => [`${Math.round(v)} bpm`, "HR"]}
            />
            <ReferenceLine y={60} stroke="#e0576b" strokeDasharray="4 4" label={{ value: "brady <60", fill: "#e0576b", fontSize: 11, position: "insideBottomRight" }} />
            <Line
              type="monotone"
              dataKey="hr"
              stroke="#4aa8ff"
              dot={false}
              strokeWidth={2}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </section>
    </div>
  );
}
