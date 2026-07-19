import { useCallback, useMemo, useRef, useState } from "react";
import { AlertBanner } from "./components/AlertBanner";
import { HistoricalView } from "./components/HistoricalView";
import { LiveView, type HrPoint, type LiveState, type WavePoint } from "./components/LiveView";
import { wsUrl } from "./api";
import { useWebSocket } from "./useWebSocket";
import type { AlertMsg, LiveMsg } from "./types";

const WAVE_CAP = 720; // ~4 s at 180 Hz (downsampled)
const HR_CAP = 150;

export default function App() {
  const [view, setView] = useState<"live" | "history">("live");
  const [sessionId, setSessionId] = useState<string>("");
  const [alert, setAlert] = useState<AlertMsg | null>(null);
  const [live, setLive] = useState<LiveState>({
    waveform: [],
    hrSeries: [],
    hr: null,
    classLabel: "",
    bradycardia: false,
    latencyMs: null,
    budgetMs: 250,
  });

  const waveIdx = useRef(0);
  const alertTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const onMessage = useCallback((m: LiveMsg) => {
    if (m.type === "hello") {
      setLive((s) => ({ ...s, budgetMs: m.budget_ms }));
      return;
    }
    if (m.type === "waveform") {
      setSessionId((prev) => prev || m.session_id);
      setLive((s) => {
        const pts: WavePoint[] = m.samples.map((v) => ({ i: waveIdx.current++, v }));
        const waveform = [...s.waveform, ...pts];
        return { ...s, waveform: waveform.slice(-WAVE_CAP) };
      });
      return;
    }
    if (m.type === "hr") {
      setSessionId((prev) => prev || m.session_id);
      setLive((s) => {
        const hrSeries: HrPoint[] =
          m.hr != null ? [...s.hrSeries, { t: m.t, hr: m.hr }].slice(-HR_CAP) : s.hrSeries;
        return {
          ...s,
          hrSeries,
          hr: m.hr ?? s.hr,
          classLabel: m.class_label,
          bradycardia: m.bradycardia,
          latencyMs: m.latency_ms,
        };
      });
      return;
    }
    if (m.type === "alert") {
      setSessionId((prev) => prev || m.session_id);
      setAlert(m);
      if (alertTimer.current) clearTimeout(alertTimer.current);
      alertTimer.current = setTimeout(() => setAlert(null), 12000);
    }
  }, []);

  const url = useMemo(() => wsUrl(), []);
  const status = useWebSocket(url, onMessage);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">◈ BUMP</span>
          <span className="tagline">Real-time bradycardia / arrhythmia monitor</span>
        </div>
        <nav className="tabs">
          <button className={view === "live" ? "active" : ""} onClick={() => setView("live")}>
            Live
          </button>
          <button className={view === "history" ? "active" : ""} onClick={() => setView("history")}>
            History
          </button>
        </nav>
      </header>

      <AlertBanner alert={view === "live" ? alert : null} onDismiss={() => setAlert(null)} />

      <main className="content">
        {view === "live" ? (
          <LiveView state={live} status={status} sessionId={sessionId} />
        ) : (
          <HistoricalView />
        )}
      </main>

      <footer className="footer">
        Not a medical device · prototype for the BUMP wearable atropine-pump decision layer
      </footer>
    </div>
  );
}
