// Small typed REST client. Base URLs come from Vite env (fall back to same origin
// so the nginx/proxy setup works without configuration).
import type { AlertRow, Reading, Session } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export function wsUrl(sessionId?: string): string {
  const base =
    import.meta.env.VITE_WS_BASE ??
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}`;
  const q = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
  return `${base}/ws/live${q}`;
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

export const api = {
  listSessions: () => getJSON<Session[]>("/api/sessions"),
  getSession: (id: string) => getJSON<Session>(`/api/sessions/${encodeURIComponent(id)}`),
  getReadings: (id: string, limit = 5000) =>
    getJSON<Reading[]>(`/api/sessions/${encodeURIComponent(id)}/readings?limit=${limit}`),
  getAlerts: (id: string) =>
    getJSON<AlertRow[]>(`/api/sessions/${encodeURIComponent(id)}/alerts`),
};
