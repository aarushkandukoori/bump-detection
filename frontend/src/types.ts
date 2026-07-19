// Mirrors the WebSocket envelope + REST payloads from the BUMP API.

export interface HelloMsg {
  type: "hello";
  budget_ms: number;
}

export interface WaveformMsg {
  type: "waveform";
  session_id: string;
  t: number;
  fs: number;
  samples: number[];
}

export interface HrMsg {
  type: "hr";
  session_id: string;
  t: number;
  hr: number | null;
  rr_ms: number | null;
  class_label: string;
  bradycardia: boolean;
  latency_ms: number;
}

export interface AlertMsg {
  type: "alert";
  session_id: string;
  t: number;
  alert_type: string;
  hr: number;
  severity: string;
  message: string;
  latency_ms: number;
}

export type LiveMsg = HelloMsg | WaveformMsg | HrMsg | AlertMsg;

export interface SessionSummary {
  readings: number;
  min_hr: number | null;
  max_hr: number | null;
  avg_hr: number | null;
  brady_beats: number;
  alert_count: number;
}

export interface Session {
  session_id: string;
  record: string | null;
  source: string;
  fs: number;
  started_at: string;
  ended_at: string | null;
  reading_count?: number;
  summary?: SessionSummary;
}

export interface Reading {
  time: string;
  beat_seq: number | null;
  instantaneous_hr: number | null;
  rr_ms: number | null;
  class_label: string | null;
  bradycardia: boolean;
  latency_ms: number | null;
}

export interface AlertRow {
  time: string;
  type: string;
  hr: number | null;
  severity: string;
  message: string;
  latency_ms: number | null;
}
