import { useEffect, useRef, useState } from "react";
import type { LiveMsg } from "./types";

export type WsStatus = "connecting" | "open" | "closed";

// Auto-reconnecting WebSocket hook. Calls onMessage for every parsed envelope.
export function useWebSocket(url: string, onMessage: (m: LiveMsg) => void): WsStatus {
  const [status, setStatus] = useState<WsStatus>("connecting");
  const cbRef = useRef(onMessage);
  cbRef.current = onMessage;

  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | null = null;
    let closed = false;

    const connect = () => {
      setStatus("connecting");
      ws = new WebSocket(url);
      ws.onopen = () => setStatus("open");
      ws.onmessage = (ev) => {
        try {
          cbRef.current(JSON.parse(ev.data) as LiveMsg);
        } catch {
          /* ignore malformed frames */
        }
      };
      ws.onclose = () => {
        setStatus("closed");
        if (!closed) retry = setTimeout(connect, 1500);
      };
      ws.onerror = () => ws?.close();
    };
    connect();

    return () => {
      closed = true;
      if (retry) clearTimeout(retry);
      ws?.close();
    };
  }, [url]);

  return status;
}
