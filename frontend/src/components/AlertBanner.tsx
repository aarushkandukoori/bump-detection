import type { AlertMsg } from "../types";

interface Props {
  alert: AlertMsg | null;
  onDismiss: () => void;
}

// Prominent, accessible alert banner. role="alert" announces to screen readers —
// this is the layer that, in a real BUMP device, would gate an atropine dose.
export function AlertBanner({ alert, onDismiss }: Props) {
  if (!alert) return null;
  const critical = alert.severity === "critical" || alert.alert_type === "bradycardia";
  return (
    <div
      className={`alert-banner ${critical ? "critical" : "warning"}`}
      role="alert"
      aria-live="assertive"
    >
      <span className="alert-dot" aria-hidden="true" />
      <div className="alert-body">
        <strong>{alert.alert_type === "bradycardia" ? "BRADYCARDIA" : "ARRHYTHMIA"}</strong>
        <span className="alert-msg">
          {alert.message} · HR {Math.round(alert.hr)} bpm · flagged in{" "}
          {alert.latency_ms.toFixed(0)} ms
        </span>
      </div>
      <button className="alert-dismiss" onClick={onDismiss} aria-label="Dismiss alert">
        ×
      </button>
    </div>
  );
}
