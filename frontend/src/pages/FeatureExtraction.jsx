import { useState, useEffect, useRef } from "react";
import {
  startFeatureExtraction,
  stopFeatureExtraction,
  getFeatureExtractionStatus,
  getFeatureExtractionLogs,
} from "../api/client.js";

export default function FeatureExtraction() {
  const [status, setStatus] = useState({ status: "idle", total: 0, done: 0 });
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const logRef = useRef(null);
  const pollRef = useRef(null);

  const refresh = async () => {
    try {
      const [s, l] = await Promise.all([
        getFeatureExtractionStatus(),
        getFeatureExtractionLogs(),
      ]);
      setStatus(s);
      setLogs(l.lines || []);
    } catch (e) {
      setError(e.message);
    }
  };

  useEffect(() => {
    refresh();
    pollRef.current = setInterval(refresh, 3000);
    return () => clearInterval(pollRef.current);
  }, []);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  const handleExtract = async () => {
    setLoading(true);
    setError("");
    try {
      await startFeatureExtraction();
      await refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    setError("");
    try {
      await stopFeatureExtraction();
      await refresh();
    } catch (e) {
      setError(e.message);
    }
  };

  const isRunning = status.status === "running";
  const isDone = status.status === "done";
  const pct =
    status.total > 0 ? Math.round((status.done / status.total) * 100) : 0;

  let badgeClass, badgeText;
  if (isRunning) {
    badgeClass = "badge-green";
    badgeText = "Running";
  } else if (isDone) {
    badgeClass = "badge-green";
    badgeText = "Done";
  } else {
    badgeClass = "badge-gray";
    badgeText = "Idle";
  }

  return (
    <div>
      <h1>Feature Extraction</h1>

      <div className="card">
        <p style={{ color: "var(--text-2)", fontSize: 13, marginBottom: 16 }}>
          Computes a 25-dimensional interpretable feature vector for every
          labeled clip and caches it as{" "}
          <code>Storage/features/&#123;clip_id&#125;.npy</code>.
        </p>

        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 16,
            flexWrap: "wrap",
          }}
        >
          <span className={`badge ${badgeClass}`}>{badgeText}</span>
          <button
            className="btn btn-primary"
            onClick={handleExtract}
            disabled={isRunning || loading}
          >
            {isRunning ? "Extracting…" : "Extract Features"}
          </button>
          <button
            className="btn btn-danger"
            onClick={handleStop}
            disabled={!isRunning}
          >
            Stop
          </button>
          {(isRunning || isDone) && status.total > 0 && (
            <span style={{ fontSize: 13, color: "var(--text-2)" }}>
              {status.done} / {status.total} clips
            </span>
          )}
        </div>

        {error && (
          <div className="error-box" style={{ marginTop: 12 }}>
            {error}
          </div>
        )}

        {status.pose_available === false && (
          <div
            className="error-box"
            style={{ marginTop: 12, borderColor: "var(--amber)", color: "var(--amber)" }}
          >
            MediaPipe pose estimation is unavailable in this environment (mediapipe&gt;=0.10.14
            removed the <code>solutions</code> API). Pose features (indices 15–18: body lean,
            arms-above-shoulder, leg stance, arm extension) were written as <strong>zero</strong>{" "}
            for every clip extracted this run. Install <code>mediapipe==0.10.13</code> and
            re-extract to restore them.
          </div>
        )}

        {(isRunning || isDone) && status.total > 0 && (
          <div style={{ marginTop: 14 }}>
            <div
              style={{
                height: 8,
                borderRadius: 4,
                background: "var(--bg-3)",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  height: "100%",
                  borderRadius: 4,
                  width: `${pct}%`,
                  background: isDone ? "var(--green)" : "var(--accent)",
                  transition: "width 0.4s ease",
                }}
              />
            </div>
            <div style={{ fontSize: 11, color: "var(--text-2)", marginTop: 4 }}>
              {pct}% complete
            </div>
          </div>
        )}
      </div>

      <div className="card" style={{ marginTop: 0 }}>
        <h2>What gets extracted</h2>
        <table className="score-guide-table">
          <thead>
            <tr>
              <th>Group</th>
              <th>Features</th>
              <th>Indices</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Optical Flow</td>
              <td>
                Mean magnitude, direction histogram (8 bins), temporal variance,
                entropy, acceleration
              </td>
              <td style={{ fontFamily: "monospace", fontSize: 12 }}>0 – 10</td>
            </tr>
            <tr>
              <td>Player Detection</td>
              <td>
                Mean player count, cluster count, spatial spread, central-zone
                ratio (YOLOv8)
              </td>
              <td style={{ fontFamily: "monospace", fontSize: 12 }}>11 – 14</td>
            </tr>
            <tr>
              <td>Pose Estimation</td>
              <td>
                Body lean angle, arms-above-shoulder ratio, leg stance width,
                arm extension (MediaPipe)
              </td>
              <td style={{ fontFamily: "monospace", fontSize: 12 }}>15 – 18</td>
            </tr>
            <tr>
              <td>Camera / Scene</td>
              <td>
                Camera motion, zoom indicator, shot boundary count, crowd
                visibility, jersey color ratio
              </td>
              <td style={{ fontFamily: "monospace", fontSize: 12 }}>19 – 24</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="card">
        <h2>Extraction Log</h2>
        <div className="log-viewer" ref={logRef}>
          {logs.length === 0 ? (
            <span style={{ color: "var(--text-2)" }}>No log output yet.</span>
          ) : (
            logs.map((line, i) => (
              <div
                key={i}
                className="log-line"
                style={{
                  color: line.startsWith("WARNING")
                    ? "var(--amber)"
                    : line.startsWith("ERROR")
                      ? "#f87171"
                      : line.startsWith("SKIP")
                        ? "var(--text-2)"
                        : undefined,
                }}
              >
                {line}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
