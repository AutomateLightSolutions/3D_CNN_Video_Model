// Shared metrics-rendering building blocks, used by the live TrainingPage,
// the History page, and the Compare page — one definition of "what a
// model's evaluation metrics look like" instead of three.
import { STATUS_BADGE } from '../constants.js'

export const fmt  = (v, digits = 4) => v != null ? Number(v).toFixed(digits) : '—'
export const fmtP = (v)             => v != null ? (Number(v) * 100).toFixed(1) + '%' : '—'

export function StatusBadge({ status }) {
  const cls = STATUS_BADGE[status] || 'badge-gray'
  return <span className={`badge ${cls}`}>{status || 'unknown'}</span>
}

export function MetricCard({ label, value, color }) {
  return (
    <div className="metric-card">
      <div className="label">{label}</div>
      <div className="value" style={color ? { color } : {}}>
        {value ?? '—'}
      </div>
    </div>
  )
}

export function PerClassF1Table({ perClassF1 }) {
  if (!perClassF1 || Object.keys(perClassF1).length === 0) return null
  const entries = Object.entries(perClassF1)
    .filter(([cls]) => cls !== 'card_event')
    .sort((a, b) => b[1] - a[1])
  return (
    <div className="card">
      <h2>Per-Class F1 — Event Classification Head</h2>
      <table className="score-guide-table">
        <thead>
          <tr><th>Event</th><th>F1 Score</th><th>Bar</th></tr>
        </thead>
        <tbody>
          {entries.map(([cls, f1]) => (
            <tr key={cls}>
              <td style={{ textTransform: 'capitalize' }}>{cls.replace(/_/g, ' ')}</td>
              <td>{f1.toFixed(3)}</td>
              <td style={{ width: 120 }}>
                <div style={{
                  height: 8, borderRadius: 4,
                  background: `linear-gradient(to right, var(--green) ${(f1 * 100).toFixed(0)}%, var(--surface-2) ${(f1 * 100).toFixed(0)}%)`,
                }} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function FeatureImportanceTable({ importance }) {
  if (!importance || Object.keys(importance).length === 0) return null
  const entries = Object.entries(importance).sort((a, b) => b[1] - a[1]).slice(0, 15)
  const maxVal = entries[0]?.[1] || 1
  return (
    <div className="card">
      <h2>Feature Importance (Top 15)</h2>
      <table className="score-guide-table">
        <thead>
          <tr><th>Feature</th><th>Importance</th><th>Bar</th></tr>
        </thead>
        <tbody>
          {entries.map(([feat, imp]) => (
            <tr key={feat}>
              <td style={{ fontFamily: 'monospace', fontSize: 12 }}>{feat.replace(/_/g, ' ')}</td>
              <td>{(imp * 100).toFixed(1)}%</td>
              <td style={{ width: 140 }}>
                <div style={{
                  height: 8, borderRadius: 4,
                  background: `linear-gradient(to right, #a855f7 ${(imp / maxVal * 100).toFixed(0)}%, var(--surface-2) ${(imp / maxVal * 100).toFixed(0)}%)`,
                }} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// The "Best Model" card — classification + regression head metric groups —
// shared between the live training page and a single history run's detail view.
export function BestMetricsPanel({ best, bestEpoch, bestValLoss, completedAt, title }) {
  if (!bestEpoch || bestEpoch <= 0) return null
  return (
    <div className="card" style={{ borderLeft: '3px solid var(--green)' }}>
      <h2 style={{ marginBottom: 4 }}>{title ?? `Best Model — Epoch ${bestEpoch}`}</h2>
      <p style={{ fontSize: 12, color: 'var(--text-2)', marginBottom: 20 }}>
        Combined Val Loss: {fmt(bestValLoss)}
        {completedAt && <> · Completed: {new Date(completedAt).toLocaleString()}</>}
      </p>

      <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 8 }}>
        Event Classification Head
      </h3>
      <div className="metric-cards">
        <MetricCard label="Val Accuracy" value={fmtP(best.val_accuracy)} color="var(--green)" />
        <MetricCard label="Precision"    value={fmt(best.precision, 3)} />
        <MetricCard label="Recall"       value={fmt(best.recall, 3)} />
        <MetricCard label="F1 Score"     value={fmt(best.weighted_f1, 3)} />
      </div>

      <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: 0.5, margin: '20px 0 8px' }}>
        Highlight Score Regression Head
      </h3>
      <div className="metric-cards">
        <MetricCard label="MAE"     value={fmt(best.mae, 4)} />
        <MetricCard label="MSE"     value={fmt(best.mse, 4)} />
        <MetricCard label="R²"      value={fmt(best.r2, 4)} />
      </div>
    </div>
  )
}
