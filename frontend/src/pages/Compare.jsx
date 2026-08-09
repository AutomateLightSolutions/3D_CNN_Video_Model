import { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { getLeaderboard } from '../api/client.js'
import { MODEL_TYPES, MODEL_LABELS } from '../constants.js'
import { fmt, fmtP } from '../components/MetricsDisplay.jsx'

const POLL_MS = 5000

const METRIC_COLS = [
  { key: 'val_accuracy', label: 'Val Accuracy', format: fmtP,             higherBetter: true },
  { key: 'precision',    label: 'Precision',    format: v => fmt(v, 3),  higherBetter: true },
  { key: 'recall',       label: 'Recall',       format: v => fmt(v, 3),  higherBetter: true },
  { key: 'weighted_f1',  label: 'F1 Score',     format: v => fmt(v, 3),  higherBetter: true },
  { key: 'mae',          label: 'MAE',          format: v => fmt(v, 4),  higherBetter: false },
  { key: 'mse',          label: 'MSE',          format: v => fmt(v, 4),  higherBetter: false },
  { key: 'r2',           label: 'R²',           format: v => fmt(v, 4),  higherBetter: true },
]

function bestValuePerColumn(rows) {
  const best = {}
  for (const col of METRIC_COLS) {
    const values = rows
      .map(r => r.metrics?.best?.[col.key])
      .filter(v => v != null)
    if (values.length === 0) { best[col.key] = null; continue }
    best[col.key] = col.higherBetter ? Math.max(...values) : Math.min(...values)
  }
  return best
}

export default function Compare() {
  const [runsByModel, setRunsByModel] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const refresh = useCallback(async () => {
    try {
      const rows = await getLeaderboard()
      const byModel = {}
      for (const r of rows) byModel[r.model_type] = r
      setRunsByModel(byModel)
      setError('')
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, POLL_MS)
    return () => clearInterval(id)
  }, [refresh])

  const rows = MODEL_TYPES.map(m => runsByModel[m] || null)
  const completedRows = rows.filter(Boolean)
  const bestPerCol = bestValuePerColumn(completedRows)

  return (
    <div>
      <h1>Compare — Latest Model Leaderboard</h1>
      {error && <div className="error-box">{error}</div>}
      <p style={{ fontSize: 12, color: 'var(--text-2)', marginBottom: 16 }}>
        Each row is that model family's most recently <em>completed</em> training run.
        The best value in each column is highlighted.
      </p>

      {loading ? (
        <div className="card"><div className="empty-state"><p>Loading…</p></div></div>
      ) : (
        <div className="card" style={{ padding: 0, overflowX: 'auto' }}>
          <table>
            <thead>
              <tr>
                <th>Model</th>
                {METRIC_COLS.map(col => <th key={col.key}>{col.label}</th>)}
                <th></th>
              </tr>
            </thead>
            <tbody>
              {MODEL_TYPES.map((modelType, i) => {
                const run = rows[i]
                const best = run?.metrics?.best || {}
                return (
                  <tr key={modelType}>
                    <td style={{ fontWeight: 600 }}>{MODEL_LABELS[modelType]}</td>
                    {METRIC_COLS.map(col => {
                      const value = best[col.key]
                      const isBest = value != null && bestPerCol[col.key] != null && value === bestPerCol[col.key]
                      return (
                        <td
                          key={col.key}
                          style={isBest ? { color: 'var(--green)', fontWeight: 700 } : {}}
                        >
                          {value != null ? col.format(value) : '—'}
                        </td>
                      )
                    })}
                    <td>
                      <Link to={`/history?model=${modelType}`} style={{ fontSize: 12, color: 'var(--accent)', textDecoration: 'none' }}>
                        History →
                      </Link>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          {completedRows.length === 0 && (
            <div className="empty-state"><p>No completed training runs yet. Finish training a model to see it here.</p></div>
          )}
        </div>
      )}
    </div>
  )
}
