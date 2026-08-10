import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { listMatches, listPredictionRuns } from '../api/client.js'
import { MODEL_LABELS } from '../constants.js'

function formatDate(iso) {
  return iso ? new Date(iso).toLocaleString() : '—'
}

export default function PredictAdmin() {
  const [matches, setMatches] = useState([])
  const [runsByMatch, setRunsByMatch] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    (async () => {
      try {
        const [ms, runs] = await Promise.all([listMatches(), listPredictionRuns()])
        const completed = runs.filter(r => r.status === 'completed')
        const grouped = {}
        for (const r of completed) {
          (grouped[r.match_id] ??= []).push(r)
        }
        setMatches(ms.filter(m => grouped[m.id]?.length > 0))
        setRunsByMatch(grouped)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    })()
  }, [])

  return (
    <div>
      <Link to="/predict" style={{ fontSize: 13, color: 'var(--accent)', textDecoration: 'none' }}>← Back to Predict</Link>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginTop: 8 }}>
        <h1>Predict Admin</h1>
        <Link className="btn btn-secondary" to="/predict/admin/calibration">All Calibration Data</Link>
      </div>
      <p style={{ color: 'var(--text-2)', marginTop: -8 }}>
        Merge-weight calibration — only matches with at least one completed prediction run show up here.
      </p>
      {error && <div className="error-box">{error}</div>}

      {loading ? (
        <div className="card"><div className="empty-state"><p>Loading…</p></div></div>
      ) : matches.length === 0 ? (
        <div className="card"><div className="empty-state"><p>None yet.</p></div></div>
      ) : (
        matches.map(m => (
          <div key={m.id} className="card" style={{ padding: 0 }}>
            <h2 style={{ padding: '16px 16px 0' }}>{m.name}</h2>
            <table>
              <thead>
                <tr><th>Model</th><th>Device</th><th>Completed</th><th></th></tr>
              </thead>
              <tbody>
                {runsByMatch[m.id].map(run => (
                  <tr key={run.id}>
                    <td style={{ fontWeight: 600 }}>{MODEL_LABELS[run.model_type] || run.model_type}</td>
                    <td>{run.device || '—'}</td>
                    <td style={{ fontSize: 12, color: 'var(--text-1)' }}>{formatDate(run.completed_at)}</td>
                    <td>
                      <Link className="btn btn-primary" style={{ padding: '4px 10px' }} to={`/predict/${run.id}/calibrate`}>
                        Open Calibration
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))
      )}
    </div>
  )
}
