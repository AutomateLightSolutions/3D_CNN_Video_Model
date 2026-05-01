import { useState, useEffect } from 'react'
import { getExportStats, downloadJson, downloadCsv } from '../api/client.js'

const CLASSES = ['goal', 'near_miss', 'penalty', 'red_card', 'key_tackle', 'free_kick', 'crowd_reaction', 'null']
const SCORE_BINS = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']

export default function Export() {
  const [stats, setStats] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    getExportStats()
      .then(setStats)
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div className="error-box">{error}</div>
  if (!stats) return <div style={{ color: 'var(--text-2)', padding: 20 }}>Loading…</div>

  const maxClass = Math.max(1, ...Object.values(stats.class_counts))
  const maxBin = Math.max(1, ...Object.values(stats.score_bins))

  return (
    <div>
      <h1>Export Dataset</h1>

      <div className="stats-grid">
        <div className="stat-box">
          <div className="value">{stats.total}</div>
          <div className="label">Total Clips</div>
        </div>
        <div className="stat-box">
          <div className="value" style={{ color: 'var(--green)' }}>{stats.labeled}</div>
          <div className="label">Labeled</div>
        </div>
        <div className="stat-box">
          <div className="value" style={{ color: 'var(--text-1)' }}>{stats.skipped}</div>
          <div className="label">Skipped</div>
        </div>
        <div className="stat-box">
          <div className="value" style={{ color: 'var(--amber)' }}>{stats.unlabeled}</div>
          <div className="label">Unlabeled</div>
        </div>
      </div>

      <div className="export-grid">
        <div className="card">
          <h2>Per-Class Breakdown</h2>
          <table>
            <thead>
              <tr><th>Event Class</th><th>Count</th><th style={{ width: 120 }}>Distribution</th></tr>
            </thead>
            <tbody>
              {CLASSES.map(cls => {
                const count = stats.class_counts[cls] || 0
                const pct = (count / maxClass) * 100
                return (
                  <tr key={cls}>
                    <td>{cls}</td>
                    <td style={{ fontVariantNumeric: 'tabular-nums' }}>{count}</td>
                    <td>
                      <div style={{ background: 'var(--bg-3)', borderRadius: 3, height: 8, overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${pct}%`, background: 'var(--accent)', borderRadius: 3 }} />
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        <div className="card">
          <h2>Score Distribution</h2>
          <table>
            <thead>
              <tr><th>Score Range</th><th>Count</th><th style={{ width: 120 }}>Distribution</th></tr>
            </thead>
            <tbody>
              {SCORE_BINS.map(bin => {
                const count = stats.score_bins[bin] || 0
                const pct = (count / maxBin) * 100
                const hue = bin === '0.0-0.2' ? 'var(--green)'
                  : bin === '0.8-1.0' ? 'var(--red)'
                  : 'var(--amber)'
                return (
                  <tr key={bin}>
                    <td>{bin}</td>
                    <td style={{ fontVariantNumeric: 'tabular-nums' }}>{count}</td>
                    <td>
                      <div style={{ background: 'var(--bg-3)', borderRadius: 3, height: 8, overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${pct}%`, background: hue, borderRadius: 3 }} />
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <h2>Download</h2>
        <div style={{ display: 'flex', gap: 12 }}>
          <button className="btn btn-primary" onClick={downloadJson} disabled={stats.labeled === 0}>
            Download JSON
          </button>
          <button className="btn btn-secondary" onClick={downloadCsv} disabled={stats.labeled === 0}>
            Download CSV
          </button>
          {stats.labeled === 0 && (
            <span style={{ color: 'var(--text-2)', fontSize: 12, alignSelf: 'center' }}>
              No labeled clips yet.
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
