import { useState, useEffect, useCallback } from 'react'
import { useSearchParams } from 'react-router-dom'
import { listTrainingRuns, deleteTrainingRun } from '../api/client.js'
import { MODEL_TYPES, MODEL_LABELS } from '../constants.js'
import { StatusBadge, BestMetricsPanel, PerClassF1Table, FeatureImportanceTable, fmt, fmtP } from '../components/MetricsDisplay.jsx'

const POLL_MS = 5000

function formatDate(iso) {
  return iso ? new Date(iso).toLocaleString() : '—'
}

function ConfigSummary({ run }) {
  const parts = [
    run.epochs != null && `${run.epochs} epochs`,
    run.batch_size != null && `batch ${run.batch_size}`,
    run.lr != null && `lr ${run.lr}`,
    run.device,
    run.backbone && `backbone: ${run.backbone}`,
  ].filter(Boolean)
  return <span style={{ fontSize: 12, color: 'var(--text-2)' }}>{parts.join(' · ')}</span>
}

export default function History() {
  const [searchParams, setSearchParams] = useSearchParams()
  const modelFilter = searchParams.get('model') || ''

  const [runs, setRuns] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [expandedId, setExpandedId] = useState(null)

  const refresh = useCallback(async () => {
    try {
      const data = await listTrainingRuns(modelFilter || undefined)
      setRuns(data)
      setError('')
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [modelFilter])

  useEffect(() => {
    setLoading(true)
    refresh()
    const id = setInterval(refresh, POLL_MS)
    return () => clearInterval(id)
  }, [refresh])

  const handleFilterChange = (value) => {
    if (value) setSearchParams({ model: value })
    else setSearchParams({})
  }

  const handleDelete = async (run) => {
    const label = MODEL_LABELS[run.model_type] || run.model_type
    const when = formatDate(run.started_at)
    if (!window.confirm(`Delete this ${label} run (started ${when})? This only removes it from history — saved checkpoint files are kept.`)) {
      return
    }
    try {
      await deleteTrainingRun(run.id)
      setRuns(prev => prev.filter(r => r.id !== run.id))
    } catch (e) {
      setError(e.message)
    }
  }

  const best = (run) => run.metrics?.best || {}

  return (
    <div>
      <h1>Training History</h1>
      {error && <div className="error-box">{error}</div>}

      <div className="card">
        <div className="form-row" style={{ marginBottom: 0 }}>
          <div className="form-group">
            <label>Model</label>
            <select value={modelFilter} onChange={e => handleFilterChange(e.target.value)}>
              <option value="">All Models</option>
              {MODEL_TYPES.map(m => <option key={m} value={m}>{MODEL_LABELS[m]}</option>)}
            </select>
          </div>
          <span style={{ fontSize: 12, color: 'var(--text-2)', marginLeft: 'auto' }}>
            {runs.length} run{runs.length === 1 ? '' : 's'}
          </span>
        </div>
      </div>

      {loading ? (
        <div className="card"><div className="empty-state"><p>Loading…</p></div></div>
      ) : runs.length === 0 ? (
        <div className="card"><div className="empty-state"><p>No training runs recorded yet. Start a training job to see it appear here.</p></div></div>
      ) : (
        <div className="card" style={{ padding: 0 }}>
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Status</th>
                <th>Started</th>
                <th>Completed</th>
                <th>Val Accuracy</th>
                <th>Macro F1</th>
                <th>MAE</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {runs.map(run => (
                <>
                  <tr
                    key={run.id}
                    style={{ cursor: 'pointer' }}
                    onClick={() => setExpandedId(expandedId === run.id ? null : run.id)}
                  >
                    <td>
                      <div style={{ fontWeight: 600 }}>{MODEL_LABELS[run.model_type] || run.model_type}</div>
                      <ConfigSummary run={run} />
                    </td>
                    <td><StatusBadge status={run.status} /></td>
                    <td style={{ fontSize: 12, color: 'var(--text-1)' }}>{formatDate(run.started_at)}</td>
                    <td style={{ fontSize: 12, color: 'var(--text-1)' }}>{formatDate(run.completed_at)}</td>
                    <td>{fmtP(best(run).val_accuracy)}</td>
                    <td>{fmt(best(run).macro_f1, 3)}</td>
                    <td>{fmt(best(run).mae, 4)}</td>
                    <td onClick={e => e.stopPropagation()}>
                      <button className="btn btn-danger" style={{ padding: '4px 10px' }} onClick={() => handleDelete(run)}>
                        Delete
                      </button>
                    </td>
                  </tr>
                  {expandedId === run.id && (
                    <tr key={`${run.id}-detail`}>
                      <td colSpan={8} style={{ background: 'var(--bg-1)', padding: 20 }}>
                        {Object.keys(best(run)).length > 0 ? (
                          <>
                            <BestMetricsPanel
                              title={`${MODEL_LABELS[run.model_type] || run.model_type} — Run Detail`}
                              best={best(run)}
                              bestEpoch={run.metrics?.current_epoch || 1}
                              bestValLoss={best(run).val_loss}
                              completedAt={run.completed_at}
                            />
                            <PerClassF1Table perClassF1={run.metrics?.per_class_f1} />
                            <FeatureImportanceTable importance={run.metrics?.feature_importance} />
                            {run.n_train != null && (
                              <p style={{ fontSize: 12, color: 'var(--text-2)' }}>
                                Trained on {run.n_train} clips · Validated on {run.n_val} clips
                              </p>
                            )}
                          </>
                        ) : (
                          <p style={{ fontSize: 13, color: 'var(--text-2)' }}>
                            No metrics were recorded for this run{run.status === 'running' ? ' yet' : ''}.
                          </p>
                        )}
                      </td>
                    </tr>
                  )}
                </>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
