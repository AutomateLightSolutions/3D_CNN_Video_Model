import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import {
  getPredictionRun, getGroundTruthStatus, uploadGroundTruth,
  evaluateWeights, evaluateAllWeights, listWeightEvals, deleteWeightEvals,
} from '../api/client.js'
import { MODEL_LABELS, WINDOW_SIZES, DEFAULT_MERGE_WEIGHTS } from '../constants.js'
import { fmt, fmtP } from '../components/MetricsDisplay.jsx'
import { bestByF1, WeightSensitivityChart } from '../components/CalibrationCharts.jsx'

function formatDate(iso) {
  return iso ? new Date(iso).toLocaleString() : '—'
}

// ---------------------------------------------------------------------------
// One editable 8s/16s/32s weight row
// ---------------------------------------------------------------------------
function WeightRow({ title, weights, onChange }) {
  const sum = WINDOW_SIZES.reduce((s, ws) => s + (Number(weights[ws]) || 0), 0)
  return (
    <div className="form-group" style={{ flex: 1 }}>
      <label>{title}</label>
      <div style={{ display: 'flex', gap: 8 }}>
        {WINDOW_SIZES.map(ws => (
          <div key={ws} style={{ flex: 1 }}>
            <span style={{ fontSize: 11, color: 'var(--text-2)' }}>{ws}s</span>
            <input
              type="number" step="0.01" min="0" max="1"
              value={weights[ws]}
              onChange={e => onChange({ ...weights, [ws]: e.target.value })}
            />
          </div>
        ))}
      </div>
      <span style={{ fontSize: 11, color: Math.abs(sum - 1) > 0.01 ? 'var(--red)' : 'var(--text-2)' }}>
        sum = {sum.toFixed(2)}{Math.abs(sum - 1) > 0.01 ? ' (renormalized automatically at eval time)' : ''}
      </span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// History table — the calibration ablation record
// ---------------------------------------------------------------------------
function HistoryTable({ evals, selected, onToggle, onToggleAll }) {
  if (evals.length === 0) {
    return <div className="empty-state"><p>No weight configurations evaluated yet.</p></div>
  }
  const allSelected = evals.length > 0 && evals.every(ev => selected.has(ev.id))
  return (
    <table>
      <thead>
        <tr>
          <th style={{ width: 32 }}>
            <input type="checkbox" checked={allSelected} onChange={e => onToggleAll(e.target.checked)} />
          </th>
          <th>Label</th><th>Weights (8/16/32)</th>
          <th>Tiles</th><th>Accuracy</th><th>F1 Score</th>
          <th>MSE</th><th>MAE</th><th>When</th>
        </tr>
      </thead>
      <tbody>
        {evals.map(ev => (
          <tr key={ev.id}>
            <td>
              <input type="checkbox" checked={selected.has(ev.id)} onChange={() => onToggle(ev.id)} />
            </td>
            <td>{ev.label || '—'}</td>
            <td style={{ fontFamily: 'monospace', fontSize: 12 }}>
              {WINDOW_SIZES.map(ws => fmt(ev.weights[ws], 2)).join(' / ')}
            </td>
            <td>{ev.n_tiles}</td>
            <td>{fmtP(ev.metrics.val_accuracy)}</td>
            <td style={{ fontWeight: 600, color: '#a855f7' }}>{fmt(ev.metrics.weighted_f1, 3)}</td>
            <td>{fmt(ev.metrics.mse, 4)}</td>
            <td>{fmt(ev.metrics.mae, 4)}</td>
            <td style={{ fontSize: 12, color: 'var(--text-2)' }}>{formatDate(ev.created_at)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

// ---------------------------------------------------------------------------
// Best-by-F1 callout — derived from the full history, not the last click
// ---------------------------------------------------------------------------
function BestMetricsCallout({ evals }) {
  const best = useMemo(() => bestByF1(evals), [evals])
  if (!best) {
    return <p style={{ fontSize: 12, color: 'var(--text-2)' }}>Evaluate at least one configuration to see the best result here.</p>
  }
  return (
    <div style={{ marginTop: 16, borderTop: '1px solid var(--border)', paddingTop: 16 }}>
      <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-2)', textTransform: 'uppercase', marginBottom: 4 }}>
        Best Result So Far (by F1 Score) — {best.n_tiles} tiles
      </h3>
      <p style={{ fontSize: 12, color: 'var(--text-2)', marginBottom: 8 }}>
        Weights: {WINDOW_SIZES.map(ws => `${ws}s=${fmt(best.weights[ws], 2)}`).join(', ')}
        {best.label && ` · ${best.label}`}
      </p>
      <div className="metric-cards">
        <div className="metric-card"><div className="label">Accuracy</div><div className="value" style={{ color: 'var(--green)' }}>{fmtP(best.metrics.val_accuracy)}</div></div>
        <div className="metric-card"><div className="label">F1 Score</div><div className="value" style={{ color: '#a855f7' }}>{fmt(best.metrics.weighted_f1, 3)}</div></div>
        <div className="metric-card"><div className="label">MSE</div><div className="value">{fmt(best.metrics.mse, 4)}</div></div>
        <div className="metric-card"><div className="label">MAE</div><div className="value">{fmt(best.metrics.mae, 4)}</div></div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------
export default function MergeCalibration() {
  const { runId } = useParams()
  const [run, setRun] = useState(null)
  const [gtStatus, setGtStatus] = useState({ uploaded: false, n_tiles: 0 })
  const [evals, setEvals] = useState([])
  const [weights, setWeights] = useState({ ...DEFAULT_MERGE_WEIGHTS })
  const [label, setLabel] = useState('')
  const [uploading, setUploading] = useState(false)
  const [evaluating, setEvaluating] = useState(false)
  const [evaluatingAll, setEvaluatingAll] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState('')
  const [info, setInfo] = useState('')
  const [selected, setSelected] = useState(new Set())
  const fileInputRef = useRef(null)

  const loadEvals = useCallback(async () => {
    try {
      setEvals(await listWeightEvals(runId))
    } catch (e) {
      setError(e.message)
    }
  }, [runId])

  useEffect(() => {
    getPredictionRun(runId).then(async r => {
      setRun(r)
      setGtStatus(await getGroundTruthStatus(r.match_id))
    }).catch(e => setError(e.message))
    loadEvals()
  }, [runId, loadEvals])

  const handleUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file || !run) return
    setError('')
    setUploading(true)
    try {
      setGtStatus(await uploadGroundTruth(run.match_id, file))
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handleEvaluate = async () => {
    setError('')
    setEvaluating(true)
    try {
      const payload = {
        weights: Object.fromEntries(WINDOW_SIZES.map(ws => [ws, Number(weights[ws]) || 0])),
        label: label.trim() || null,
      }
      await evaluateWeights(runId, payload)
      await loadEvals()
    } catch (err) {
      setError(err.message)
    } finally {
      setEvaluating(false)
    }
  }

  const handleEvaluateAll = async () => {
    setError('')
    setInfo('')
    setEvaluatingAll(true)
    try {
      const result = await evaluateAllWeights(runId, 0.05)
      setInfo(
        `Evaluated all ${result.total_combos} weight combinations (step 0.05): `
        + `${result.created} new, ${result.skipped_duplicate} already evaluated (skipped).`
      )
      await loadEvals()
    } catch (err) {
      setError(err.message)
    } finally {
      setEvaluatingAll(false)
    }
  }

  const toggleSelect = (id) => {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id); else next.add(id)
      return next
    })
  }
  const toggleSelectAll = (checked) => {
    setSelected(checked ? new Set(evals.map(ev => ev.id)) : new Set())
  }

  const handleDeleteSelected = async () => {
    if (selected.size === 0) return
    if (!window.confirm(`Delete ${selected.size} selected evaluation${selected.size === 1 ? '' : 's'}?`)) return
    setError('')
    setDeleting(true)
    try {
      await deleteWeightEvals(runId, [...selected])
      setSelected(new Set())
      await loadEvals()
    } catch (err) {
      setError(err.message)
    } finally {
      setDeleting(false)
    }
  }

  const applyUniform = () => {
    setWeights(Object.fromEntries(WINDOW_SIZES.map(ws => [ws, (1 / WINDOW_SIZES.length).toFixed(4)])))
  }
  const applyDefault = () => {
    setWeights({ ...DEFAULT_MERGE_WEIGHTS })
  }

  return (
    <div>
      <Link to="/predict/admin" style={{ fontSize: 13, color: 'var(--accent)', textDecoration: 'none' }}>← Back to Predict Admin</Link>
      <h1 style={{ marginTop: 8 }}>Merge Weight Calibration</h1>
      {run && (
        <p style={{ color: 'var(--text-2)', marginTop: -8 }}>
          Run #{run.id} · {MODEL_LABELS[run.model_type] || run.model_type} · re-scores this run's already-computed
          per-window predictions under different weights — no model re-inference, so results are instant.
        </p>
      )}
      {error && <div className="error-box">{error}</div>}
      {info && <div className="card" style={{ borderLeft: '3px solid var(--green)', padding: '10px 16px', fontSize: 13 }}>{info}</div>}

      <div className="card">
        <h2>Ground Truth</h2>
        {gtStatus.uploaded ? (
          <p>
            <span className="badge badge-green">Uploaded</span>{' '}
            {gtStatus.n_tiles} tiles have ground truth for this match.
            Re-upload to replace.
          </p>
        ) : (
          <p style={{ color: 'var(--text-2)' }}>
            No ground truth uploaded yet for this match. Upload a CSV with <code>Start,End,Event,Score</code> columns
            (dense, overlapping caption-level rows are fine — they get aggregated onto this run's 8s tile grid).
          </p>
        )}
        <input ref={fileInputRef} type="file" accept=".csv" onChange={handleUpload} disabled={uploading} />
        {uploading && <span style={{ marginLeft: 8, fontSize: 12, color: 'var(--text-2)' }}>Uploading…</span>}
      </div>

      <div className="card">
        <h2>Try a Weight Configuration</h2>
        <div className="form-row">
          <WeightRow title="Weights (used for both class vote and score merge)" weights={weights} onChange={setWeights} />
        </div>
        <div className="form-row" style={{ alignItems: 'flex-end' }}>
          <div className="form-group" style={{ flex: 1 }}>
            <label>Label (optional)</label>
            <input type="text" value={label} onChange={e => setLabel(e.target.value)} placeholder="e.g. performance-weighted" />
          </div>
          <button className="btn btn-secondary" type="button" onClick={applyUniform}>Uniform</button>
          <button className="btn btn-secondary" type="button" onClick={applyDefault}>Current Config</button>
          <button
            className="btn btn-primary" type="button"
            disabled={evaluating || evaluatingAll || !gtStatus.uploaded}
            onClick={handleEvaluate}
          >
            {evaluating ? 'Evaluating…' : 'Evaluate'}
          </button>
          <button
            className="btn btn-secondary" type="button"
            disabled={evaluating || evaluatingAll || !gtStatus.uploaded}
            onClick={handleEvaluateAll}
            title="Evaluate every (w8, w16, w32) combination summing to 1, in steps of 0.05 — 231 combinations, skipping any already evaluated"
          >
            {evaluatingAll ? 'Evaluating all… (this can take ~20s)' : 'Evaluate All (step 0.05)'}
          </button>
        </div>
        {!gtStatus.uploaded && (
          <p style={{ fontSize: 12, color: 'var(--text-2)' }}>Upload ground truth above before evaluating.</p>
        )}

        <BestMetricsCallout evals={evals} />
      </div>

      <WeightSensitivityChart evals={evals} />

      <div className="card" style={{ padding: 0 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', padding: '16px 16px 0' }}>
          <div>
            <h2 style={{ marginBottom: 4 }}>Evaluation History</h2>
            <p style={{ fontSize: 12, color: 'var(--text-2)' }}>
              Every configuration you've evaluated for this run, most recent first — this is the ablation record:
              compare rows to justify which weight configuration was actually chosen and why.
            </p>
          </div>
          <button
            className="btn btn-danger" type="button" style={{ padding: '4px 10px', flexShrink: 0 }}
            disabled={selected.size === 0 || deleting}
            onClick={handleDeleteSelected}
          >
            {deleting ? 'Deleting…' : `Delete Selected (${selected.size})`}
          </button>
        </div>
        <HistoryTable evals={evals} selected={selected} onToggle={toggleSelect} onToggleAll={toggleSelectAll} />
      </div>
    </div>
  )
}
