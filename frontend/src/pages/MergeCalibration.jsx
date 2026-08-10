import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import {
  getPredictionRun, getGroundTruthStatus, uploadGroundTruth,
  evaluateWeights, evaluateAllWeights, listWeightEvals, deleteWeightEvals,
  evaluateVisualScoreWeights, evaluateAllVisualScoreWeights, listVisualScoreEvals, deleteVisualScoreEvals,
} from '../api/client.js'
import { MODEL_LABELS, WINDOW_SIZES, DEFAULT_MERGE_WEIGHTS, DEFAULT_VISUAL_SCORE_WEIGHTS } from '../constants.js'
import { fmt, fmtP } from '../components/MetricsDisplay.jsx'
import { bestByF1, WeightSensitivityChart, VisualScoreMetricCharts, useRowSelection } from '../components/CalibrationCharts.jsx'

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
// Merge-weight history table — the calibration ablation record
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
// Merge Weights tab — full original page body
// ---------------------------------------------------------------------------
function MergeWeightsTab({ runId, gtStatus }) {
  const [evals, setEvals] = useState([])
  const [weights, setWeights] = useState({ ...DEFAULT_MERGE_WEIGHTS })
  const [label, setLabel] = useState('')
  const [evaluating, setEvaluating] = useState(false)
  const [evaluatingAll, setEvaluatingAll] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState('')
  const [info, setInfo] = useState('')
  const { selected, toggle, toggleAll, clear } = useRowSelection()

  const loadEvals = useCallback(async () => {
    try {
      setEvals(await listWeightEvals(runId))
    } catch (e) {
      setError(e.message)
    }
  }, [runId])

  useEffect(() => { loadEvals() }, [loadEvals])

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

  const handleDeleteSelected = async () => {
    if (selected.size === 0) return
    if (!window.confirm(`Delete ${selected.size} selected evaluation${selected.size === 1 ? '' : 's'}?`)) return
    setError('')
    setDeleting(true)
    try {
      await deleteWeightEvals(runId, [...selected])
      clear()
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
      {error && <div className="error-box">{error}</div>}
      {info && <div className="card" style={{ borderLeft: '3px solid var(--green)', padding: '10px 16px', fontSize: 13 }}>{info}</div>}

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
        <HistoryTable
          evals={evals} selected={selected} onToggle={toggle}
          onToggleAll={checked => toggleAll(evals.map(ev => ev.id), checked)}
        />
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// VisualScore Weights tab — calibrates the training-label formula
// VisualScore = base_weight*BaseScore(event_class) + flow_weight*OpticalFlow,
// against ground-truth tiles with score != 0, using each tile's own kept 8s
// clip to compute flow. Pure regression (MAE/MSE/R2) — no F1/accuracy here.
// ---------------------------------------------------------------------------
function bestByR2(evals) {
  if (evals.length === 0) return null
  return evals.reduce((best, ev) => (
    best === null || ev.metrics.r2 > best.metrics.r2 ? ev : best
  ), null)
}

function VisualHistoryTable({ evals, selected, onToggle, onToggleAll }) {
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
          <th>Label</th><th>Base Weight</th><th>Flow Weight</th>
          <th>Tiles</th><th>MAE</th><th>MSE</th><th>R²</th><th>When</th>
        </tr>
      </thead>
      <tbody>
        {evals.map(ev => (
          <tr key={ev.id}>
            <td>
              <input type="checkbox" checked={selected.has(ev.id)} onChange={() => onToggle(ev.id)} />
            </td>
            <td>{ev.label || '—'}</td>
            <td style={{ fontFamily: 'monospace', fontSize: 12 }}>{fmt(ev.weights.base, 2)}</td>
            <td style={{ fontFamily: 'monospace', fontSize: 12 }}>{fmt(ev.weights.flow, 2)}</td>
            <td>{ev.n_tiles}</td>
            <td>{fmt(ev.metrics.mae, 4)}</td>
            <td>{fmt(ev.metrics.mse, 4)}</td>
            <td style={{ fontWeight: 600, color: '#a855f7' }}>{fmt(ev.metrics.r2, 4)}</td>
            <td style={{ fontSize: 12, color: 'var(--text-2)' }}>{formatDate(ev.created_at)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function BestVisualMetricsCallout({ evals }) {
  const best = useMemo(() => bestByR2(evals), [evals])
  if (!best) {
    return <p style={{ fontSize: 12, color: 'var(--text-2)' }}>Evaluate at least one configuration to see the best result here.</p>
  }
  return (
    <div style={{ marginTop: 16, borderTop: '1px solid var(--border)', paddingTop: 16 }}>
      <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-2)', textTransform: 'uppercase', marginBottom: 4 }}>
        Best Result So Far (by R²) — {best.n_tiles} tiles
      </h3>
      <p style={{ fontSize: 12, color: 'var(--text-2)', marginBottom: 8 }}>
        base={fmt(best.weights.base, 2)}, flow={fmt(best.weights.flow, 2)}
        {best.label && ` · ${best.label}`}
      </p>
      <div className="metric-cards">
        <div className="metric-card"><div className="label">MAE</div><div className="value">{fmt(best.metrics.mae, 4)}</div></div>
        <div className="metric-card"><div className="label">MSE</div><div className="value">{fmt(best.metrics.mse, 4)}</div></div>
        <div className="metric-card"><div className="label">R²</div><div className="value" style={{ color: '#a855f7' }}>{fmt(best.metrics.r2, 4)}</div></div>
      </div>
    </div>
  )
}


function VisualScoreTab({ runId, gtStatus }) {
  const [evals, setEvals] = useState([])
  const [baseWeight, setBaseWeight] = useState(String(DEFAULT_VISUAL_SCORE_WEIGHTS.base))
  const [label, setLabel] = useState('')
  const [evaluating, setEvaluating] = useState(false)
  const [evaluatingAll, setEvaluatingAll] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState('')
  const [info, setInfo] = useState('')
  const { selected, toggle, toggleAll, clear } = useRowSelection()

  const loadEvals = useCallback(async () => {
    try {
      setEvals(await listVisualScoreEvals(runId))
    } catch (e) {
      setError(e.message)
    }
  }, [runId])

  useEffect(() => { loadEvals() }, [loadEvals])

  const flowWeight = Math.max(0, 1 - (Number(baseWeight) || 0))

  const handleEvaluate = async () => {
    setError('')
    setEvaluating(true)
    try {
      const b = Number(baseWeight) || 0
      const payload = { weights: { base: b, flow: 1 - b }, label: label.trim() || null }
      await evaluateVisualScoreWeights(runId, payload)
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
      const result = await evaluateAllVisualScoreWeights(runId, 0.05)
      setInfo(
        `Evaluated all ${result.total_combos} weight combinations (step 0.05): `
        + `${result.created} new, ${result.skipped_duplicate} already evaluated (skipped). `
        + `First run computes optical flow per tile (~1s/tile) — later runs are instant.`
      )
      await loadEvals()
    } catch (err) {
      setError(err.message)
    } finally {
      setEvaluatingAll(false)
    }
  }

  const handleDeleteSelected = async () => {
    if (selected.size === 0) return
    if (!window.confirm(`Delete ${selected.size} selected evaluation${selected.size === 1 ? '' : 's'}?`)) return
    setError('')
    setDeleting(true)
    try {
      await deleteVisualScoreEvals(runId, [...selected])
      clear()
      await loadEvals()
    } catch (err) {
      setError(err.message)
    } finally {
      setDeleting(false)
    }
  }

  return (
    <div>
      {error && <div className="error-box">{error}</div>}
      {info && <div className="card" style={{ borderLeft: '3px solid var(--green)', padding: '10px 16px', fontSize: 13 }}>{info}</div>}

      <div className="card">
        <h2>Try a Weight Configuration</h2>
        <p style={{ fontSize: 12, color: 'var(--text-2)' }}>
          VisualScore = base_weight × BaseScore(event_class) + flow_weight × OpticalFlow — calibrated against ground-truth
          tiles with a nonzero commentary score, using each tile's own kept 8s clip for optical flow.
        </p>
        <div className="form-row" style={{ alignItems: 'flex-end' }}>
          <div className="form-group">
            <label>Base Weight</label>
            <input type="number" step="0.01" min="0" max="1" value={baseWeight} onChange={e => setBaseWeight(e.target.value)} style={{ width: 100 }} />
          </div>
          <div className="form-group">
            <label>Flow Weight</label>
            <input type="number" value={flowWeight.toFixed(2)} disabled style={{ width: 100 }} />
          </div>
          <div className="form-group" style={{ flex: 1 }}>
            <label>Label (optional)</label>
            <input type="text" value={label} onChange={e => setLabel(e.target.value)} placeholder="e.g. flow-heavy" />
          </div>
          <button className="btn btn-secondary" type="button" onClick={() => setBaseWeight(String(DEFAULT_VISUAL_SCORE_WEIGHTS.base))}>Current Config</button>
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
            title="Evaluate every base/flow combination summing to 1, in steps of 0.05 — 21 combinations, skipping any already evaluated"
          >
            {evaluatingAll ? 'Evaluating all… (first run computes flow per tile, can take minutes)' : 'Evaluate All (step 0.05)'}
          </button>
        </div>
        {!gtStatus.uploaded && (
          <p style={{ fontSize: 12, color: 'var(--text-2)' }}>Upload ground truth above before evaluating.</p>
        )}

        <BestVisualMetricsCallout evals={evals} />
      </div>

      <VisualScoreMetricCharts evals={evals} />

      <div className="card" style={{ padding: 0 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', padding: '16px 16px 0' }}>
          <div>
            <h2 style={{ marginBottom: 4 }}>Evaluation History</h2>
            <p style={{ fontSize: 12, color: 'var(--text-2)' }}>
              Every configuration evaluated for this run, most recent first.
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
        <VisualHistoryTable
          evals={evals} selected={selected} onToggle={toggle}
          onToggleAll={checked => toggleAll(evals.map(ev => ev.id), checked)}
        />
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
  const [tab, setTab] = useState('merge')
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const fileInputRef = useRef(null)

  useEffect(() => {
    getPredictionRun(runId).then(async r => {
      setRun(r)
      setGtStatus(await getGroundTruthStatus(r.match_id))
    }).catch(e => setError(e.message))
  }, [runId])

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

  return (
    <div>
      <Link to="/predict/admin" style={{ fontSize: 13, color: 'var(--accent)', textDecoration: 'none' }}>← Back to Predict Admin</Link>
      <h1 style={{ marginTop: 8 }}>Weight Calibration</h1>
      {run && (
        <p style={{ color: 'var(--text-2)', marginTop: -8 }}>
          Run #{run.id} · {MODEL_LABELS[run.model_type] || run.model_type}
        </p>
      )}
      {error && <div className="error-box">{error}</div>}

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

      <div style={{ display: 'flex', gap: 8, margin: '12px 0' }}>
        <button
          type="button" className={`btn ${tab === 'merge' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setTab('merge')}
        >
          Merge Weights
        </button>
        <button
          type="button" className={`btn ${tab === 'visual' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setTab('visual')}
        >
          VisualScore Weights
        </button>
      </div>

      {tab === 'merge'
        ? <MergeWeightsTab runId={runId} gtStatus={gtStatus} />
        : <VisualScoreTab runId={runId} gtStatus={gtStatus} />}
    </div>
  )
}
