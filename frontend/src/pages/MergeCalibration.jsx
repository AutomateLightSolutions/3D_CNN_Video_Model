import { useState, useEffect, useCallback, useRef } from 'react'
import { useParams } from 'react-router-dom'
import {
  getPredictionRun, getGroundTruthStatus, uploadGroundTruth,
  evaluateWeights, listWeightEvals,
} from '../api/client.js'
import { MODEL_LABELS, WINDOW_SIZES, DEFAULT_CLASS_VOTE_WEIGHTS, DEFAULT_SCORE_MERGE_WEIGHTS } from '../constants.js'
import { fmt, fmtP } from '../components/MetricsDisplay.jsx'

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
function HistoryTable({ evals }) {
  if (evals.length === 0) {
    return <div className="empty-state"><p>No weight configurations evaluated yet.</p></div>
  }
  return (
    <table>
      <thead>
        <tr>
          <th>Label</th><th>Class Vote (8/16/32)</th><th>Score Merge (8/16/32)</th>
          <th>Tiles</th><th>Accuracy</th><th>Macro F1</th><th>Weighted F1</th>
          <th>MAE</th><th>R²</th><th>When</th>
        </tr>
      </thead>
      <tbody>
        {evals.map(ev => (
          <tr key={ev.id}>
            <td>{ev.label || '—'}</td>
            <td style={{ fontFamily: 'monospace', fontSize: 12 }}>
              {WINDOW_SIZES.map(ws => fmt(ev.class_vote_weights[ws], 2)).join(' / ')}
            </td>
            <td style={{ fontFamily: 'monospace', fontSize: 12 }}>
              {WINDOW_SIZES.map(ws => fmt(ev.score_merge_weights[ws], 2)).join(' / ')}
            </td>
            <td>{ev.n_tiles}</td>
            <td>{fmtP(ev.metrics.val_accuracy)}</td>
            <td style={{ fontWeight: 600, color: '#a855f7' }}>{fmt(ev.metrics.macro_f1, 3)}</td>
            <td>{fmt(ev.metrics.weighted_f1, 3)}</td>
            <td>{fmt(ev.metrics.mae, 4)}</td>
            <td>{fmt(ev.metrics.r2, 4)}</td>
            <td style={{ fontSize: 12, color: 'var(--text-2)' }}>{formatDate(ev.created_at)}</td>
          </tr>
        ))}
      </tbody>
    </table>
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
  const [classWeights, setClassWeights] = useState({ ...DEFAULT_CLASS_VOTE_WEIGHTS })
  const [scoreWeights, setScoreWeights] = useState({ ...DEFAULT_SCORE_MERGE_WEIGHTS })
  const [label, setLabel] = useState('')
  const [uploading, setUploading] = useState(false)
  const [evaluating, setEvaluating] = useState(false)
  const [error, setError] = useState('')
  const [latest, setLatest] = useState(null)
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
        class_vote_weights: Object.fromEntries(WINDOW_SIZES.map(ws => [ws, Number(classWeights[ws]) || 0])),
        score_merge_weights: Object.fromEntries(WINDOW_SIZES.map(ws => [ws, Number(scoreWeights[ws]) || 0])),
        label: label.trim() || null,
      }
      const result = await evaluateWeights(runId, payload)
      setLatest(result)
      await loadEvals()
    } catch (err) {
      setError(err.message)
    } finally {
      setEvaluating(false)
    }
  }

  const applyUniform = () => {
    const u = Object.fromEntries(WINDOW_SIZES.map(ws => [ws, (1 / WINDOW_SIZES.length).toFixed(4)]))
    setClassWeights(u)
    setScoreWeights(u)
  }
  const applyDefault = () => {
    setClassWeights({ ...DEFAULT_CLASS_VOTE_WEIGHTS })
    setScoreWeights({ ...DEFAULT_SCORE_MERGE_WEIGHTS })
  }

  return (
    <div>
      <h1>Merge Weight Calibration</h1>
      {run && (
        <p style={{ color: 'var(--text-2)', marginTop: -8 }}>
          Run #{run.id} · {MODEL_LABELS[run.model_type] || run.model_type} · re-scores this run's already-computed
          per-window predictions under different weights — no model re-inference, so results are instant.
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

      <div className="card">
        <h2>Try a Weight Configuration</h2>
        <div className="form-row">
          <WeightRow title="Class Vote Weights" weights={classWeights} onChange={setClassWeights} />
          <WeightRow title="Score Merge Weights" weights={scoreWeights} onChange={setScoreWeights} />
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
            disabled={evaluating || !gtStatus.uploaded}
            onClick={handleEvaluate}
          >
            {evaluating ? 'Evaluating…' : 'Evaluate'}
          </button>
        </div>
        {!gtStatus.uploaded && (
          <p style={{ fontSize: 12, color: 'var(--text-2)' }}>Upload ground truth above before evaluating.</p>
        )}

        {latest && (
          <div style={{ marginTop: 16, borderTop: '1px solid var(--border)', paddingTop: 16 }}>
            <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-2)', textTransform: 'uppercase', marginBottom: 8 }}>
              Latest Result ({latest.n_tiles} tiles)
            </h3>
            <div className="metric-cards">
              <div className="metric-card"><div className="label">Accuracy</div><div className="value" style={{ color: 'var(--green)' }}>{fmtP(latest.metrics.val_accuracy)}</div></div>
              <div className="metric-card"><div className="label">Macro F1</div><div className="value" style={{ color: '#a855f7' }}>{fmt(latest.metrics.macro_f1, 3)}</div></div>
              <div className="metric-card"><div className="label">Weighted F1</div><div className="value">{fmt(latest.metrics.weighted_f1, 3)}</div></div>
              <div className="metric-card"><div className="label">MAE</div><div className="value">{fmt(latest.metrics.mae, 4)}</div></div>
              <div className="metric-card"><div className="label">R²</div><div className="value">{fmt(latest.metrics.r2, 4)}</div></div>
            </div>
          </div>
        )}
      </div>

      <div className="card" style={{ padding: 0 }}>
        <h2 style={{ padding: '16px 16px 0' }}>Evaluation History</h2>
        <p style={{ padding: '0 16px', fontSize: 12, color: 'var(--text-2)' }}>
          Every configuration you've evaluated for this run, most recent first — this is the ablation record:
          compare rows to justify which weight configuration was actually chosen and why.
        </p>
        <HistoryTable evals={evals} />
      </div>
    </div>
  )
}
