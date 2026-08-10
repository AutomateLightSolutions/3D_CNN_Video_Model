import { useState, useEffect, useCallback, useMemo } from 'react'
import { Link } from 'react-router-dom'
import {
  listAllWeightEvals, deleteWeightEvalsGlobal,
  listAllVisualScoreEvals, deleteVisualScoreEvalsGlobal,
} from '../api/client.js'
import { MODEL_LABELS, WINDOW_SIZES } from '../constants.js'
import { fmt, fmtP } from '../components/MetricsDisplay.jsx'
import { bestByF1, WeightSensitivityChart, VisualScoreMetricCharts, useRowSelection } from '../components/CalibrationCharts.jsx'

function formatDate(iso) {
  return iso ? new Date(iso).toLocaleString() : '—'
}

// ---------------------------------------------------------------------------
// Merge Weights overview tab
// ---------------------------------------------------------------------------
function bestPerModel(evals) {
  const byModel = {}
  for (const ev of evals) {
    (byModel[ev.model_type] ??= []).push(ev)
  }
  return Object.entries(byModel)
    .map(([modelType, modelEvals]) => ({ modelType, best: bestByF1(modelEvals), n: modelEvals.length }))
    .sort((a, b) => (b.best?.metrics.weighted_f1 ?? 0) - (a.best?.metrics.weighted_f1 ?? 0))
}

function BestPerModelTable({ evals }) {
  const rows = useMemo(() => bestPerModel(evals), [evals])
  if (rows.length === 0) {
    return <p style={{ fontSize: 12, color: 'var(--text-2)' }}>No evaluations yet.</p>
  }
  return (
    <table>
      <thead>
        <tr>
          <th>Model</th><th>Best Weights (8/16/32)</th>
          <th>Accuracy</th><th>F1 Score</th><th>MSE</th><th>MAE</th><th>Configs Tried</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(({ modelType, best, n }) => (
          <tr key={modelType}>
            <td style={{ fontWeight: 600 }}>{MODEL_LABELS[modelType] || modelType}</td>
            <td style={{ fontFamily: 'monospace', fontSize: 12 }}>
              {WINDOW_SIZES.map(ws => fmt(best.weights[ws], 2)).join(' / ')}
            </td>
            <td>{fmtP(best.metrics.val_accuracy)}</td>
            <td style={{ fontWeight: 600, color: '#a855f7' }}>{fmt(best.metrics.weighted_f1, 3)}</td>
            <td>{fmt(best.metrics.mse, 4)}</td>
            <td>{fmt(best.metrics.mae, 4)}</td>
            <td style={{ color: 'var(--text-2)' }}>{n}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function AllEvalsTable({ evals, selected, onToggle, onToggleAll }) {
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
          <th>Model</th><th>Match</th><th>Label</th><th>Weights (8/16/32)</th>
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
            <td style={{ fontWeight: 600 }}>{MODEL_LABELS[ev.model_type] || ev.model_type}</td>
            <td style={{ fontSize: 12 }}>{ev.match_name}</td>
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

function MergeWeightsOverviewTab() {
  const [evals, setEvals] = useState([])
  const [loading, setLoading] = useState(true)
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState('')
  const { selected, toggle, toggleAll, clear } = useRowSelection()

  const load = useCallback(async () => {
    try {
      setEvals(await listAllWeightEvals())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const handleDeleteSelected = async () => {
    if (selected.size === 0) return
    if (!window.confirm(`Delete ${selected.size} selected evaluation${selected.size === 1 ? '' : 's'}?`)) return
    setError('')
    setDeleting(true)
    try {
      await deleteWeightEvalsGlobal([...selected])
      clear()
      await load()
    } catch (err) {
      setError(err.message)
    } finally {
      setDeleting(false)
    }
  }

  if (loading) return <div className="card"><div className="empty-state"><p>Loading…</p></div></div>

  return (
    <>
      {error && <div className="error-box">{error}</div>}
      <div className="card">
        <h2>Best Weight Combination Per Model</h2>
        <BestPerModelTable evals={evals} />
      </div>

      <WeightSensitivityChart evals={evals} />

      <div className="card" style={{ padding: 0 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', padding: '16px 16px 0' }}>
          <div>
            <h2 style={{ marginBottom: 4 }}>All Evaluations</h2>
            <p style={{ fontSize: 12, color: 'var(--text-2)' }}>
              Every configuration evaluated anywhere, most recent first.
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
        <AllEvalsTable
          evals={evals} selected={selected} onToggle={toggle}
          onToggleAll={checked => toggleAll(evals.map(ev => ev.id), checked)}
        />
      </div>
    </>
  )
}

// ---------------------------------------------------------------------------
// VisualScore overview tab — best (base, flow) combo per model, aggregated
// across every match, mirroring the merge-weights tab above but for the
// training-label formula (regression-only metrics, no accuracy/F1).
// ---------------------------------------------------------------------------
function bestByR2(evals) {
  if (evals.length === 0) return null
  return evals.reduce((best, ev) => (
    best === null || ev.metrics.r2 > best.metrics.r2 ? ev : best
  ), null)
}

function bestVisualPerModel(evals) {
  const byModel = {}
  for (const ev of evals) {
    (byModel[ev.model_type] ??= []).push(ev)
  }
  return Object.entries(byModel)
    .map(([modelType, modelEvals]) => ({ modelType, best: bestByR2(modelEvals), n: modelEvals.length }))
    .sort((a, b) => (b.best?.metrics.r2 ?? -Infinity) - (a.best?.metrics.r2 ?? -Infinity))
}

function BestVisualPerModelTable({ evals }) {
  const rows = useMemo(() => bestVisualPerModel(evals), [evals])
  if (rows.length === 0) {
    return <p style={{ fontSize: 12, color: 'var(--text-2)' }}>No evaluations yet.</p>
  }
  return (
    <table>
      <thead>
        <tr>
          <th>Model</th><th>Base Weight</th><th>Flow Weight</th>
          <th>MAE</th><th>MSE</th><th>R²</th><th>Configs Tried</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(({ modelType, best, n }) => (
          <tr key={modelType}>
            <td style={{ fontWeight: 600 }}>{MODEL_LABELS[modelType] || modelType}</td>
            <td style={{ fontFamily: 'monospace', fontSize: 12 }}>{fmt(best.weights.base, 2)}</td>
            <td style={{ fontFamily: 'monospace', fontSize: 12 }}>{fmt(best.weights.flow, 2)}</td>
            <td>{fmt(best.metrics.mae, 4)}</td>
            <td>{fmt(best.metrics.mse, 4)}</td>
            <td style={{ fontWeight: 600, color: '#a855f7' }}>{fmt(best.metrics.r2, 4)}</td>
            <td style={{ color: 'var(--text-2)' }}>{n}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function AllVisualEvalsTable({ evals, selected, onToggle, onToggleAll }) {
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
          <th>Model</th><th>Match</th><th>Label</th><th>Base</th><th>Flow</th>
          <th>Tiles</th><th>MAE</th><th>MSE</th><th>R²</th><th>When</th>
        </tr>
      </thead>
      <tbody>
        {evals.map(ev => (
          <tr key={ev.id}>
            <td>
              <input type="checkbox" checked={selected.has(ev.id)} onChange={() => onToggle(ev.id)} />
            </td>
            <td style={{ fontWeight: 600 }}>{MODEL_LABELS[ev.model_type] || ev.model_type}</td>
            <td style={{ fontSize: 12 }}>{ev.match_name}</td>
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

function VisualScoreOverviewTab() {
  const [evals, setEvals] = useState([])
  const [loading, setLoading] = useState(true)
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState('')
  const { selected, toggle, toggleAll, clear } = useRowSelection()

  const load = useCallback(async () => {
    try {
      setEvals(await listAllVisualScoreEvals())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const handleDeleteSelected = async () => {
    if (selected.size === 0) return
    if (!window.confirm(`Delete ${selected.size} selected evaluation${selected.size === 1 ? '' : 's'}?`)) return
    setError('')
    setDeleting(true)
    try {
      await deleteVisualScoreEvalsGlobal([...selected])
      clear()
      await load()
    } catch (err) {
      setError(err.message)
    } finally {
      setDeleting(false)
    }
  }

  if (loading) return <div className="card"><div className="empty-state"><p>Loading…</p></div></div>

  return (
    <>
      {error && <div className="error-box">{error}</div>}
      <div className="card">
        <h2>Best VisualScore Weights Per Model</h2>
        <BestVisualPerModelTable evals={evals} />
      </div>

      <VisualScoreMetricCharts evals={evals} />

      <div className="card" style={{ padding: 0 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', padding: '16px 16px 0' }}>
          <div>
            <h2 style={{ marginBottom: 4 }}>All Evaluations</h2>
            <p style={{ fontSize: 12, color: 'var(--text-2)' }}>
              Every configuration evaluated anywhere, most recent first.
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
        <AllVisualEvalsTable
          evals={evals} selected={selected} onToggle={toggle}
          onToggleAll={checked => toggleAll(evals.map(ev => ev.id), checked)}
        />
      </div>
    </>
  )
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------
export default function CalibrationOverview() {
  const [tab, setTab] = useState('merge')

  return (
    <div>
      <Link to="/predict/admin" style={{ fontSize: 13, color: 'var(--accent)', textDecoration: 'none' }}>← Back to Predict Admin</Link>
      <h1 style={{ marginTop: 8 }}>Calibration Overview</h1>
      <p style={{ color: 'var(--text-2)', marginTop: -8 }}>
        Every weight evaluation across every match and model, combined — not scoped to one run.
      </p>

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

      {tab === 'merge' ? <MergeWeightsOverviewTab /> : <VisualScoreOverviewTab />}
    </div>
  )
}
