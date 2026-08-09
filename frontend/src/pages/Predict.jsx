import { useState, useEffect, useCallback, useRef } from 'react'
import {
  listMatches, getAvailableModels, startPrediction, listPredictionRuns,
  getPredictionProgress, getPredictionLog, getPredictionSegments, deletePredictionRun,
} from '../api/client.js'
import { MODEL_LABELS, PREDICTION_TABS } from '../constants.js'
import { StatusBadge, fmt } from '../components/MetricsDisplay.jsx'

const POLL_MS = 3000

function formatDate(iso) {
  return iso ? new Date(iso).toLocaleString() : '—'
}

function fmtTime(s) {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60).toString().padStart(2, '0')
  return `${m}:${sec}`
}

// ---------------------------------------------------------------------------
// One tile's expanded detail: the 3 raw per-window predictions + clip preview
// ---------------------------------------------------------------------------
function SegmentDetail({ segment }) {
  return (
    <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', padding: '12px 0' }}>
      {segment.clip_url && (
        <video controls src={segment.clip_url} style={{ width: 280, borderRadius: 6, background: '#000' }} />
      )}
      <table className="score-guide-table" style={{ flex: 1, minWidth: 280 }}>
        <thead>
          <tr><th>Window</th><th>Event</th><th>Confidence</th><th>Score</th></tr>
        </thead>
        <tbody>
          {[...segment.windows].sort((a, b) => a.window_size - b.window_size).map(w => (
            <tr key={w.window_size}>
              <td>{w.window_size}s</td>
              <td style={{ textTransform: 'capitalize' }}>{w.event_class.replace(/_/g, ' ')}</td>
              <td>{(w.confidence * 100).toFixed(1)}%</td>
              <td>{fmt(w.highlight_score, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Results table — merged timeline, one row per tile
// ---------------------------------------------------------------------------
function ResultsTable({ segments }) {
  const [expandedTile, setExpandedTile] = useState(null)
  if (segments.length === 0) {
    return <div className="empty-state"><p>No tiles produced for this run.</p></div>
  }
  return (
    <table>
      <thead>
        <tr><th>Tile</th><th>Time Range</th><th>Event</th><th>Highlight Score</th></tr>
      </thead>
      <tbody>
        {segments.map(seg => (
          <>
            <tr
              key={seg.id}
              style={{ cursor: 'pointer' }}
              onClick={() => setExpandedTile(expandedTile === seg.id ? null : seg.id)}
            >
              <td>#{seg.tile_index}</td>
              <td>{fmtTime(seg.global_start_time)} – {fmtTime(seg.global_end_time)}</td>
              <td style={{ textTransform: 'capitalize', fontWeight: 500 }}>
                {seg.predicted_event.replace(/_/g, ' ')}
              </td>
              <td style={{ width: 160 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div style={{
                    height: 8, borderRadius: 4, flex: 1,
                    background: `linear-gradient(to right, var(--green) ${(seg.highlight_score * 100).toFixed(0)}%, var(--surface-2) ${(seg.highlight_score * 100).toFixed(0)}%)`,
                  }} />
                  <span style={{ fontSize: 12 }}>{fmt(seg.highlight_score, 2)}</span>
                </div>
              </td>
            </tr>
            {expandedTile === seg.id && (
              <tr key={`${seg.id}-detail`}>
                <td colSpan={4} style={{ background: 'var(--bg-1)', padding: '0 16px' }}>
                  <SegmentDetail segment={seg} />
                </td>
              </tr>
            )}
          </>
        ))}
      </tbody>
    </table>
  )
}

// ---------------------------------------------------------------------------
// Live progress + results for one run (polls while running)
// ---------------------------------------------------------------------------
function RunPanel({ run, onStatusChange }) {
  const [progress, setProgress] = useState({ tiles_total: 0, tiles_done: 0, status: run.status })
  const [logLines, setLogLines] = useState([])
  const [segments, setSegments] = useState([])
  const logRef = useRef(null)
  const pollRef = useRef(null)

  const poll = useCallback(async () => {
    try {
      const [p, l] = await Promise.all([getPredictionProgress(run.id), getPredictionLog(run.id)])
      setProgress(p)
      setLogLines(l.lines || [])
      if (p.status !== 'running') {
        const segs = await getPredictionSegments(run.id)
        setSegments(segs)
        clearInterval(pollRef.current)
        onStatusChange?.()
      }
    } catch {}
  }, [run.id, onStatusChange])

  useEffect(() => {
    poll()
    if (run.status === 'running') {
      pollRef.current = setInterval(poll, POLL_MS)
      return () => clearInterval(pollRef.current)
    }
    getPredictionSegments(run.id).then(setSegments).catch(() => {})
  }, [poll, run.status])

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logLines])

  const { tiles_total: total, tiles_done: done, status } = progress
  const pct = total > 0 ? Math.round((done / total) * 100) : 0

  return (
    <div>
      {status === 'running' && (
        <div className="extraction-panel" style={{ borderColor: 'var(--amber)' }}>
          <div className="extraction-panel-header">
            <span className="extraction-title">⟳ Predicting…</span>
            <span style={{ fontSize: 12, color: 'var(--text-2)' }}>
              {total > 0 ? `${done} / ${total} tiles` : 'Reading video…'}
            </span>
          </div>
          <div style={{ margin: '10px 0 6px' }}>
            <div className="progress-bar-wrap" style={{ height: 10 }}>
              <div className="progress-bar-fill" style={{ width: `${pct}%`, background: 'var(--amber)' }} />
            </div>
          </div>
          <div className="extraction-log" ref={logRef}>
            {logLines.length === 0
              ? <span style={{ color: 'var(--text-2)' }}>Waiting for output…</span>
              : logLines.map((line, i) => <div key={i} className="log-line">{line}</div>)}
          </div>
        </div>
      )}

      {status === 'error' && (
        <div className="error-box">Prediction run failed: {run.error_message || 'unknown error'}</div>
      )}

      {status !== 'running' && segments.length > 0 && (
        <div className="card" style={{ padding: 0 }}>
          <ResultsTable segments={segments} />
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------
export default function Predict() {
  const [matches, setMatches] = useState([])
  const [matchId, setMatchId] = useState('')
  const [tab, setTab] = useState('deep')
  const [modelByTab, setModelByTab] = useState({ deep: 'r3d', interpretable: 'rf' })
  const [device, setDevice] = useState('cuda')
  const [available, setAvailable] = useState({ available: {}, hybrid_backbone: null })
  const [runs, setRuns] = useState([])
  const [expandedRunId, setExpandedRunId] = useState(null)
  const [error, setError] = useState('')
  const [starting, setStarting] = useState(false)

  useEffect(() => {
    listMatches().then(ms => {
      setMatches(ms)
      if (ms.length > 0) setMatchId(String(ms[0].id))
    }).catch(e => setError(e.message))
    getAvailableModels().then(setAvailable).catch(() => {})
  }, [])

  const loadRuns = useCallback(async () => {
    if (!matchId) { setRuns([]); return }
    try {
      setRuns(await listPredictionRuns(Number(matchId)))
    } catch (e) {
      setError(e.message)
    }
  }, [matchId])

  useEffect(() => { loadRuns() }, [loadRuns])

  const model = modelByTab[tab]
  const modelReady = available.available?.[model]
  const anyRunning = runs.some(r => r.status === 'running')

  const handleStart = async () => {
    if (!matchId) { setError('Select a match first'); return }
    setError('')
    setStarting(true)
    try {
      const run = await startPrediction({ match_id: Number(matchId), model_type: model, device })
      await loadRuns()
      setExpandedRunId(run.id)
    } catch (e) {
      setError(e.message)
    } finally {
      setStarting(false)
    }
  }

  const handleDelete = async (run) => {
    const label = MODEL_LABELS[run.model_type] || run.model_type
    if (!window.confirm(`Delete this ${label} prediction run? This removes its results and extracted clips.`)) return
    try {
      await deletePredictionRun(run.id)
      await loadRuns()
    } catch (e) {
      setError(e.message)
    }
  }

  return (
    <div>
      <h1>Predict Match</h1>
      {error && <div className="error-box">{error}</div>}

      <div className="card">
        <h2>Run a Trained Model on a Full Match</h2>

        <div className="form-row">
          <div className="form-group" style={{ flex: 1 }}>
            <label>Match</label>
            <select value={matchId} onChange={e => setMatchId(e.target.value)}>
              {matches.length === 0 && <option value="">No matches registered</option>}
              {matches.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label>Device</label>
            <select value={device} onChange={e => setDevice(e.target.value)}>
              <option value="cuda">cuda</option>
              <option value="cpu">cpu</option>
            </select>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8, margin: '12px 0' }}>
          {Object.entries(PREDICTION_TABS).map(([key, cfg]) => (
            <button
              key={key}
              type="button"
              className={`btn ${tab === key ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setTab(key)}
            >
              {cfg.label}
            </button>
          ))}
        </div>

        <div className="form-row">
          <div className="form-group" style={{ flex: 1 }}>
            <label>Model</label>
            <select
              value={modelByTab[tab]}
              onChange={e => setModelByTab(prev => ({ ...prev, [tab]: e.target.value }))}
            >
              {PREDICTION_TABS[tab].models.map(m => (
                <option key={m} value={m} disabled={available.available?.[m] === false}>
                  {MODEL_LABELS[m]}{available.available?.[m] === false ? ' (not trained yet)' : ''}
                </option>
              ))}
            </select>
          </div>
          <button
            className="btn btn-primary"
            disabled={starting || !matchId || !modelReady || anyRunning}
            onClick={handleStart}
          >
            {starting ? 'Starting…' : anyRunning ? 'Run in progress…' : 'Start Prediction'}
          </button>
        </div>

        {tab === 'interpretable' && (
          <p style={{ fontSize: 12, color: 'var(--text-2)' }}>
            Interpretable models run extra per-clip analysis (player detection, pose, optical flow),
            so this tab takes noticeably longer than Deep Learning on the same match length.
          </p>
        )}
        {model === 'hybrid' && available.hybrid_backbone && (
          <p style={{ fontSize: 12, color: 'var(--text-2)' }}>
            Hybrid will use the <strong>{available.hybrid_backbone}</strong> backbone — the one its
            last completed training run used.
          </p>
        )}
      </div>

      <div className="card" style={{ padding: 0 }}>
        {runs.length === 0 ? (
          <div className="empty-state"><p>No prediction runs for this match yet.</p></div>
        ) : (
          <table>
            <thead>
              <tr><th>Model</th><th>Status</th><th>Started</th><th>Completed</th><th></th></tr>
            </thead>
            <tbody>
              {runs.map(run => (
                <>
                  <tr
                    key={run.id}
                    style={{ cursor: 'pointer' }}
                    onClick={() => setExpandedRunId(expandedRunId === run.id ? null : run.id)}
                  >
                    <td style={{ fontWeight: 600 }}>{MODEL_LABELS[run.model_type] || run.model_type}</td>
                    <td><StatusBadge status={run.status} /></td>
                    <td style={{ fontSize: 12, color: 'var(--text-1)' }}>{formatDate(run.started_at)}</td>
                    <td style={{ fontSize: 12, color: 'var(--text-1)' }}>{formatDate(run.completed_at)}</td>
                    <td onClick={e => e.stopPropagation()} style={{ display: 'flex', gap: 8 }}>
                      {run.status === 'completed' && (
                        <a
                          className="btn btn-secondary" style={{ padding: '4px 10px' }}
                          href={`/predict/${run.id}/calibrate`} target="_blank" rel="noopener noreferrer"
                        >
                          Admin
                        </a>
                      )}
                      <button className="btn btn-danger" style={{ padding: '4px 10px' }} onClick={() => handleDelete(run)}>
                        Delete
                      </button>
                    </td>
                  </tr>
                  {expandedRunId === run.id && (
                    <tr key={`${run.id}-detail`}>
                      <td colSpan={5} style={{ background: 'var(--bg-1)', padding: 16 }}>
                        <RunPanel run={run} onStatusChange={loadRuns} />
                      </td>
                    </tr>
                  )}
                </>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
