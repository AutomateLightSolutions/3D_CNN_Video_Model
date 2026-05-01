import { useState, useEffect, useRef } from 'react'
import {
  createMatch, listMatches, extractClips, deleteMatch,
  getExtractionProgress, getExtractionLog,
} from '../api/client.js'

function statusBadge(status) {
  const map = {
    registered: 'badge-gray',
    extracting:  'badge-amber',
    ready:       'badge-green',
    error:       'badge-red',
  }
  return <span className={`badge ${map[status] || 'badge-gray'}`}>{status}</span>
}

// ---------------------------------------------------------------------------
// Live extraction panel — shown as a full-width card while extracting
// ---------------------------------------------------------------------------
function ExtractionPanel({ match, onDone }) {
  const [progress, setProgress]   = useState({ clips_total: 0, clips_done: 0 })
  const [logLines, setLogLines]   = useState([])
  const [elapsed, setElapsed]     = useState(0)
  const [matchStatus, setMatchStatus] = useState(match.status)
  const logRef    = useRef(null)
  const startRef  = useRef(Date.now())
  const pollRef   = useRef(null)
  const timerRef  = useRef(null)

  useEffect(() => {
    timerRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startRef.current) / 1000))
    }, 1000)

    const poll = async () => {
      try {
        const [p, l, matches] = await Promise.all([
          getExtractionProgress(match.id),
          getExtractionLog(match.id),
          listMatches(),
        ])
        setProgress(p)
        setLogLines(l.lines || [])

        const current = matches.find(m => m.id === match.id)
        if (current) setMatchStatus(current.status)

        if (current && current.status !== 'extracting') {
          clearInterval(pollRef.current)
          clearInterval(timerRef.current)
          setTimeout(onDone, 800)   // small delay so the user sees final state
        }
      } catch {}
    }

    poll()
    pollRef.current = setInterval(poll, 2000)

    return () => {
      clearInterval(pollRef.current)
      clearInterval(timerRef.current)
    }
  }, [match.id])

  // Auto-scroll log to bottom
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logLines])

  const { clips_total: total, clips_done: done } = progress
  const pct = total > 0 ? Math.round((done / total) * 100) : 0
  const isError = matchStatus === 'error'
  const isDone  = matchStatus === 'ready'
  const fmtTime = (s) => `${Math.floor(s / 60)}m ${s % 60}s`

  return (
    <div className="extraction-panel" style={{ borderColor: isError ? 'var(--red)' : isDone ? 'var(--green)' : 'var(--amber)' }}>
      {/* Header row */}
      <div className="extraction-panel-header">
        <span className="extraction-title">
          {isError ? '⚠ Extraction failed' : isDone ? '✓ Extraction complete' : '⟳ Extracting clips…'}
          <span style={{ marginLeft: 8, fontWeight: 400, color: 'var(--text-2)', fontSize: 12 }}>{match.name}</span>
        </span>
        <span style={{ fontSize: 12, color: 'var(--text-2)' }}>
          {fmtTime(elapsed)} elapsed
        </span>
      </div>

      {/* Progress bar */}
      <div style={{ margin: '10px 0 6px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--text-1)', marginBottom: 4 }}>
          <span>{total > 0 ? `${done} / ${total} clips` : 'Scanning video…'}</span>
          <span style={{ fontWeight: 600 }}>{total > 0 ? `${pct}%` : ''}</span>
        </div>
        <div className="progress-bar-wrap" style={{ height: 10 }}>
          <div
            className="progress-bar-fill"
            style={{
              width: `${pct}%`,
              background: isError ? 'var(--red)' : isDone ? 'var(--green)' : 'var(--amber)',
            }}
          />
        </div>
      </div>

      {/* Log viewer */}
      <div className="extraction-log" ref={logRef}>
        {logLines.length === 0
          ? <span style={{ color: 'var(--text-2)' }}>Waiting for output…</span>
          : logLines.map((line, i) => (
            <div
              key={i}
              className="log-line"
              style={{
                color: line.includes('ERROR') || line.includes('FATAL') ? 'var(--red)'
                  : line.includes('WARN') ? 'var(--amber)'
                  : line.includes('Done.') ? 'var(--green)'
                  : undefined,
              }}
            >{line}</div>
          ))
        }
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------
export default function MatchList() {
  const [matches,  setMatches]  = useState([])
  const [name,     setName]     = useState('')
  const [filePath, setFilePath] = useState('')
  const [error,    setError]    = useState('')
  const [loading,  setLoading]  = useState(false)
  const [extractingIds, setExtractingIds] = useState(new Set())

  const load = async () => {
    try {
      const ms = await listMatches()
      setMatches(ms)
      // keep extractingIds in sync: remove matches that are no longer extracting
      setExtractingIds(prev => {
        const next = new Set(prev)
        for (const id of prev) {
          const m = ms.find(x => x.id === id)
          if (!m || m.status !== 'extracting') next.delete(id)
        }
        return next
      })
    } catch (e) {
      setError(e.message)
    }
  }

  useEffect(() => { load() }, [])

  const handleCreate = async (e) => {
    e.preventDefault()
    if (!name.trim() || !filePath.trim()) { setError('Name and file path are required'); return }
    setError('')
    setLoading(true)
    try {
      await createMatch(name.trim(), filePath.trim())
      setName('')
      setFilePath('')
      await load()
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleExtract = async (id) => {
    setError('')
    try {
      await extractClips(id)
      setExtractingIds(prev => new Set(prev).add(id))
      await load()
    } catch (e) {
      setError(e.message)
    }
  }

  const handleDelete = async (id, name) => {
    if (!window.confirm(`Delete match "${name}" and all its clips and labels?`)) return
    setError('')
    try {
      await deleteMatch(id)
      await load()
    } catch (e) {
      setError(e.message)
    }
  }

  const handleDone = async (id) => {
    setExtractingIds(prev => { const s = new Set(prev); s.delete(id); return s })
    await load()
  }

  const activeExtractions = matches.filter(
    m => m.status === 'extracting' || extractingIds.has(m.id)
  )

  return (
    <div>
      <h1>Matches</h1>

      {error && <div className="error-box">{error}</div>}

      {/* Live extraction panels */}
      {activeExtractions.map(m => (
        <ExtractionPanel key={m.id} match={m} onDone={() => handleDone(m.id)} />
      ))}

      {/* Register form */}
      <div className="card">
        <h2>Register Match</h2>
        <form onSubmit={handleCreate}>
          <div className="form-row">
            <div className="form-group">
              <label>Match Name</label>
              <input
                type="text"
                value={name}
                onChange={e => setName(e.target.value)}
                placeholder="Ireland_vs_SouthAfrica_2024"
              />
            </div>
            <div className="form-group" style={{ flex: 1 }}>
              <label>Video File Path (absolute)</label>
              <input
                type="text"
                value={filePath}
                onChange={e => setFilePath(e.target.value)}
                placeholder="C:\Users\ASUS\Downloads\match.mp4"
                style={{ minWidth: 320 }}
              />
            </div>
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? 'Registering…' : 'Register'}
            </button>
          </div>
        </form>
      </div>

      {/* Matches table */}
      <div className="card">
        <h2>All Matches</h2>
        {matches.length === 0 ? (
          <div className="empty-state"><p>No matches registered yet.</p></div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Status</th>
                <th>Duration</th>
                <th>Clips</th>
                <th>Labeled</th>
                <th>Action</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {matches.map(m => (
                <tr key={m.id}>
                  <td style={{ fontWeight: 500 }}>{m.name}</td>
                  <td>{statusBadge(m.status)}</td>
                  <td style={{ color: 'var(--text-1)' }}>
                    {m.duration_seconds != null ? `${m.duration_seconds.toFixed(1)}s` : '—'}
                  </td>
                  <td>{m.clip_count}</td>
                  <td>
                    {m.clip_count > 0
                      ? <span style={{ color: m.labeled_count === m.clip_count ? 'var(--green)' : 'var(--text-1)' }}>
                          {m.labeled_count} / {m.clip_count}
                        </span>
                      : '—'}
                  </td>
                  <td>
                    <button
                      className="btn btn-secondary"
                      disabled={m.status === 'extracting'}
                      onClick={() => handleExtract(m.id)}
                    >
                      {m.status === 'extracting' ? 'Extracting…'
                        : m.status === 'ready'    ? 'Re-extract'
                        : 'Extract Clips'}
                    </button>
                  </td>
                  <td>
                    <button
                      className="btn btn-danger"
                      disabled={m.status === 'extracting'}
                      onClick={() => handleDelete(m.id, m.name)}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
