import { useState, useEffect, useRef } from 'react'
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, PointElement, LineElement,
  Title, Tooltip, Legend,
} from 'chart.js'
import { Line } from 'react-chartjs-2'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend)

const POLL_MS = 3000

function parseLog(lines) {
  const epochs = [], trainLoss = [], valLoss = [], valAcc = [], macroF1 = []
  for (const line of lines) {
    const m = line.match(
      /^EPOCH\s+(\d+)\s+TRAIN_LOSS\s+([\d.]+)\s+VAL_LOSS\s+([\d.]+)\s+VAL_ACC\s+([\d.]+)(?:\s+MACRO_F1\s+([\d.]+))?/
    )
    if (m) {
      epochs.push(parseInt(m[1]))
      trainLoss.push(parseFloat(m[2]))
      valLoss.push(parseFloat(m[3]))
      valAcc.push(parseFloat(m[4]))
      macroF1.push(m[5] ? parseFloat(m[5]) : null)
    }
  }
  return { epochs, trainLoss, valLoss, valAcc, macroF1 }
}

function MetricCard({ label, value, color }) {
  return (
    <div className="metric-card">
      <div className="label">{label}</div>
      <div className="value" style={color ? { color } : {}}>
        {value ?? '—'}
      </div>
    </div>
  )
}

function PerClassF1Table({ perClassF1 }) {
  if (!perClassF1 || Object.keys(perClassF1).length === 0) return null
  const entries = Object.entries(perClassF1).sort((a, b) => b[1] - a[1])
  return (
    <div className="card">
      <h2>Per-Class F1 — Best Model</h2>
      <table className="score-guide-table">
        <thead>
          <tr><th>Event</th><th>F1 Score</th><th>Bar</th></tr>
        </thead>
        <tbody>
          {entries.map(([cls, f1]) => (
            <tr key={cls}>
              <td style={{ textTransform: 'capitalize' }}>{cls.replace(/_/g, ' ')}</td>
              <td>{f1.toFixed(3)}</td>
              <td style={{ width: 120 }}>
                <div style={{
                  height: 8, borderRadius: 4,
                  background: `linear-gradient(to right, var(--green) ${(f1 * 100).toFixed(0)}%, var(--surface-2) ${(f1 * 100).toFixed(0)}%)`,
                }} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function TrainingPage({ title, defaultEpochs, defaultBatch, api }) {
  const [epochs,    setEpochs]    = useState(defaultEpochs)
  const [batchSize, setBatchSize] = useState(defaultBatch)
  const [lr,        setLr]        = useState('1e-3')
  const [device,    setDevice]    = useState('cuda')

  const [status,    setStatus]    = useState({ status: 'stopped' })
  const [logs,      setLogs]      = useState([])
  const [metrics,   setMetrics]   = useState(null)
  const [chartData, setChartData] = useState(null)
  const [error,     setError]     = useState('')

  const logRef  = useRef(null)
  const pollRef = useRef(null)

  const refresh = async () => {
    try {
      const [s, l, m] = await Promise.all([api.status(), api.logs(), api.metrics()])
      setStatus(s)
      setLogs(l.lines || [])
      if (m && Object.keys(m).length > 0) setMetrics(m)

      const parsed = parseLog(l.lines || [])
      if (parsed.epochs.length > 0) {
        setChartData({
          labels: parsed.epochs,
          datasets: [
            {
              label: 'Train Loss',
              data: parsed.trainLoss,
              borderColor: '#3b82f6',
              backgroundColor: 'rgba(59,130,246,0.1)',
              tension: 0.3, pointRadius: 2,
            },
            {
              label: 'Val Loss',
              data: parsed.valLoss,
              borderColor: '#f59e0b',
              backgroundColor: 'rgba(245,158,11,0.1)',
              tension: 0.3, pointRadius: 2,
            },
            {
              label: 'Val Acc',
              data: parsed.valAcc,
              borderColor: '#22c55e',
              backgroundColor: 'rgba(34,197,94,0.1)',
              tension: 0.3, pointRadius: 2,
              yAxisID: 'y2',
            },
            ...(parsed.macroF1.some(v => v !== null) ? [{
              label: 'Macro F1',
              data: parsed.macroF1,
              borderColor: '#a855f7',
              backgroundColor: 'rgba(168,85,247,0.1)',
              tension: 0.3, pointRadius: 2,
              yAxisID: 'y2',
            }] : []),
          ],
        })
      }
    } catch (e) {
      setError(e.message)
    }
  }

  useEffect(() => {
    refresh()
    pollRef.current = setInterval(refresh, POLL_MS)
    return () => clearInterval(pollRef.current)
  }, [])

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logs])

  const handleStart = async () => {
    setError('')
    try {
      await api.start({ epochs: parseInt(epochs), batch_size: parseInt(batchSize), lr: parseFloat(lr), device })
      await refresh()
    } catch (e) { setError(e.message) }
  }

  const handleStop = async () => {
    setError('')
    try { await api.stop(); await refresh() }
    catch (e) { setError(e.message) }
  }

  const isRunning = status.status === 'running'
  const cur  = metrics?.current  || {}
  const best = metrics?.best     || {}

  const fmt  = (v, digits = 4) => v != null ? Number(v).toFixed(digits) : '—'
  const fmtP = (v)             => v != null ? (Number(v) * 100).toFixed(1) + '%' : '—'

  const chartOptions = {
    responsive: true,
    animation: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { labels: { color: '#b0b0b0', font: { size: 11 } } },
    },
    scales: {
      x: {
        title: { display: true, text: 'Epoch', color: '#707070' },
        ticks: { color: '#707070' },
        grid:  { color: '#2a2a2a' },
      },
      y: {
        title: { display: true, text: 'Loss', color: '#707070' },
        ticks: { color: '#707070' },
        grid:  { color: '#2a2a2a' },
      },
      y2: {
        position: 'right',
        title: { display: true, text: 'Score', color: '#707070' },
        ticks: { color: '#707070' },
        grid:  { drawOnChartArea: false },
        min: 0, max: 1,
      },
    },
  }

  return (
    <div>
      <h1>{title}</h1>
      {error && <div className="error-box">{error}</div>}

      {/* Controls + live metrics */}
      <div className="card">
        <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'flex-end', marginBottom: 20 }}>
          <div>
            <label style={{ display: 'block', fontSize: 11, color: 'var(--text-2)', marginBottom: 4 }}>Epochs</label>
            <input
              type="number" value={epochs} min={1} max={200}
              onChange={e => setEpochs(e.target.value)}
              disabled={isRunning}
              style={{ width: 80, padding: '4px 8px', borderRadius: 6, border: '1px solid var(--border)', background: 'var(--surface-2)', color: 'var(--text-1)' }}
            />
          </div>
          <div>
            <label style={{ display: 'block', fontSize: 11, color: 'var(--text-2)', marginBottom: 4 }}>Batch Size</label>
            <input
              type="number" value={batchSize} min={1} max={32}
              onChange={e => setBatchSize(e.target.value)}
              disabled={isRunning}
              style={{ width: 80, padding: '4px 8px', borderRadius: 6, border: '1px solid var(--border)', background: 'var(--surface-2)', color: 'var(--text-1)' }}
            />
          </div>
          <div>
            <label style={{ display: 'block', fontSize: 11, color: 'var(--text-2)', marginBottom: 4 }}>Learning Rate</label>
            <select
              value={lr} onChange={e => setLr(e.target.value)}
              disabled={isRunning}
              style={{ padding: '4px 8px', borderRadius: 6, border: '1px solid var(--border)', background: 'var(--surface-2)', color: 'var(--text-1)' }}
            >
              {['1e-2','1e-3','1e-4','1e-5'].map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>
          <div>
            <label style={{ display: 'block', fontSize: 11, color: 'var(--text-2)', marginBottom: 4 }}>Device</label>
            <select
              value={device} onChange={e => setDevice(e.target.value)}
              disabled={isRunning}
              style={{ padding: '4px 8px', borderRadius: 6, border: '1px solid var(--border)', background: 'var(--surface-2)', color: 'var(--text-1)' }}
            >
              <option value="cuda">CUDA (GPU)</option>
              <option value="cpu">CPU</option>
            </select>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span className={`badge ${isRunning ? 'badge-green' : 'badge-gray'}`}>
              {isRunning ? 'Running' : (metrics?.status === 'done' ? 'Done' : 'Stopped')}
            </span>
            <button className="btn btn-primary" onClick={handleStart} disabled={isRunning}>Start</button>
            <button className="btn btn-danger"  onClick={handleStop}  disabled={!isRunning}>Stop</button>
          </div>
        </div>

        {/* Current epoch metrics */}
        <div className="metric-cards">
          <MetricCard label="Epoch"       value={metrics ? `${metrics.current_epoch ?? '—'} / ${metrics.total_epochs ?? '—'}` : '—'} />
          <MetricCard label="Train Loss"  value={fmt(cur.train_loss)} />
          <MetricCard label="Val Loss"    value={fmt(cur.val_loss)} />
          <MetricCard label="Val Acc"     value={fmtP(cur.val_accuracy)} color="var(--green)" />
          <MetricCard label="Macro F1"    value={fmt(cur.macro_f1, 3)} color="#a855f7" />
          <MetricCard label="MAE"         value={fmt(cur.mae, 4)} />
          <MetricCard label="R²"          value={fmt(cur.r2, 4)} />
        </div>
      </div>

      {/* Best model summary */}
      {metrics?.best_epoch > 0 && (
        <div className="card" style={{ borderLeft: '3px solid var(--green)' }}>
          <h2 style={{ marginBottom: 12 }}>Best Model — Epoch {metrics.best_epoch}</h2>
          <div className="metric-cards">
            <MetricCard label="Val Loss"    value={fmt(metrics.best_val_loss)} />
            <MetricCard label="Val Acc"     value={fmtP(best.val_accuracy)} color="var(--green)" />
            <MetricCard label="Macro F1"    value={fmt(best.macro_f1, 3)}    color="#a855f7" />
            <MetricCard label="Weighted F1" value={fmt(best.weighted_f1, 3)} />
            <MetricCard label="MAE"         value={fmt(best.mae, 4)} />
            <MetricCard label="R²"          value={fmt(best.r2, 4)} />
            <MetricCard label="Pearson"     value={fmt(best.pearson, 4)} />
          </div>
          {metrics.completed_at && (
            <p style={{ fontSize: 11, color: 'var(--text-2)', marginTop: 8 }}>
              Completed: {new Date(metrics.completed_at).toLocaleString()}
            </p>
          )}
        </div>
      )}

      {/* Charts + log */}
      <div className="training-layout">
        <div className="card">
          <h2>Training Curves</h2>
          {chartData ? (
            <Line data={chartData} options={chartOptions} />
          ) : (
            <div className="empty-state"><p>No training data yet.</p></div>
          )}
        </div>
        <div className="card">
          <h2>Training Log</h2>
          <div className="log-viewer" ref={logRef}>
            {logs.length === 0
              ? <span style={{ color: 'var(--text-2)' }}>No log output yet.</span>
              : logs.map((line, i) => <div key={i} className="log-line">{line}</div>)
            }
          </div>
        </div>
      </div>

      <PerClassF1Table perClassF1={metrics?.per_class_f1} />
    </div>
  )
}
