import { useState, useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, PointElement, LineElement,
  Title, Tooltip, Legend,
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import { BestMetricsPanel, PerClassF1Table, FeatureImportanceTable } from './MetricsDisplay.jsx'

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

export default function TrainingPage({ title, defaultEpochs, defaultBatch, api, extraControls, modelType }) {
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
  const best = metrics?.best || {}

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
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', flexWrap: 'wrap', gap: 8 }}>
        <h1>{title}</h1>
        {modelType && (
          <Link to={`/history?model=${modelType}`} style={{ fontSize: 12, color: 'var(--accent)', textDecoration: 'none', marginBottom: 20 }}>
            View full history →
          </Link>
        )}
      </div>
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
          {extraControls}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: 12, color: 'var(--text-2)' }}>
              Epoch {metrics ? `${metrics.current_epoch ?? '—'} / ${metrics.total_epochs ?? '—'}` : '—'}
            </span>
            <span className={`badge ${isRunning ? 'badge-green' : 'badge-gray'}`}>
              {isRunning ? 'Running' : (metrics?.status === 'done' ? 'Done' : 'Stopped')}
            </span>
            <button className="btn btn-primary" onClick={handleStart} disabled={isRunning}>Start</button>
            <button className="btn btn-danger"  onClick={handleStop}  disabled={!isRunning}>Stop</button>
          </div>
        </div>
      </div>

      {/* Best model performance, grouped by prediction head */}
      <BestMetricsPanel
        best={best}
        bestEpoch={metrics?.best_epoch}
        bestValLoss={metrics?.best_val_loss}
        completedAt={metrics?.completed_at}
      />

      <PerClassF1Table perClassF1={metrics?.per_class_f1} />
      <FeatureImportanceTable importance={metrics?.feature_importance} />

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
    </div>
  )
}
