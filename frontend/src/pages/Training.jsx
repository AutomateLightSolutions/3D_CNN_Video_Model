import { useState, useEffect, useRef } from 'react'
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, PointElement, LineElement,
  Title, Tooltip, Legend,
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import { startTraining, stopTraining, getTrainingStatus, getTrainingLogs } from '../api/client.js'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend)

function parseLog(lines) {
  const epochs = [], trainLoss = [], valLoss = [], valAcc = []
  for (const line of lines) {
    const m = line.match(/^EPOCH\s+(\d+)\s+TRAIN_LOSS\s+([\d.]+)\s+VAL_LOSS\s+([\d.]+)\s+VAL_ACC\s+([\d.]+)/)
    if (m) {
      epochs.push(parseInt(m[1]))
      trainLoss.push(parseFloat(m[2]))
      valLoss.push(parseFloat(m[3]))
      valAcc.push(parseFloat(m[4]))
    }
  }
  return { epochs, trainLoss, valLoss, valAcc }
}

export default function Training() {
  const [status, setStatus] = useState({ status: 'stopped' })
  const [logs, setLogs] = useState([])
  const [chartData, setChartData] = useState(null)
  const [error, setError] = useState('')
  const logRef = useRef(null)
  const pollRef = useRef(null)

  const refresh = async () => {
    try {
      const [s, l] = await Promise.all([getTrainingStatus(), getTrainingLogs()])
      setStatus(s)
      setLogs(l.lines || [])
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
              tension: 0.3,
              pointRadius: 2,
            },
            {
              label: 'Val Loss',
              data: parsed.valLoss,
              borderColor: '#f59e0b',
              backgroundColor: 'rgba(245,158,11,0.1)',
              tension: 0.3,
              pointRadius: 2,
            },
          ],
        })
      }
    } catch (e) {
      setError(e.message)
    }
  }

  useEffect(() => {
    refresh()
    pollRef.current = setInterval(refresh, 3000)
    return () => clearInterval(pollRef.current)
  }, [])

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logs])

  const handleStart = async () => {
    setError('')
    try { await startTraining(); await refresh() }
    catch (e) { setError(e.message) }
  }

  const handleStop = async () => {
    setError('')
    try { await stopTraining(); await refresh() }
    catch (e) { setError(e.message) }
  }

  const isRunning = status.status === 'running'

  return (
    <div>
      <h1>Training</h1>
      {error && <div className="error-box">{error}</div>}

      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 16 }}>
          <span className={`badge ${isRunning ? 'badge-green' : 'badge-gray'}`}>
            {isRunning ? 'Running' : 'Stopped'}
          </span>
          <button className="btn btn-primary" onClick={handleStart} disabled={isRunning}>
            Start Training
          </button>
          <button className="btn btn-danger" onClick={handleStop} disabled={!isRunning}>
            Stop
          </button>
        </div>

        <div className="metric-cards">
          <div className="metric-card">
            <div className="label">Epoch</div>
            <div className="value">{status.current_epoch ?? '—'}</div>
          </div>
          <div className="metric-card">
            <div className="label">Train Loss</div>
            <div className="value">{status.train_loss != null ? status.train_loss.toFixed(4) : '—'}</div>
          </div>
          <div className="metric-card">
            <div className="label">Val Loss</div>
            <div className="value">{status.val_loss != null ? status.val_loss.toFixed(4) : '—'}</div>
          </div>
          <div className="metric-card">
            <div className="label">Val Acc</div>
            <div className="value" style={{ color: 'var(--green)' }}>
              {status.val_acc != null ? (status.val_acc * 100).toFixed(1) + '%' : '—'}
            </div>
          </div>
        </div>
      </div>

      <div className="training-layout">
        <div className="card">
          <h2>Loss Curve</h2>
          {chartData ? (
            <Line
              data={chartData}
              options={{
                responsive: true,
                animation: false,
                plugins: {
                  legend: { labels: { color: '#b0b0b0', font: { size: 11 } } },
                },
                scales: {
                  x: {
                    title: { display: true, text: 'Epoch', color: '#707070' },
                    ticks: { color: '#707070' },
                    grid: { color: '#2a2a2a' },
                  },
                  y: {
                    title: { display: true, text: 'Loss', color: '#707070' },
                    ticks: { color: '#707070' },
                    grid: { color: '#2a2a2a' },
                  },
                },
              }}
            />
          ) : (
            <div className="empty-state"><p>No training data yet.</p></div>
          )}
        </div>

        <div className="card">
          <h2>Training Log</h2>
          <div className="log-viewer" ref={logRef}>
            {logs.length === 0 ? (
              <span style={{ color: 'var(--text-2)' }}>No log output yet.</span>
            ) : (
              logs.map((line, i) => (
                <div key={i} className="log-line">{line}</div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
