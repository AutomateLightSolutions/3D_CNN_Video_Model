// Shared building blocks for the merge-weight calibration pages — the
// per-run page (MergeCalibration.jsx) and the cross-match overview
// (CalibrationOverview.jsx) both render the same F1-vs-weight scatter
// panels and the same "best by F1" logic; kept here once instead of twice.
import { useMemo } from 'react'
import {
  Chart as ChartJS,
  LinearScale, PointElement,
  Title, Tooltip, Legend,
} from 'chart.js'
import { Scatter } from 'react-chartjs-2'
import { WINDOW_SIZES } from '../constants.js'
import { fmt } from './MetricsDisplay.jsx'

ChartJS.register(LinearScale, PointElement, Title, Tooltip, Legend)

export function bestByF1(evals) {
  if (evals.length === 0) return null
  return evals.reduce((best, ev) => (
    best === null || ev.metrics.weighted_f1 > best.metrics.weighted_f1 ? ev : best
  ), null)
}

export function relTime(iso) {
  const ms = Date.now() - new Date(iso).getTime()
  const h = Math.floor(ms / 3600000)
  if (h < 1) return 'just now'
  if (h < 24) return `${h}h ago`
  return `${Math.floor(h / 24)}d ago`
}

const WINDOW_COLORS = { 8: '#3b82f6', 16: '#f59e0b', 32: '#a855f7' }

function scatterPointsForWindow(evals, ws) {
  return evals
    .map(ev => {
      const raw = ev.weights[ws] ?? ev.weights[String(ws)]
      if (raw == null) return null
      return {
        x: Number(raw), y: ev.metrics.weighted_f1,
        label: ev.label, weights: ev.weights, createdAt: ev.created_at, evId: ev.id,
      }
    })
    .filter(Boolean)
}

function scatterOptionsFor(ws) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          title: (items) => items[0]?.raw?.label || 'Config',
          label: (item) => {
            const r = item.raw
            return [
              `weights: 8s=${fmt(r.weights[8] ?? r.weights['8'], 2)}, `
              + `16s=${fmt(r.weights[16] ?? r.weights['16'], 2)}, `
              + `32s=${fmt(r.weights[32] ?? r.weights['32'], 2)}`,
              `F1 score: ${r.y.toFixed(4)}`,
              relTime(r.createdAt),
            ]
          },
        },
      },
    },
    scales: {
      x: {
        title: { display: true, text: `${ws}s weight share`, color: '#707070', font: { size: 10 } },
        ticks: { color: '#707070', font: { size: 9 } },
        grid: { color: '#2a2a2a' },
        min: 0, max: 1,
      },
      y: {
        title: { display: true, text: 'F1 score', color: '#707070', font: { size: 10 } },
        ticks: { color: '#707070', font: { size: 9 } },
        grid: { color: '#2a2a2a' },
        min: 0, max: 1,
      },
    },
  }
}

function WindowScatterPanel({ ws, evals, best }) {
  const chartData = useMemo(() => {
    const points = scatterPointsForWindow(evals, ws)
    const bestRaw = best ? (best.weights[ws] ?? best.weights[String(ws)]) : null
    const bestPoint = best && bestRaw != null
      ? [{ x: Number(bestRaw), y: best.metrics.weighted_f1, label: best.label, weights: best.weights, createdAt: best.created_at, evId: best.id }]
      : []
    return {
      datasets: [
        { label: 'Configs', data: points, backgroundColor: WINDOW_COLORS[ws], pointRadius: 3, pointHoverRadius: 5 },
        {
          label: 'Best', data: bestPoint, pointRadius: 6, pointHoverRadius: 7,
          borderColor: '#22c55e', borderWidth: 2, backgroundColor: 'rgba(34,197,94,0.15)',
        },
      ],
    }
  }, [evals, ws, best])

  const options = useMemo(() => scatterOptionsFor(ws), [ws])

  return (
    <div className="card" style={{ flex: 1, minWidth: 0 }}>
      <h2 style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: 0.5, color: 'var(--text-2)' }}>
        {ws}s Window Weight Share
      </h2>
      <div style={{ height: 160 }}>
        <Scatter data={chartData} options={options} />
      </div>
    </div>
  )
}

// evals: array of { weights, metrics, label, created_at, ... } — works for
// both a single run's evals and the full cross-match list.
export function WeightSensitivityChart({ evals }) {
  const best = useMemo(() => bestByF1(evals), [evals])
  if (evals.length === 0) return null
  return (
    <div style={{ display: 'flex', flexDirection: 'row', gap: 16, flexWrap: 'wrap' }}>
      {WINDOW_SIZES.map(ws => <WindowScatterPanel key={ws} ws={ws} evals={evals} best={best} />)}
    </div>
  )
}
