// Shared building blocks for the merge-weight calibration pages — the
// per-run page (MergeCalibration.jsx) and the cross-match overview
// (CalibrationOverview.jsx) both render the same F1-vs-weight scatter
// panels and the same "best by F1" logic; kept here once instead of twice.
import { useMemo, useState } from 'react'
import {
  Chart as ChartJS,
  LinearScale, PointElement, LineElement,
  Title, Tooltip, Legend,
} from 'chart.js'
import { Scatter, Line } from 'react-chartjs-2'
import { WINDOW_SIZES } from '../constants.js'
import { fmt } from './MetricsDisplay.jsx'

ChartJS.register(LinearScale, PointElement, LineElement, Title, Tooltip, Legend)

// Checkbox-selection state for a history table (select one / select all /
// clear) — identical logic needed by every eval-history table across the
// per-run and cross-match calibration pages.
export function useRowSelection() {
  const [selected, setSelected] = useState(new Set())

  const toggle = (id) => {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id); else next.add(id)
      return next
    })
  }
  const toggleAll = (ids, checked) => {
    setSelected(checked ? new Set(ids) : new Set())
  }
  const clear = () => setSelected(new Set())

  return { selected, toggle, toggleAll, clear }
}

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

// ---------------------------------------------------------------------------
// VisualScore weight calibration — R², MSE, MAE vs BaseScore weight.
// One chart per metric (not one chart with all three overlaid): R² can go
// deeply negative while MSE/MAE sit in a small 0-1-ish band, so sharing an
// axis crushes the readable ones flat. Three small panels, same layout
// pattern as the merge-weight scatter panels above.
// ---------------------------------------------------------------------------
const VISUAL_METRICS = [
  { key: 'r2', label: 'R²', color: '#a855f7', higherBetter: true },
  { key: 'mse', label: 'MSE', color: '#f59e0b', higherBetter: false },
  { key: 'mae', label: 'MAE', color: '#3b82f6', higherBetter: false },
]

// A real linear x-axis (0..1, numeric {x,y} points) rather than string
// category labels — category scales space ticks by index, not by value, so
// min/max on them don't mean what they look like they mean and duplicate or
// unsorted x-values can visually compress the range. Two datasets share
// this axis: one plotted against each config's base weight, the other
// against its flow weight (=1-base) — since they're complements, the two
// lines mirror each other across x=0.5, showing the metric's response to
// either weight increasing on one chart.
function visualMetricOptions(metric) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: { legend: { labels: { color: '#b0b0b0', font: { size: 10 }, boxWidth: 12 } } },
    scales: {
      x: {
        type: 'linear',
        title: { display: true, text: 'Weight', color: '#707070', font: { size: 10 } },
        ticks: { color: '#707070', font: { size: 9 }, stepSize: 0.1 },
        grid: { color: '#2a2a2a' },
        min: 0, max: 1,
      },
      y: {
        title: { display: true, text: metric.label, color: '#707070', font: { size: 10 } },
        ticks: { color: '#707070', font: { size: 9 } },
        grid: { color: '#2a2a2a' },
      },
    },
  }
}

function VisualMetricPanel({ metric, evals, best }) {
  const chartData = useMemo(() => {
    const baseLine = [...evals]
      .map(ev => ({ x: ev.weights.base, y: ev.metrics[metric.key] }))
      .sort((a, b) => a.x - b.x)
    const flowLine = [...evals]
      .map(ev => ({ x: ev.weights.flow, y: ev.metrics[metric.key] }))
      .sort((a, b) => a.x - b.x)
    return {
      datasets: [
        {
          label: 'Base weight', data: baseLine,
          borderColor: metric.color, backgroundColor: metric.color + '1a',
          tension: 0.3, pointRadius: 3, pointHoverRadius: 5,
        },
        {
          label: 'Flow weight', data: flowLine,
          borderColor: '#22c55e', backgroundColor: '#22c55e1a',
          borderDash: [4, 3],
          tension: 0.3, pointRadius: 3, pointHoverRadius: 5,
        },
      ],
    }
  }, [evals, metric])

  const options = useMemo(() => visualMetricOptions(metric), [metric])

  return (
    <div className="card" style={{ flex: 1, minWidth: 0 }}>
      <h2 style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: 0.5, color: 'var(--text-2)' }}>
        {metric.label} vs Weight
        {best && <span style={{ color: 'var(--green)', fontWeight: 400, textTransform: 'none', marginLeft: 6 }}>
          · best {fmt(best.metrics[metric.key], 4)} @ base={fmt(best.weights.base, 2)}
        </span>}
      </h2>
      <div style={{ height: 180 }}>
        <Line data={chartData} options={options} />
      </div>
    </div>
  )
}

// evals: array of { weights: {base, flow}, metrics: {mae, mse, r2}, ... } —
// works for both a single run's evals and the full cross-match list.
export function VisualScoreMetricCharts({ evals }) {
  if (evals.length === 0) return null
  return (
    <div style={{ display: 'flex', flexDirection: 'row', gap: 16, flexWrap: 'wrap' }}>
      {VISUAL_METRICS.map(metric => {
        const best = evals.reduce((b, ev) => {
          if (b === null) return ev
          const better = metric.higherBetter
            ? ev.metrics[metric.key] > b.metrics[metric.key]
            : ev.metrics[metric.key] < b.metrics[metric.key]
          return better ? ev : b
        }, null)
        return <VisualMetricPanel key={metric.key} metric={metric} evals={evals} best={best} />
      })}
    </div>
  )
}
