export default function ScoreSlider({ value, onChange }) {
  // Color interpolates: green (0) -> amber (0.5) -> red (1)
  const color = (() => {
    if (value <= 0.5) {
      const t = value / 0.5
      const r = Math.round(34 + (245 - 34) * t)
      const g = Math.round(197 + (158 - 197) * t)
      const b = Math.round(94 + (11 - 94) * t)
      return `rgb(${r},${g},${b})`
    } else {
      const t = (value - 0.5) / 0.5
      const r = Math.round(245 + (239 - 245) * t)
      const g = Math.round(158 + (68 - 158) * t)
      const b = Math.round(11 + (68 - 11) * t)
      return `rgb(${r},${g},${b})`
    }
  })()

  const trackStyle = {
    background: `linear-gradient(to right, #22c55e 0%, #f59e0b 50%, #ef4444 100%)`,
  }

  return (
    <div className="score-slider-wrap">
      <label>Highlight Score</label>
      <div className="score-row">
        <span className="score-value" style={{ color }}>{value.toFixed(2)}</span>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={value}
          style={trackStyle}
          onChange={(e) => onChange(parseFloat(e.target.value))}
        />
      </div>
    </div>
  )
}
