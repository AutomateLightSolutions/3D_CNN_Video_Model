import { useRef, useEffect, useCallback } from 'react'

const HANDLE_R = 7
const BAR_H = 18
const PADDING = 16

export default function TimelineEditor({ matchDuration, tStart, tEnd, onChange }) {
  const canvasRef = useRef(null)
  const dragging = useRef(null) // 'start' | 'end' | null
  const stateRef = useRef({ tStart, tEnd, matchDuration })

  useEffect(() => {
    stateRef.current = { tStart, tEnd, matchDuration }
    draw()
  }, [tStart, tEnd, matchDuration])

  const getCanvas = () => canvasRef.current

  const timeToX = (t, W) => {
    const dur = stateRef.current.matchDuration || 1
    return PADDING + (t / dur) * (W - 2 * PADDING)
  }

  const xToTime = (x, W) => {
    const dur = stateRef.current.matchDuration || 1
    const t = ((x - PADDING) / (W - 2 * PADDING)) * dur
    return Math.max(0, Math.min(dur, t))
  }

  const draw = useCallback(() => {
    const canvas = getCanvas()
    if (!canvas) return
    const { tStart: s, tEnd: e, matchDuration: dur } = stateRef.current
    const W = canvas.width
    const H = canvas.height
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, W, H)

    const trackY = H / 2 - BAR_H / 2
    const xS = timeToX(s, W)
    const xE = timeToX(e, W)
    const xFull = timeToX(dur || 1, W)

    // Full match track
    ctx.fillStyle = '#2a2a2a'
    ctx.beginPath()
    ctx.roundRect(PADDING, trackY, xFull - PADDING, BAR_H, 4)
    ctx.fill()

    // Highlighted region
    ctx.fillStyle = 'rgba(59,130,246,0.35)'
    ctx.beginPath()
    ctx.roundRect(xS, trackY, xE - xS, BAR_H, 4)
    ctx.fill()

    // Border on highlight
    ctx.strokeStyle = 'rgba(59,130,246,0.8)'
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.roundRect(xS, trackY, xE - xS, BAR_H, 4)
    ctx.stroke()

    // Handle circles
    const midY = H / 2
    ;[[xS, '#3b82f6'], [xE, '#3b82f6']].forEach(([x, color]) => {
      ctx.fillStyle = '#1e1e1e'
      ctx.beginPath()
      ctx.arc(x, midY, HANDLE_R + 1, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = color
      ctx.beginPath()
      ctx.arc(x, midY, HANDLE_R, 0, Math.PI * 2)
      ctx.fill()
    })
  }, [])

  const hitHandle = (x, W) => {
    const { tStart: s, tEnd: e } = stateRef.current
    const xS = timeToX(s, W)
    const xE = timeToX(e, W)
    if (Math.abs(x - xS) <= HANDLE_R + 4) return 'start'
    if (Math.abs(x - xE) <= HANDLE_R + 4) return 'end'
    return null
  }

  const onMouseDown = (ev) => {
    const canvas = getCanvas()
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const x = (ev.clientX - rect.left) * scaleX
    const hit = hitHandle(x, canvas.width)
    if (hit) { dragging.current = hit; ev.preventDefault() }
  }

  const onMouseMove = useCallback((ev) => {
    if (!dragging.current) return
    const canvas = getCanvas()
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const x = (ev.clientX - rect.left) * scaleX
    const t = xToTime(x, canvas.width)
    const { tStart: s, tEnd: e } = stateRef.current
    if (dragging.current === 'start') {
      const newS = Math.min(t, e - 0.1)
      stateRef.current.tStart = newS
      onChange?.({ tStart: newS, tEnd: e })
    } else {
      const newE = Math.max(t, s + 0.1)
      stateRef.current.tEnd = newE
      onChange?.({ tStart: s, tEnd: newE })
    }
    draw()
  }, [draw, onChange])

  const onMouseUp = () => { dragging.current = null }

  useEffect(() => {
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    return () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }
  }, [onMouseMove])

  const dur = tEnd - tStart
  const fmt = (s) => (s == null ? '—' : s.toFixed(2) + 's')

  return (
    <div className="timeline-wrap">
      <canvas
        ref={canvasRef}
        className="timeline-canvas"
        width={560}
        height={48}
        onMouseDown={onMouseDown}
        style={{ width: '100%', height: 48 }}
      />
      <div className="timeline-labels">
        <span>Start: {fmt(tStart)}</span>
        <span>Duration: {fmt(dur)}</span>
        <span>End: {fmt(tEnd)}</span>
      </div>
    </div>
  )
}
