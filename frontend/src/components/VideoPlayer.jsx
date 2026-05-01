import { useRef, useEffect, useState } from 'react'

export default function VideoPlayer({ src, onTimeUpdate }) {
  const videoRef = useRef(null)
  const [playing, setPlaying] = useState(false)
  const [looping, setLooping] = useState(true)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    v.src = src
    v.load()
    v.play().catch(() => {})
    setPlaying(true)
  }, [src])

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    v.loop = looping
  }, [looping])

  useEffect(() => {
    const handler = (e) => {
      if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return
      const v = videoRef.current
      if (!v) return
      if (e.code === 'Space') {
        e.preventDefault()
        playing ? v.pause() : v.play()
      }
      if (e.code === 'ArrowLeft') { e.preventDefault(); v.currentTime = Math.max(0, v.currentTime - 2) }
      if (e.code === 'ArrowRight') { e.preventDefault(); v.currentTime = Math.min(v.duration, v.currentTime + 2) }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [playing])

  const fmt = (s) => {
    if (isNaN(s)) return '0:00'
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60).toString().padStart(2, '0')
    return `${m}:${sec}`
  }

  return (
    <div>
      <div className="video-wrapper">
        <video
          ref={videoRef}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onTimeUpdate={() => {
            const t = videoRef.current?.currentTime || 0
            setCurrentTime(t)
            onTimeUpdate?.(t)
          }}
          onLoadedMetadata={() => setDuration(videoRef.current?.duration || 0)}
        />
      </div>
      <div className="video-controls">
        <button onClick={() => { const v = videoRef.current; v.currentTime = Math.max(0, v.currentTime - 2) }}>-2s</button>
        <button onClick={() => {
          const v = videoRef.current
          playing ? v.pause() : v.play()
        }}>
          {playing ? '⏸' : '▶'}
        </button>
        <button onClick={() => { const v = videoRef.current; v.currentTime = Math.min(v.duration, v.currentTime + 2) }}>+2s</button>
        <button className={`loop-btn${looping ? ' active' : ''}`} onClick={() => setLooping(l => !l)}>
          ↺ {looping ? 'Loop on' : 'Loop off'}
        </button>
        <span className="time">{fmt(currentTime)} / {fmt(duration)}</span>
      </div>
    </div>
  )
}
