import { useState, useEffect, useRef } from 'react'
import { getFeatureExtractionStatus } from '../api/client.js'

export default function FeatureExtractionPanel({ required = false }) { // NOSONAR
  const [status, setStatus] = useState({ status: 'idle', total: 0, done: 0 })
  const pollRef = useRef(null)

  useEffect(() => {
    const refresh = async () => {
      try {
        const s = await getFeatureExtractionStatus()
        setStatus(s)
      } catch (e) {
        console.warn('Feature status poll failed:', e.message)
      }
    }
    refresh()
    pollRef.current = setInterval(refresh, 5000)
    return () => clearInterval(pollRef.current)
  }, [])

  const isRunning = status.status === 'running'
  const isDone    = status.status === 'done'

  let statusText, statusColor
  if (isRunning) {
    statusText  = `Extracting… ${status.done} / ${status.total}`
    statusColor = 'var(--amber)'
  } else if (isDone) {
    statusText  = `${status.done} files ready`
    statusColor = 'var(--green)'
  } else {
    statusText  = required
      ? 'Not extracted — go to Feature Extraction page first'
      : 'Not extracted (recommended)'
    statusColor = required ? '#f87171' : 'var(--text-2)'
  }

  const labelSuffix = required ? '(Required)' : '(Recommended)'
  const labelColor  = required ? 'var(--amber)' : 'var(--text-2)'
  const borderColor = required ? 'var(--amber)' : 'var(--border)'

  return (
    <div style={{ borderLeft: `2px solid ${borderColor}`, paddingLeft: 12 }}>
      <div style={{ fontSize: 11, color: 'var(--text-2)', marginBottom: 4 }}>
        {'Features '}
        <span style={{ color: labelColor }}>{labelSuffix}</span>
      </div>
      <span style={{ fontSize: 12, color: statusColor }}>{statusText}</span>
    </div>
  )
}
