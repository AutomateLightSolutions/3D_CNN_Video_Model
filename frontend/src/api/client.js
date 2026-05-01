const BASE = '/api'

async function req(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options)
  if (!res.ok) {
    let msg = `HTTP ${res.status}`
    try { const d = await res.json(); msg = d.detail || JSON.stringify(d) } catch {}
    throw new Error(msg)
  }
  const ct = res.headers.get('content-type') || ''
  if (ct.includes('application/json')) return res.json()
  return res
}

// --- Matches ---
export const createMatch = (name, file_path) =>
  req('/matches', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, file_path }) })

export const listMatches = () => req('/matches')

export const extractClips = (id) =>
  req(`/matches/${id}/extract`, { method: 'POST' })

export const deleteMatch = (id) =>
  req(`/matches/${id}`, { method: 'DELETE' })

export const getExtractionProgress = (id) =>
  req(`/matches/${id}/progress`)

export const getExtractionLog = (id) =>
  req(`/matches/${id}/extraction-log`)

// --- Clips ---
export const listClips = ({ matchId, status, windowSize } = {}) => {
  const p = new URLSearchParams()
  if (matchId)    p.set('match_id', matchId)
  if (status)     p.set('status', status)
  if (windowSize) p.set('window_size', windowSize)
  return req(`/clips?${p}`)
}

export const getClip = (id) => req(`/clips/${id}`)

export const getNextClip = ({ matchId, afterClipId } = {}) => {
  const p = new URLSearchParams()
  if (matchId)    p.set('match_id', matchId)
  if (afterClipId) p.set('after_clip_id', afterClipId)
  return req(`/clips/next?${p}`)
}

export const skipClip = (id) =>
  req(`/clips/${id}/skip`, { method: 'POST' })

// --- Labels ---
export const saveLabel = (payload) =>
  req('/labels', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })

export const listLabels = (matchId) => {
  const p = new URLSearchParams()
  if (matchId) p.set('match_id', matchId)
  return req(`/labels?${p}`)
}

// --- Training ---
export const startTraining = () =>
  req('/training/start', { method: 'POST' })

export const stopTraining = () =>
  req('/training/stop', { method: 'POST' })

export const getTrainingStatus = () => req('/training/status')

export const getTrainingLogs = () => req('/training/logs')

// --- Export ---
export const getExportStats = () => req('/export/stats')

export const downloadJson = () => {
  window.location.href = `${BASE}/export/json`
}

export const downloadCsv = () => {
  window.location.href = `${BASE}/export/csv`
}
