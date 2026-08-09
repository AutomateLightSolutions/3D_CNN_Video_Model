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

const json = (body) => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

// --- Matches ---
export const createMatch    = (name, file_path) => req('/matches', json({ name, file_path }))
export const browseMatchFile = ()               => req('/matches/browse', { method: 'POST' })
export const listMatches    = ()                 => req('/matches')
export const extractClips   = (id)              => req(`/matches/${id}/extract`, { method: 'POST' })
export const deleteMatch    = (id)              => req(`/matches/${id}`, { method: 'DELETE' })
export const getExtractionProgress = (id)       => req(`/matches/${id}/progress`)
export const getExtractionLog      = (id)       => req(`/matches/${id}/extraction-log`)

// --- Clips ---
export const listClips = ({ matchId, status, windowSize } = {}) => {
  const p = new URLSearchParams()
  if (matchId)    p.set('match_id', matchId)
  if (status)     p.set('status', status)
  if (windowSize) p.set('window_size', windowSize)
  return req(`/clips?${p}`)
}
export const getClip     = (id)                              => req(`/clips/${id}`)
export const getNextClip = ({ matchId, afterClipId, windowSize } = {}) => {
  const p = new URLSearchParams()
  if (matchId)     p.set('match_id', matchId)
  if (afterClipId) p.set('after_clip_id', afterClipId)
  if (windowSize)  p.set('window_size', windowSize)
  return req(`/clips/next?${p}`)
}
export const skipClip = (id) => req(`/clips/${id}/skip`, { method: 'POST' })

// --- Labels ---
export const saveLabel  = (payload) => req('/labels', json(payload))
export const listLabels = (matchId) => {
  const p = new URLSearchParams()
  if (matchId) p.set('match_id', matchId)
  return req(`/labels?${p}`)
}

// --- Training helpers (internal) ---
function trainingApi(model) {
  const base = `/training/${model}`
  return {
    start:   (cfg = {}) => req(`${base}/start`,   json(cfg)),
    stop:    ()         => req(`${base}/stop`,     { method: 'POST' }),
    status:  ()         => req(`${base}/status`),
    logs:    ()         => req(`${base}/logs`),
    metrics: ()         => req(`${base}/metrics`),
  }
}

// --- Training — R3D-18 ---
const _r3d = trainingApi('r3d')
export const startR3DTraining    = (cfg) => _r3d.start(cfg)
export const stopR3DTraining     = ()    => _r3d.stop()
export const getR3DTrainingStatus  = ()  => _r3d.status()
export const getR3DTrainingLogs    = ()  => _r3d.logs()
export const getR3DTrainingMetrics = ()  => _r3d.metrics()

// --- Training — VideoMAE ---
const _videomae = trainingApi('videomae')
export const startVideoMAETraining    = (cfg) => _videomae.start(cfg)
export const stopVideoMAETraining     = ()    => _videomae.stop()
export const getVideoMAETrainingStatus  = ()  => _videomae.status()
export const getVideoMAETrainingLogs    = ()  => _videomae.logs()
export const getVideoMAETrainingMetrics = ()  => _videomae.metrics()

// --- Training — SlowFast ---
const _slowfast = trainingApi('slowfast')
export const startSlowFastTraining    = (cfg) => _slowfast.start(cfg)
export const stopSlowFastTraining     = ()    => _slowfast.stop()
export const getSlowFastTrainingStatus  = ()  => _slowfast.status()
export const getSlowFastTrainingLogs    = ()  => _slowfast.logs()
export const getSlowFastTrainingMetrics = ()  => _slowfast.metrics()

// --- Training — Interpretable + RF ---
const _rf = trainingApi('rf')
export const startRFTraining    = (cfg) => _rf.start(cfg)
export const stopRFTraining     = ()    => _rf.stop()
export const getRFTrainingStatus  = ()  => _rf.status()
export const getRFTrainingLogs    = ()  => _rf.logs()
export const getRFTrainingMetrics = ()  => _rf.metrics()

// --- Training — Interpretable + MLP ---
const _mlp = trainingApi('mlp')
export const startMLPTraining    = (cfg) => _mlp.start(cfg)
export const stopMLPTraining     = ()    => _mlp.stop()
export const getMLPTrainingStatus  = ()  => _mlp.status()
export const getMLPTrainingLogs    = ()  => _mlp.logs()
export const getMLPTrainingMetrics = ()  => _mlp.metrics()

// --- Training — Interpretable + Deep Hybrid ---
const _hybrid = trainingApi('hybrid')
export const startHybridTraining    = (cfg) => _hybrid.start(cfg)
export const stopHybridTraining     = ()    => _hybrid.stop()
export const getHybridTrainingStatus  = ()  => _hybrid.status()
export const getHybridTrainingLogs    = ()  => _hybrid.logs()
export const getHybridTrainingMetrics = ()  => _hybrid.metrics()

// --- Training History ---
export const listTrainingRuns = (modelType) => {
  const p = new URLSearchParams()
  if (modelType) p.set('model_type', modelType)
  return req(`/history/runs?${p}`)
}
export const deleteTrainingRun = (id) => req(`/history/runs/${id}`, { method: 'DELETE' })
export const getLeaderboard    = ()   => req('/history/leaderboard')

// --- Feature Extraction ---
export const startFeatureExtraction     = ()  => req('/features/extract', { method: 'POST' })
export const stopFeatureExtraction      = ()  => req('/features/stop',   { method: 'POST' })
export const getFeatureExtractionStatus = ()  => req('/features/status')
export const getFeatureExtractionLogs   = ()  => req('/features/logs')

// --- Export ---
export const getExportStats = () => req('/export/stats')
export const downloadJson   = () => { globalThis.location.href = `${BASE}/export/json` }
export const downloadCsv    = () => { globalThis.location.href = `${BASE}/export/csv` }

// --- Predictions (full-match multi-window inference) ---
export const getAvailableModels    = ()        => req('/predictions/available-models')
export const startPrediction       = (cfg)     => req('/predictions/start', json(cfg))
export const listPredictionRuns    = (matchId) => {
  const p = new URLSearchParams()
  if (matchId) p.set('match_id', matchId)
  return req(`/predictions?${p}`)
}
export const getPredictionRun      = (id) => req(`/predictions/${id}`)
export const getPredictionProgress = (id) => req(`/predictions/${id}/progress`)
export const getPredictionLog      = (id) => req(`/predictions/${id}/log`)
export const getPredictionSegments = (id) => req(`/predictions/${id}/segments`)
export const deletePredictionRun   = (id) => req(`/predictions/${id}`, { method: 'DELETE' })

// --- Merge-weight calibration (admin) ---
export const getGroundTruthStatus = (matchId) => req(`/matches/${matchId}/ground-truth`)
export const uploadGroundTruth    = (matchId, file) => {
  const body = new FormData()
  body.append('file', file)
  return req(`/matches/${matchId}/ground-truth`, { method: 'POST', body })
}
export const evaluateWeights   = (runId, payload) => req(`/predictions/${runId}/weight-evals`, json(payload))
export const listWeightEvals   = (runId) => req(`/predictions/${runId}/weight-evals`)
