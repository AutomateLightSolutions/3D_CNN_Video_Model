import { useState, useMemo } from 'react'
import TrainingPage from '../components/TrainingPage.jsx'
import FeatureExtractionPanel from '../components/FeatureExtractionPanel.jsx'
import {
  startHybridTraining, stopHybridTraining,
  getHybridTrainingStatus, getHybridTrainingLogs, getHybridTrainingMetrics,
} from '../api/client.js'

const inputStyle = {
  padding: '4px 8px', borderRadius: 6,
  border: '1px solid var(--border)',
  background: 'var(--surface-2)', color: 'var(--text-1)',
}

export default function TrainHybrid() {
  const [backbone,   setBackbone]   = useState('r3d')
  const [isRunning,  setIsRunning]  = useState(false)

  const api = useMemo(() => ({
    start:   (cfg) => startHybridTraining({ ...cfg, backbone }),
    stop:    ()    => stopHybridTraining(),
    status:  ()    => getHybridTrainingStatus().then(s => { setIsRunning(s.status === 'running'); return s }),
    logs:    ()    => getHybridTrainingLogs(),
    metrics: ()    => getHybridTrainingMetrics(),
  }), [backbone])

  const extraControls = (
    <>
      <FeatureExtractionPanel required={true} />
      <div>
        <label htmlFor="backbone-select" style={{ display: 'block', fontSize: 11, color: 'var(--text-2)', marginBottom: 4 }}>
          Backbone
        </label>
        <select
          id="backbone-select"
          value={backbone}
          onChange={e => setBackbone(e.target.value)}
          disabled={isRunning}
          style={inputStyle}
        >
          <option value="r3d">R3D-18</option>
          <option value="videomae">VideoMAE</option>
          <option value="slowfast">SlowFast</option>
        </select>
      </div>
    </>
  )

  return (
    <TrainingPage
      title="Training — Interpretable + Deep (Hybrid)"
      defaultEpochs={30}
      defaultBatch={8}
      api={api}
      extraControls={extraControls}
    />
  )
}
