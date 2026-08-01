import TrainingPage from '../components/TrainingPage.jsx'
import FeatureExtractionPanel from '../components/FeatureExtractionPanel.jsx'
import {
  startRFTraining, stopRFTraining,
  getRFTrainingStatus, getRFTrainingLogs, getRFTrainingMetrics,
} from '../api/client.js'

const api = {
  start:   (cfg) => startRFTraining(cfg),
  stop:    ()    => stopRFTraining(),
  status:  ()    => getRFTrainingStatus(),
  logs:    ()    => getRFTrainingLogs(),
  metrics: ()    => getRFTrainingMetrics(),
}

export default function TrainRF() {
  return (
    <TrainingPage
      title="Training — Interpretable + Random Forest"
      defaultEpochs={10}
      defaultBatch={0}
      api={api}
      modelType="rf"
      extraControls={<FeatureExtractionPanel required={true} />}
    />
  )
}
