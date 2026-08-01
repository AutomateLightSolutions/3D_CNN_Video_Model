import TrainingPage from '../components/TrainingPage.jsx'
import FeatureExtractionPanel from '../components/FeatureExtractionPanel.jsx'
import {
  startR3DTraining, stopR3DTraining,
  getR3DTrainingStatus, getR3DTrainingLogs, getR3DTrainingMetrics,
} from '../api/client.js'

const api = {
  start:   (cfg) => startR3DTraining(cfg),
  stop:    ()    => stopR3DTraining(),
  status:  ()    => getR3DTrainingStatus(),
  logs:    ()    => getR3DTrainingLogs(),
  metrics: ()    => getR3DTrainingMetrics(),
}

export default function TrainR3D() {
  return (
    <TrainingPage
      title="Training — R3D-18"
      defaultEpochs={40}
      defaultBatch={4}
      api={api}
      modelType="r3d"
      extraControls={<FeatureExtractionPanel required={false} />}
    />
  )
}
