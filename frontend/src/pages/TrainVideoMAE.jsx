import TrainingPage from '../components/TrainingPage.jsx'
import {
  startVideoMAETraining, stopVideoMAETraining,
  getVideoMAETrainingStatus, getVideoMAETrainingLogs, getVideoMAETrainingMetrics,
} from '../api/client.js'

const api = {
  start:   (cfg) => startVideoMAETraining(cfg),
  stop:    ()    => stopVideoMAETraining(),
  status:  ()    => getVideoMAETrainingStatus(),
  logs:    ()    => getVideoMAETrainingLogs(),
  metrics: ()    => getVideoMAETrainingMetrics(),
}

export default function TrainVideoMAE() {
  return (
    <TrainingPage
      title="Training — VideoMAE-base"
      defaultEpochs={20}
      defaultBatch={2}
      api={api}
    />
  )
}
