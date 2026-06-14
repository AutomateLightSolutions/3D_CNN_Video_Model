import TrainingPage from '../components/TrainingPage.jsx'
import {
  startSlowFastTraining, stopSlowFastTraining,
  getSlowFastTrainingStatus, getSlowFastTrainingLogs, getSlowFastTrainingMetrics,
} from '../api/client.js'

const api = {
  start:   (cfg) => startSlowFastTraining(cfg),
  stop:    ()    => stopSlowFastTraining(),
  status:  ()    => getSlowFastTrainingStatus(),
  logs:    ()    => getSlowFastTrainingLogs(),
  metrics: ()    => getSlowFastTrainingMetrics(),
}

export default function TrainSlowFast() {
  return (
    <TrainingPage
      title="Training — SlowFast R50"
      defaultEpochs={30}
      defaultBatch={2}
      api={api}
    />
  )
}
