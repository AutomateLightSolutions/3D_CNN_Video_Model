import TrainingPage from '../components/TrainingPage.jsx'
import {
  startMLPTraining, stopMLPTraining,
  getMLPTrainingStatus, getMLPTrainingLogs, getMLPTrainingMetrics,
} from '../api/client.js'

const api = {
  start:   (cfg) => startMLPTraining(cfg),
  stop:    ()    => stopMLPTraining(),
  status:  ()    => getMLPTrainingStatus(),
  logs:    ()    => getMLPTrainingLogs(),
  metrics: ()    => getMLPTrainingMetrics(),
}

export default function TrainMLP() {
  return (
    <TrainingPage
      title="Training — Interpretable + MLP"
      defaultEpochs={30}
      defaultBatch={32}
      api={api}
    />
  )
}
