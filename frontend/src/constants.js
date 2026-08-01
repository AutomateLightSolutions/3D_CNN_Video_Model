// Single source of truth for model identity across TrainingPage, History,
// and Compare — add a 7th model here and all three pages pick it up.
export const MODEL_TYPES = ['r3d', 'videomae', 'slowfast', 'rf', 'mlp', 'hybrid']

export const MODEL_LABELS = {
  r3d:      'R3D-18',
  videomae: 'VideoMAE-base',
  slowfast: 'SlowFast R50',
  rf:       'Interp + RF',
  mlp:      'Interp + MLP',
  hybrid:   'Interp + Deep (Hybrid)',
}

export const STATUS_BADGE = {
  running:   'badge-blue',
  completed: 'badge-green',
  stopped:   'badge-red',
}
