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
  error:     'badge-red',
}

// Predict page tabs — mirrors the sidebar's "Deep Learning" / "Interpretable" split.
export const PREDICTION_TABS = {
  deep:          { label: 'Deep Learning', models: ['r3d', 'videomae', 'slowfast'] },
  interpretable: { label: 'Interpretable', models: ['rf', 'mlp', 'hybrid'] },
}

// Multi-window merge — mirrors backend config.py's WINDOW_SIZES / MERGE_WEIGHTS.
// One weight per window, shared by both the class-vote and score-merge steps.
export const WINDOW_SIZES = [8, 16, 32]
export const DEFAULT_MERGE_WEIGHTS = { 8: 0.6, 16: 0.3, 32: 0.1 }

// VisualScore training-label formula — mirrors backend config.py's VISUAL_SCORE_WEIGHTS.
export const DEFAULT_VISUAL_SCORE_WEIGHTS = { base: 0.75, flow: 0.25 }
