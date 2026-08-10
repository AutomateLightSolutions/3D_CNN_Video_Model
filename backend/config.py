import os
from pathlib import Path

BASE_DIR = Path(os.environ.get("APP_BASE_DIR", Path(__file__).parent / "Storage"))
CLIPS_DIR = BASE_DIR / "clips"
EXPORT_DIR = BASE_DIR / "exports"
DB_PATH = BASE_DIR / "labels.db"

# Per-model output directories
R3D_MODEL_DIR      = BASE_DIR / "models" / "r3d"
VIDEOMAE_MODEL_DIR = BASE_DIR / "models" / "videomae"
SLOWFAST_MODEL_DIR = BASE_DIR / "models" / "slowfast"

# Backward-compat alias used by existing code
MODEL_DIR = R3D_MODEL_DIR

# Per-model training log paths
R3D_LOG_PATH      = BASE_DIR / "r3d_training.log"
VIDEOMAE_LOG_PATH = BASE_DIR / "videomae_training.log"
SLOWFAST_LOG_PATH = BASE_DIR / "slowfast_training.log"

# Backward-compat alias
TRAINING_LOG_PATH = R3D_LOG_PATH

# Per-model PID files (one running process per model)
R3D_PID_FILE      = BASE_DIR / "r3d_trainer.pid"
VIDEOMAE_PID_FILE = BASE_DIR / "videomae_trainer.pid"
SLOWFAST_PID_FILE = BASE_DIR / "slowfast_trainer.pid"

# Backward-compat alias
PID_FILE = R3D_PID_FILE

# Per-model metrics files (written each epoch by trainer)
_METRICS_FILENAME     = "metrics.json"
R3D_METRICS_PATH      = R3D_MODEL_DIR / _METRICS_FILENAME
VIDEOMAE_METRICS_PATH = VIDEOMAE_MODEL_DIR / _METRICS_FILENAME
SLOWFAST_METRICS_PATH = SLOWFAST_MODEL_DIR / _METRICS_FILENAME

# Interpretable feature cache
FEATURES_DIR      = BASE_DIR / "features"
FEATURES_PROGRESS = FEATURES_DIR / "progress.json"

# Interpretable Features + Random Forest
RF_MODEL_DIR   = BASE_DIR / "models" / "rf"
RF_LOG_PATH    = BASE_DIR / "rf_training.log"
RF_METRICS_PATH = RF_MODEL_DIR / _METRICS_FILENAME
RF_PID_FILE    = BASE_DIR / "rf_trainer.pid"

# Interpretable Features + MLP
MLP_MODEL_DIR    = BASE_DIR / "models" / "mlp"
MLP_LOG_PATH     = BASE_DIR / "mlp_training.log"
MLP_METRICS_PATH = MLP_MODEL_DIR / _METRICS_FILENAME
MLP_PID_FILE     = BASE_DIR / "mlp_trainer.pid"

# Interpretable + Deep Features Hybrid
HYBRID_MODEL_DIR    = BASE_DIR / "models" / "hybrid"
HYBRID_LOG_PATH     = BASE_DIR / "hybrid_training.log"
HYBRID_METRICS_PATH = HYBRID_MODEL_DIR / _METRICS_FILENAME
HYBRID_PID_FILE     = BASE_DIR / "hybrid_trainer.pid"

HIGHLIGHT_CLASSES = [
    "try", "goal_kick", "card_event", "penalty",
    "scrum", "maul", "lineout", "kick_off",
    "tmo_replay", "normal_play",
]

# BaseScore per class for VisualScore formula:
# VisualScore = 0.60 * BaseScore(class) + 0.40 * OpticalFlowMagnitude_norm
# Derived from audience survey on highlight-worthiness per event type
# (survey score / 100, normalised to [0, 1]).
BASE_SCORES = {
    "try":         1.0000,   # 100.00
    "goal_kick":   0.7814,   # 78.14
    "card_event":  1.0000,   # 100.00
    "penalty":     0.3546,   # 35.46
    "lineout":     0.1210,   # 12.10
    "scrum":       0.1461,   # 14.61
    "maul":        0.2353,   # 23.53
    "kick_off":    0.1931,   # 19.31
    "tmo_replay":  0.7500,   # 75.00
    "normal_play": 0.0054,   # 0.54
}

CONTEXT_CLASSES = ["tmo_replay"]

WINDOW_CONFIG = {
    8:  {"frames": 16, "stride_s": 4},
    16: {"frames": 32, "stride_s": 8},
    32: {"frames": 48, "stride_s": 16},
}

WINDOW_SIZES = [8, 16, 32]

CLIP_FPS = 25
CLIP_RESOLUTION = 224

# ---------------------------------------------------------------------------
# Full-match prediction ("Predict" feature)
# ---------------------------------------------------------------------------

# Nested under CLIPS_DIR so the existing /media StaticFiles mount + _clip_url()
# helper in main.py serve prediction clips without any extra wiring.
PREDICTIONS_DIR = CLIPS_DIR / "_predictions"

TILE_SIZE = 8  # seconds — the merged-timeline resolution

# Minimum labeled examples a (window_size, event_class) pair needs in the DB
# before that window size is trusted to vote for that class during merge.
# Placeholder — revisit once real per-window label volumes are known.
CLASS_SUPPORT_MIN_COUNT = 5

# Merge weight favoring the finest (8s) window, shared by both the class-vote
# and score-merge steps; renormalized over whichever windows actually
# contribute for a given tile (see inference.py).
MERGE_WEIGHTS = {8: 0.5, 16: 0.05, 32: 0.45}

PREDICTION_MODEL_TYPES = ("r3d", "videomae", "slowfast", "rf", "mlp", "hybrid")
