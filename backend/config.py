import os
from pathlib import Path

BASE_DIR = Path(os.environ.get("APP_BASE_DIR", Path.home() / "highlight_system"))
CLIPS_DIR = BASE_DIR / "clips"
MODEL_DIR = BASE_DIR / "models"
EXPORT_DIR = BASE_DIR / "exports"
DB_PATH = BASE_DIR / "labels.db"
TRAINING_LOG_PATH = BASE_DIR / "training.log"
PID_FILE = BASE_DIR / "trainer.pid"

HIGHLIGHT_CLASSES = [
    "try", "conversion", "kick_off", "penalty_kick", "drop_goal",
    "near_try", "scrum", "lineout", "touch_kick", "intercept",
    "tackle", "ruck", "maul", "red_card", "yellow_card",
    "turnover", "line_break", "cross_kick", "penalty", "knock_on",
    "replay", "tmo_review", "normal_play"
]

# Classes that inherit score from parent event — used in UI logic
CONTEXT_CLASSES = ["replay", "tmo_review"]

# Window config: window_size_s -> {frames, stride_s}
WINDOW_CONFIG = {
    8:  {"frames": 16, "stride_s": 4},
    16: {"frames": 32, "stride_s": 8},
    32: {"frames": 48, "stride_s": 16},
}

WINDOW_SIZES = [8, 16, 32]  # seconds

# Keep for backward compat
CLIP_FPS = 25
CLIP_RESOLUTION = 224
