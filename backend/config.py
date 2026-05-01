import os
from pathlib import Path

BASE_DIR = Path(os.environ.get("APP_BASE_DIR", Path.home() / "highlight_system"))
CLIPS_DIR = BASE_DIR / "clips"
MODEL_DIR = BASE_DIR / "models"
EXPORT_DIR = BASE_DIR / "exports"
DB_PATH = BASE_DIR / "labels.db"
TRAINING_LOG_PATH = BASE_DIR / "training.log"
PID_FILE = BASE_DIR / "trainer.pid"

EVENT_CLASSES = [
    "goal", "near_miss", "penalty", "red_card",
    "key_tackle", "free_kick", "crowd_reaction", "null"
]
WINDOW_SIZES = [2, 4, 8]
CLIP_FPS = 25
CLIP_RESOLUTION = 224
