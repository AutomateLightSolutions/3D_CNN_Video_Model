from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime


class MatchCreate(BaseModel):
    name: str
    file_path: str


class LabelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    clip_id: int
    event_class: str
    highlight_score: float
    merged_visual_score: Optional[float] = None
    t_start_adjusted: float
    t_end_adjusted: float
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ClipOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    match_id: int
    clip_path: str
    clip_url: str
    t_start: float
    t_end: float
    window_size: int
    status: str
    label: Optional[LabelOut] = None


class MatchOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    file_path: str
    duration_seconds: Optional[float] = None
    fps: Optional[float] = None
    status: str
    created_at: datetime
    clip_count: int
    labeled_count: int


class ExtractionProgress(BaseModel):
    clips_total: int
    clips_done: int


class LabelCreate(BaseModel):
    clip_id: int
    event_class: str
    highlight_score: float
    t_start_adjusted: float
    t_end_adjusted: float
    notes: Optional[str] = None


class TrainingStatus(BaseModel):
    status: str
    current_epoch: Optional[int] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    val_acc: Optional[float] = None


class TrainingConfig(BaseModel):
    epochs: int = 40
    batch_size: int = 4
    lr: float = 1e-3
    device: str = "cuda"
    backbone: str = "r3d"  # for trainer_hybrid: "r3d" | "videomae" | "slowfast"


class ClipFilterStatus(BaseModel):
    active: bool
    n_clips: int = 0


class TrainingRunOut(BaseModel):
    id: int
    model_type: str
    backbone: Optional[str] = None
    status: str
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    lr: Optional[float] = None
    device: Optional[str] = None
    n_train: Optional[int] = None
    n_val: Optional[int] = None
    metrics: dict = {}
    started_at: datetime
    completed_at: Optional[datetime] = None


class PredictionConfig(BaseModel):
    match_id: int
    model_type: str
    device: str = "cuda"


class PredictionWindowResultOut(BaseModel):
    window_size: int
    event_class: str
    confidence: float
    highlight_score: float
    class_probs: list = []


class PredictionSegmentOut(BaseModel):
    id: int
    tile_index: int
    global_start_time: float
    global_end_time: float
    predicted_event: str
    highlight_score: float
    clip_url: Optional[str] = None
    windows: list = []


class PredictionRunOut(BaseModel):
    id: int
    match_id: int
    model_type: str
    backbone: Optional[str] = None
    device: Optional[str] = None
    status: str
    class_support: dict = {}
    error_message: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None


class PredictionProgress(BaseModel):
    tiles_total: int
    tiles_done: int
    status: str


# ---------------------------------------------------------------------------
# Merge-weight calibration (admin)
# ---------------------------------------------------------------------------

class GroundTruthStatus(BaseModel):
    uploaded: bool
    n_tiles: int = 0


class WeightEvalRequest(BaseModel):
    weights: dict    # {"8": float, "16": float, "32": float} — used for both class vote and score merge
    label: Optional[str] = None


class WeightEvalOut(BaseModel):
    id: int
    prediction_run_id: int
    label: Optional[str] = None
    weights: dict
    n_tiles: int
    metrics: dict
    created_at: datetime


class WeightEvalWithContextOut(WeightEvalOut):
    """WeightEvalOut plus which model/match it belongs to — for the
    cross-match calibration overview page, where evals from every run are
    listed together instead of scoped to one run."""
    model_type: str
    backbone: Optional[str] = None
    match_id: int
    match_name: str


class EvaluateAllResult(BaseModel):
    created: int
    skipped_duplicate: int
    total_combos: int
