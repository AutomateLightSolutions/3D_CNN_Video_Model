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
