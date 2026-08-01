from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, UniqueConstraint
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    duration_seconds = Column(Float, nullable=True)
    fps = Column(Float, nullable=True)
    status = Column(String, default="registered")
    created_at = Column(DateTime, default=datetime.utcnow)

    clips = relationship("Clip", back_populates="match", cascade="all, delete-orphan")


class Clip(Base):
    __tablename__ = "clips"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    clip_path = Column(String, nullable=False)
    t_start = Column(Float, nullable=False)
    t_end = Column(Float, nullable=False)
    window_size = Column(Integer, nullable=False)
    status = Column(String, default="unlabeled")

    match = relationship("Match", back_populates="clips")
    label = relationship("Label", back_populates="clip", uselist=False, cascade="all, delete-orphan")


class TrainingRun(Base):
    """One row per training run (one per Start click), across all 6 model
    families. Unlike each trainer's own metrics.json — which is overwritten
    every run — this is the permanent, deletable history record."""
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String, nullable=False)   # "r3d" | "videomae" | "slowfast" | "rf" | "mlp" | "hybrid"
    backbone = Column(String, nullable=True)       # set only for "hybrid"
    status = Column(String, default="running")     # "running" | "completed" | "stopped"

    epochs = Column(Integer, nullable=True)
    batch_size = Column(Integer, nullable=True)
    lr = Column(Float, nullable=True)
    device = Column(String, nullable=True)

    n_train = Column(Integer, nullable=True)
    n_val = Column(Integer, nullable=True)
    metrics_json = Column(String, nullable=True)   # JSON blob: best/per_class_f1/confusion_matrix/feature_importance

    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class Label(Base):
    __tablename__ = "labels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    clip_id = Column(Integer, ForeignKey("clips.id"), unique=True, nullable=False)
    event_class = Column(String, nullable=False)
    highlight_score = Column(Float, nullable=False)
    # VisualScore = 0.60 * BaseScore(event_class) + 0.40 * OpticalFlowMagnitude_norm
    # Populated by feature_extractor.py once optical flow is available for this clip.
    merged_visual_score = Column(Float, nullable=True)
    t_start_adjusted = Column(Float, nullable=False)
    t_end_adjusted = Column(Float, nullable=False)
    notes = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    clip = relationship("Clip", back_populates="label")
