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
    prediction_runs = relationship("PredictionRun", back_populates="match", cascade="all, delete-orphan")


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


class PredictionRun(Base):
    """One full-match inference run: a chosen model_type applied to a Match's
    source video, tiled into the merged highlight timeline. Deliberately
    separate from Clip/Label (the curated human-annotated training set) —
    prediction output is disposable derived data, not training data."""
    __tablename__ = "prediction_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    model_type = Column(String, nullable=False)   # "r3d" | "videomae" | "slowfast" | "rf" | "mlp" | "hybrid"
    backbone = Column(String, nullable=True)       # set only for "hybrid"
    device = Column(String, nullable=True)
    status = Column(String, default="running")     # "running" | "completed" | "stopped" | "error"

    # Snapshot of {window_size: [event_class, ...]} computed from the live DB
    # at run start — kept for audit reproducibility if label counts change later.
    class_support_json = Column(String, nullable=True)
    error_message = Column(String, nullable=True)

    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    match = relationship("Match", back_populates="prediction_runs")
    segments = relationship("PredictionSegment", back_populates="run", cascade="all, delete-orphan")


class PredictionSegment(Base):
    """One merged tile on the prediction timeline (default 8s resolution)."""
    __tablename__ = "prediction_segments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("prediction_runs.id"), nullable=False)
    tile_index = Column(Integer, nullable=False)

    # Seconds since t=0 of the source match file — the sync key a separate
    # audio/commentary system will later use to fuse with this timeline.
    global_start_time = Column(Float, nullable=False)
    global_end_time = Column(Float, nullable=False)

    predicted_event = Column(String, nullable=False)   # merged (masked-vote) class
    highlight_score = Column(Float, nullable=False)     # merged (weighted-average) score
    clip_path = Column(String, nullable=True)            # the 8s tile's clip only

    run = relationship("PredictionRun", back_populates="segments")
    windows = relationship("PredictionWindowResult", back_populates="segment", cascade="all, delete-orphan")


class PredictionWindowResult(Base):
    """Raw (pre-merge) prediction from one window size (8/16/32s) for one
    tile — kept alongside the merged result for debugging/audit, and so the
    merge can be recomputed later without rerunning the model."""
    __tablename__ = "prediction_window_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    segment_id = Column(Integer, ForeignKey("prediction_segments.id"), nullable=False)
    window_size = Column(Integer, nullable=False)   # 8 | 16 | 32

    event_class = Column(String, nullable=False)     # this window's own top-1 class
    confidence = Column(Float, nullable=False)        # this window's own top-1 softmax prob
    highlight_score = Column(Float, nullable=False)   # this window's own raw score-head output
    class_probs_json = Column(String, nullable=False) # full 10-float softmax vector

    segment = relationship("PredictionSegment", back_populates="windows")
