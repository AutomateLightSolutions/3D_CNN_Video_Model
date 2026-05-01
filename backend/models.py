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


class Label(Base):
    __tablename__ = "labels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    clip_id = Column(Integer, ForeignKey("clips.id"), unique=True, nullable=False)
    event_class = Column(String, nullable=False)
    highlight_score = Column(Float, nullable=False)
    t_start_adjusted = Column(Float, nullable=False)
    t_end_adjusted = Column(Float, nullable=False)
    notes = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    clip = relationship("Clip", back_populates="label")
