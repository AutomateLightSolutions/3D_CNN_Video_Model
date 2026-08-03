"""Persistence and lifecycle rules for PredictionRun records.

Directly analogous to training_history.py, but simpler: inference runs as a
single in-process BackgroundTask (not an OS subprocess with a PID file), so
there's no separate "is it still alive" check — inference.run_inference
itself sets the terminal status directly. What this module owns instead is
list/delete (including disk cleanup, unlike training_history's DB-only
delete — prediction clips are disposable derived data, not curated training
data) and the concurrency guard, since two runs sharing one GPU in-process
would silently contend with each other.
"""

import shutil
from typing import Optional

from sqlalchemy.orm import Session

import models
from config import PREDICTIONS_DIR


def start_run(db: Session, match_id: int, model_type: str, device: str) -> models.PredictionRun:
    run = models.PredictionRun(
        match_id=match_id, model_type=model_type, device=device, status="running",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def list_runs(db: Session, match_id: Optional[int] = None):
    q = db.query(models.PredictionRun)
    if match_id:
        q = q.filter(models.PredictionRun.match_id == match_id)
    return q.order_by(models.PredictionRun.started_at.desc()).all()


def latest_running_any(db: Session) -> Optional[models.PredictionRun]:
    """Any PredictionRun currently 'running', across all matches/models —
    the concurrency guard, since inference shares the server's GPU in-process."""
    return (
        db.query(models.PredictionRun)
        .filter(models.PredictionRun.status == "running")
        .order_by(models.PredictionRun.started_at.desc())
        .first()
    )


def delete_run(db: Session, run_id: int) -> bool:
    """Remove the history record AND its on-disk clip directory — unlike
    training_history.delete_run (checkpoints are curated, kept on purpose),
    prediction clips are disposable derived data."""
    run = db.query(models.PredictionRun).filter(models.PredictionRun.id == run_id).first()
    if run is None:
        return False
    db.delete(run)
    db.commit()
    shutil.rmtree(PREDICTIONS_DIR / str(run_id), ignore_errors=True)
    return True


def sweep_stale_running(db: Session) -> None:
    """Run once at server startup. Any row still 'running' belongs to a
    previous server process that was killed mid-run."""
    stale = db.query(models.PredictionRun).filter(models.PredictionRun.status == "running").all()
    for run in stale:
        run.status = "stopped"
    if stale:
        db.commit()
