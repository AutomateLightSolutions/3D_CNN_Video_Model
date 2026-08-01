"""Persistence and lifecycle rules for TrainingRun history records.

Kept separate from main.py (route/transport layer) and from the trainer
subprocesses (which only know how to write their own metrics.json). This
module owns the one question those two don't: "which run is this, and did
it finish, get stopped, or die?"
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

import models

_METRICS_PATH_BY_MODEL: dict = {}


def _metrics_path_for(model_type: str) -> Optional[Path]:
    """Lazy import to avoid a hard dependency on config's directory layout
    at module import time; looks up the metrics.json path for a model_type
    the same way main.py does."""
    if not _METRICS_PATH_BY_MODEL:
        import config
        _METRICS_PATH_BY_MODEL.update({
            "r3d": config.R3D_METRICS_PATH,
            "videomae": config.VIDEOMAE_METRICS_PATH,
            "slowfast": config.SLOWFAST_METRICS_PATH,
            "rf": config.RF_METRICS_PATH,
            "mlp": config.MLP_METRICS_PATH,
            "hybrid": config.HYBRID_METRICS_PATH,
        })
    return _METRICS_PATH_BY_MODEL.get(model_type)


def _read_metrics(model_type: str) -> dict:
    path = _metrics_path_for(model_type)
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def start_run(db: Session, model_type: str, cfg, backbone: Optional[str] = None) -> models.TrainingRun:
    """Create the history record for a run that was just launched.

    Also sweeps any record left in "running" state for this model_type —
    the caller only reaches here after confirming no process is currently
    alive for it, so a lingering "running" row means a previous process
    died without anyone finalizing it.
    """
    _sweep(db, db.query(models.TrainingRun).filter(
        models.TrainingRun.model_type == model_type,
        models.TrainingRun.status == "running",
    ))

    run = models.TrainingRun(
        model_type=model_type,
        backbone=backbone,
        status="running",
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        device=cfg.device,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def finalize_if_stopped(db: Session, model_type: str, is_running: bool, metrics: dict) -> None:
    """Call on every status poll. No-op unless there's an open 'running'
    record for this model and its process is no longer alive."""
    if is_running:
        return
    run = _latest_running(db, model_type)
    if run is None:
        return
    _apply_final_metrics(run, metrics, status="completed" if metrics.get("status") == "done" else "stopped")
    db.commit()


def mark_stopped(db: Session, model_type: str, metrics: dict) -> None:
    """Call immediately after a user-initiated stop, for instant feedback
    instead of waiting for the next status poll to finalize it."""
    run = _latest_running(db, model_type)
    if run is None:
        return
    _apply_final_metrics(run, metrics, status="stopped")
    db.commit()


def sweep_stale_running(db: Session) -> None:
    """Run once at server startup. Any row still 'running' belongs to a
    previous server process (e.g. the app was killed) and can't be trusted."""
    _sweep(db, db.query(models.TrainingRun).filter(models.TrainingRun.status == "running"))


def list_runs(db: Session, model_type: Optional[str] = None):
    q = db.query(models.TrainingRun)
    if model_type:
        q = q.filter(models.TrainingRun.model_type == model_type)
    return q.order_by(models.TrainingRun.started_at.desc()).all()


def delete_run(db: Session, run_id: int) -> bool:
    """Remove a history record only. Checkpoint files on disk are left
    untouched by design — this is a list-curation action, not a disk-cleanup
    action."""
    run = db.query(models.TrainingRun).filter(models.TrainingRun.id == run_id).first()
    if run is None:
        return False
    db.delete(run)
    db.commit()
    return True


def latest_completed_per_model(db: Session) -> dict:
    """One entry per model_type: its most recently completed run."""
    result: dict = {}
    runs = (
        db.query(models.TrainingRun)
        .filter(models.TrainingRun.status == "completed")
        .order_by(models.TrainingRun.completed_at.desc())
        .all()
    )
    for run in runs:
        result.setdefault(run.model_type, run)
    return result


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _latest_running(db: Session, model_type: str) -> Optional[models.TrainingRun]:
    return (
        db.query(models.TrainingRun)
        .filter(models.TrainingRun.model_type == model_type, models.TrainingRun.status == "running")
        .order_by(models.TrainingRun.started_at.desc())
        .first()
    )


def _sweep(db: Session, query) -> None:
    """Finalize rows left in 'running' state (server restart, crash, or a
    completion the status-poller never got to see). Reads each row's own
    metrics.json before giving up on it — a run that actually finished
    should end up 'completed' with its real numbers, not 'stopped' with
    nothing, just because nobody polled it in time."""
    stale = query.all()
    for run in stale:
        metrics = _read_metrics(run.model_type)
        if metrics.get("status") == "done":
            _apply_final_metrics(run, metrics, status="completed")
        else:
            run.status = "stopped"
            run.completed_at = datetime.utcnow()
    if stale:
        db.commit()


def _apply_final_metrics(run: models.TrainingRun, metrics: dict, status: str) -> None:
    run.status = status
    run.completed_at = datetime.utcnow()
    run.n_train = metrics.get("n_train")
    run.n_val = metrics.get("n_val")
    run.metrics_json = json.dumps({
        "best": metrics.get("best", {}),
        "per_class_f1": metrics.get("per_class_f1", {}),
        "confusion_matrix": metrics.get("confusion_matrix", []),
        "feature_importance": metrics.get("feature_importance", {}),
        "classes": metrics.get("classes", []),
        "current_epoch": metrics.get("current_epoch"),
        "total_epochs": metrics.get("total_epochs"),
    })
