import csv
import json
import logging
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

import models
import schemas
from clip_extractor import run_extraction
from config import (
    CLIPS_DIR, EXPORT_DIR,
    R3D_MODEL_DIR, VIDEOMAE_MODEL_DIR, SLOWFAST_MODEL_DIR,
    R3D_LOG_PATH, VIDEOMAE_LOG_PATH, SLOWFAST_LOG_PATH,
    R3D_PID_FILE, VIDEOMAE_PID_FILE, SLOWFAST_PID_FILE,
    R3D_METRICS_PATH, VIDEOMAE_METRICS_PATH, SLOWFAST_METRICS_PATH,
    RF_MODEL_DIR, RF_LOG_PATH, RF_PID_FILE, RF_METRICS_PATH,
    MLP_MODEL_DIR, MLP_LOG_PATH, MLP_PID_FILE, MLP_METRICS_PATH,
    HYBRID_MODEL_DIR, HYBRID_LOG_PATH, HYBRID_PID_FILE, HYBRID_METRICS_PATH,
    FEATURES_DIR, FEATURES_PROGRESS,
    HIGHLIGHT_CLASSES, WINDOW_SIZES, WINDOW_CONFIG,
    DB_PATH,
)
from database import Base, engine, get_db

logger = logging.getLogger(__name__)

_MATCH_NOT_FOUND = "Match not found"
_CLIP_NOT_FOUND  = "Clip not found"

Base.metadata.create_all(bind=engine)

for _d in [
    CLIPS_DIR, EXPORT_DIR,
    R3D_MODEL_DIR, VIDEOMAE_MODEL_DIR, SLOWFAST_MODEL_DIR,
    RF_MODEL_DIR, MLP_MODEL_DIR, HYBRID_MODEL_DIR,
    FEATURES_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Highlight Annotation System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=str(CLIPS_DIR), html=False), name="media")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip_url(clip_path: str) -> str:
    try:
        rel = Path(clip_path).relative_to(CLIPS_DIR)
        return "/media/" + str(rel).replace("\\", "/")
    except ValueError:
        return clip_path


def _label_out(label: models.Label) -> Optional[schemas.LabelOut]:
    if label is None:
        return None
    return schemas.LabelOut(
        id=label.id,
        clip_id=label.clip_id,
        event_class=label.event_class,
        highlight_score=label.highlight_score,
        t_start_adjusted=label.t_start_adjusted,
        t_end_adjusted=label.t_end_adjusted,
        notes=label.notes,
        created_at=label.created_at,
        updated_at=label.updated_at,
    )


def _clip_out(clip: models.Clip) -> schemas.ClipOut:
    return schemas.ClipOut(
        id=clip.id,
        match_id=clip.match_id,
        clip_path=clip.clip_path,
        clip_url=_clip_url(clip.clip_path),
        t_start=clip.t_start,
        t_end=clip.t_end,
        window_size=clip.window_size,
        status=clip.status,
        label=_label_out(clip.label),
    )


def _match_out(match: models.Match, db: Session) -> schemas.MatchOut:
    clip_count = db.query(models.Clip).filter(models.Clip.match_id == match.id).count()
    labeled_count = (
        db.query(models.Clip)
        .filter(models.Clip.match_id == match.id, models.Clip.status == "labeled")
        .count()
    )
    return schemas.MatchOut(
        id=match.id,
        name=match.name,
        file_path=match.file_path,
        duration_seconds=match.duration_seconds,
        fps=match.fps,
        status=match.status,
        created_at=match.created_at,
        clip_count=clip_count,
        labeled_count=labeled_count,
    )


# ---------------------------------------------------------------------------
# Training helpers (shared logic for all 3 models)
# ---------------------------------------------------------------------------

def _launch_kwargs():
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}


def _start_trainer(script_name: str, output_dir: Path, log_path: Path, pid_file: Path, cfg: schemas.TrainingConfig, extra_args: list = None):
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            return {"message": "Already running", "pid": pid}
        except (ValueError, OSError):
            pid_file.unlink(missing_ok=True)

    script = Path(__file__).parent / script_name
    cmd = [
        sys.executable, str(script),
        "--data_dir",   str(CLIPS_DIR),
        "--output_dir", str(output_dir),
        "--epochs",     str(cfg.epochs),
        "--batch_size", str(cfg.batch_size),
        "--lr",         str(cfg.lr),
        "--device",     cfg.device,
    ]
    if extra_args:
        cmd.extend(extra_args)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w", encoding="utf-8", buffering=1)

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).parent),
            stdout=log_fh,
            stderr=log_fh,
            stdin=subprocess.DEVNULL,
            **_launch_kwargs(),
        )
    except Exception as exc:
        log_fh.write(f"ERROR: failed to launch trainer: {exc}\n")
        log_fh.close()
        raise HTTPException(status_code=500, detail=str(exc))

    log_fh.close()
    pid_file.write_text(str(proc.pid))
    return {"message": "Training started", "pid": proc.pid}


def _stop_trainer(pid_file: Path):
    if not pid_file.exists():
        return {"message": "No training process found"}
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        pid_file.unlink(missing_ok=True)
        return {"message": "Stopped", "pid": pid}
    except (ValueError, OSError):
        pid_file.unlink(missing_ok=True)
        return {"message": "Process not found, cleaned up"}


def _trainer_status(pid_file: Path, log_path: Path) -> schemas.TrainingStatus:
    running = False
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            running = True
        except (ValueError, OSError):
            pid_file.unlink(missing_ok=True)

    current_epoch = train_loss = val_loss = val_acc = None
    if log_path.exists():
        for line in reversed(log_path.read_text().splitlines()):
            if line.startswith("EPOCH"):
                parts = line.split()
                try:
                    current_epoch = int(parts[1])
                    train_loss    = float(parts[3])
                    val_loss      = float(parts[5])
                    val_acc       = float(parts[7])
                    break
                except (IndexError, ValueError):
                    pass

    return schemas.TrainingStatus(
        status="running" if running else "stopped",
        current_epoch=current_epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        val_acc=val_acc,
    )


def _trainer_logs(log_path: Path):
    if not log_path.exists():
        return {"lines": []}
    return {"lines": log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-100:]}


def _trainer_metrics(metrics_path: Path):
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------

@app.post("/matches", response_model=schemas.MatchOut)
def create_match(body: schemas.MatchCreate, db: Session = Depends(get_db)):
    file_path = Path(body.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=400, detail=f"File not found: {body.file_path}")
    match = models.Match(name=body.name, file_path=str(file_path), status="registered")
    db.add(match)
    db.commit()
    db.refresh(match)
    return _match_out(match, db)


@app.get("/matches", response_model=List[schemas.MatchOut])
def list_matches(db: Session = Depends(get_db)):
    matches = db.query(models.Match).order_by(models.Match.created_at.desc()).all()
    return [_match_out(m, db) for m in matches]


@app.delete("/matches/{match_id}")
def delete_match(match_id: int, db: Session = Depends(get_db)):
    match = db.query(models.Match).filter(models.Match.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail=_MATCH_NOT_FOUND)
    if match.status == "extracting":
        raise HTTPException(status_code=409, detail="Cannot delete while extraction is running")
    db.delete(match)
    db.commit()
    return {"message": "Match deleted"}


@app.post("/matches/{match_id}/extract")
def extract_clips(match_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    match = db.query(models.Match).filter(models.Match.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail=_MATCH_NOT_FOUND)
    if match.status == "extracting":
        raise HTTPException(status_code=409, detail="Extraction already in progress")
    background_tasks.add_task(run_extraction, match_id)
    return {"message": "Extraction started"}


@app.get("/matches/{match_id}/progress", response_model=schemas.ExtractionProgress)
def extraction_progress(match_id: int, db: Session = Depends(get_db)):
    match = db.query(models.Match).filter(models.Match.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail=_MATCH_NOT_FOUND)
    if not match.duration_seconds:
        return schemas.ExtractionProgress(clips_total=0, clips_done=0)
    duration = match.duration_seconds
    total = 0
    for ws, cfg in WINDOW_CONFIG.items():
        stride = cfg["stride_s"]
        if duration >= ws:
            total += int((duration - ws) / stride) + 1
    done = db.query(models.Clip).filter(models.Clip.match_id == match_id).count()
    return schemas.ExtractionProgress(clips_total=total, clips_done=done)


@app.get("/matches/{match_id}/extraction-log")
def extraction_log(match_id: int):
    from config import BASE_DIR
    log_path = BASE_DIR / f"extraction_{match_id}.log"
    if not log_path.exists():
        return {"lines": []}
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"lines": lines[-60:]}


# ---------------------------------------------------------------------------
# Clips  — /clips/next MUST come before /clips/{clip_id}
# ---------------------------------------------------------------------------

@app.get("/clips/next")
def next_clip(
    match_id: Optional[int] = None,
    after_clip_id: Optional[int] = None,
    window_size: Optional[int] = None,
    db: Session = Depends(get_db),
):
    q = db.query(models.Clip).filter(models.Clip.status == "unlabeled")
    if match_id:
        q = q.filter(models.Clip.match_id == match_id)
    if after_clip_id:
        q = q.filter(models.Clip.id > after_clip_id)
    if window_size:
        q = q.filter(models.Clip.window_size == window_size)
    clip = q.order_by(models.Clip.id).first()
    if not clip:
        return None
    return _clip_out(clip)


@app.get("/clips", response_model=List[schemas.ClipOut])
def list_clips(
    match_id: Optional[int] = None,
    status: Optional[str] = None,
    window_size: Optional[int] = None,
    db: Session = Depends(get_db),
):
    q = db.query(models.Clip)
    if match_id:
        q = q.filter(models.Clip.match_id == match_id)
    if status:
        q = q.filter(models.Clip.status == status)
    if window_size:
        q = q.filter(models.Clip.window_size == window_size)
    return [_clip_out(c) for c in q.order_by(models.Clip.t_start).all()]


@app.get("/clips/{clip_id}", response_model=schemas.ClipOut)
def get_clip(clip_id: int, db: Session = Depends(get_db)):
    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail=_CLIP_NOT_FOUND)
    return _clip_out(clip)


@app.post("/clips/{clip_id}/skip")
def skip_clip(clip_id: int, db: Session = Depends(get_db)):
    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail=_CLIP_NOT_FOUND)
    clip.status = "skipped"
    db.commit()
    return {"message": "Clip skipped"}


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

@app.post("/labels", response_model=schemas.LabelOut)
def create_or_update_label(body: schemas.LabelCreate, db: Session = Depends(get_db)):
    clip = db.query(models.Clip).filter(models.Clip.id == body.clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail=_CLIP_NOT_FOUND)

    existing = db.query(models.Label).filter(models.Label.clip_id == body.clip_id).first()
    if existing:
        existing.event_class      = body.event_class
        existing.highlight_score  = body.highlight_score
        existing.t_start_adjusted = body.t_start_adjusted
        existing.t_end_adjusted   = body.t_end_adjusted
        existing.notes            = body.notes
        existing.updated_at       = datetime.now(timezone.utc)
        label = existing
    else:
        label = models.Label(
            clip_id=body.clip_id,
            event_class=body.event_class,
            highlight_score=body.highlight_score,
            t_start_adjusted=body.t_start_adjusted,
            t_end_adjusted=body.t_end_adjusted,
            notes=body.notes,
        )
        db.add(label)

    clip.status = "labeled"
    db.commit()
    db.refresh(label)
    return _label_out(label)


@app.get("/labels", response_model=List[schemas.LabelOut])
def list_labels(match_id: Optional[int] = None, db: Session = Depends(get_db)):
    q = db.query(models.Label)
    if match_id:
        q = q.join(models.Clip).filter(models.Clip.match_id == match_id)
    return [_label_out(lbl) for lbl in q.all()]


# ---------------------------------------------------------------------------
# Training — R3D-18
# ---------------------------------------------------------------------------

@app.post("/training/r3d/start")
def start_r3d_training(cfg: schemas.TrainingConfig = schemas.TrainingConfig()):
    return _start_trainer("trainer_r3d.py", R3D_MODEL_DIR, R3D_LOG_PATH, R3D_PID_FILE, cfg)


@app.post("/training/r3d/stop")
def stop_r3d_training():
    return _stop_trainer(R3D_PID_FILE)


@app.get("/training/r3d/status", response_model=schemas.TrainingStatus)
def r3d_training_status():
    return _trainer_status(R3D_PID_FILE, R3D_LOG_PATH)


@app.get("/training/r3d/logs")
def r3d_training_logs():
    return _trainer_logs(R3D_LOG_PATH)


@app.get("/training/r3d/metrics")
def r3d_training_metrics():
    return _trainer_metrics(R3D_METRICS_PATH)


# ---------------------------------------------------------------------------
# Training — VideoMAE
# ---------------------------------------------------------------------------

@app.post("/training/videomae/start")
def start_videomae_training(cfg: schemas.TrainingConfig = schemas.TrainingConfig(epochs=20, batch_size=2)):
    return _start_trainer("trainer_videomae.py", VIDEOMAE_MODEL_DIR, VIDEOMAE_LOG_PATH, VIDEOMAE_PID_FILE, cfg)


@app.post("/training/videomae/stop")
def stop_videomae_training():
    return _stop_trainer(VIDEOMAE_PID_FILE)


@app.get("/training/videomae/status", response_model=schemas.TrainingStatus)
def videomae_training_status():
    return _trainer_status(VIDEOMAE_PID_FILE, VIDEOMAE_LOG_PATH)


@app.get("/training/videomae/logs")
def videomae_training_logs():
    return _trainer_logs(VIDEOMAE_LOG_PATH)


@app.get("/training/videomae/metrics")
def videomae_training_metrics():
    return _trainer_metrics(VIDEOMAE_METRICS_PATH)


# ---------------------------------------------------------------------------
# Training — SlowFast
# ---------------------------------------------------------------------------

@app.post("/training/slowfast/start")
def start_slowfast_training(cfg: schemas.TrainingConfig = schemas.TrainingConfig(epochs=30, batch_size=2)):
    return _start_trainer("trainer_slowfast.py", SLOWFAST_MODEL_DIR, SLOWFAST_LOG_PATH, SLOWFAST_PID_FILE, cfg)


@app.post("/training/slowfast/stop")
def stop_slowfast_training():
    return _stop_trainer(SLOWFAST_PID_FILE)


@app.get("/training/slowfast/status", response_model=schemas.TrainingStatus)
def slowfast_training_status():
    return _trainer_status(SLOWFAST_PID_FILE, SLOWFAST_LOG_PATH)


@app.get("/training/slowfast/logs")
def slowfast_training_logs():
    return _trainer_logs(SLOWFAST_LOG_PATH)


@app.get("/training/slowfast/metrics")
def slowfast_training_metrics():
    return _trainer_metrics(SLOWFAST_METRICS_PATH)


# ---------------------------------------------------------------------------
# Feature Extraction (prerequisite for RF / MLP / Hybrid trainers)
# ---------------------------------------------------------------------------

_FEATURE_PID_FILE = FEATURES_DIR / "extractor.pid"
_FEATURE_LOG_PATH = FEATURES_DIR / "extractor.log"


@app.post("/features/extract")
def start_feature_extraction():
    if _FEATURE_PID_FILE.exists():
        try:
            pid = int(_FEATURE_PID_FILE.read_text().strip())
            os.kill(pid, 0)
            return {"message": "Already running", "pid": pid}
        except (ValueError, OSError):
            _FEATURE_PID_FILE.unlink(missing_ok=True)

    script = Path(__file__).parent / "feature_extractor.py"
    cmd = [
        sys.executable, str(script),
        "--output_dir", str(FEATURES_DIR),
    ]

    _FEATURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(_FEATURE_LOG_PATH, "w", encoding="utf-8", buffering=1)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).parent),
            stdout=log_fh, stderr=log_fh,
            stdin=subprocess.DEVNULL,
            **_launch_kwargs(),
        )
    except Exception as exc:
        log_fh.close()
        raise HTTPException(status_code=500, detail=str(exc))
    log_fh.close()
    _FEATURE_PID_FILE.write_text(str(proc.pid))
    return {"message": "Feature extraction started", "pid": proc.pid}


@app.get("/features/status")
def feature_extraction_status():
    running = False
    if _FEATURE_PID_FILE.exists():
        try:
            pid = int(_FEATURE_PID_FILE.read_text().strip())
            os.kill(pid, 0)
            running = True
        except (ValueError, OSError):
            _FEATURE_PID_FILE.unlink(missing_ok=True)

    progress = {"total": 0, "done": 0, "status": "idle"}
    if FEATURES_PROGRESS.exists():
        try:
            progress = json.loads(FEATURES_PROGRESS.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    if progress.get("status") == "done":
        # Process finished — clean up stale PID file if present
        _FEATURE_PID_FILE.unlink(missing_ok=True)
    elif running:
        progress["status"] = "running"
    return progress


@app.post("/features/stop")
def stop_feature_extraction():
    return _stop_trainer(_FEATURE_PID_FILE)


@app.get("/features/logs")
def feature_extraction_logs():
    return _trainer_logs(_FEATURE_LOG_PATH)


# ---------------------------------------------------------------------------
# Training — Interpretable Features + Random Forest
# ---------------------------------------------------------------------------

@app.post("/training/rf/start")
def start_rf_training(cfg: schemas.TrainingConfig = schemas.TrainingConfig(epochs=10, batch_size=0)):
    return _start_trainer("trainer_rf.py", RF_MODEL_DIR, RF_LOG_PATH, RF_PID_FILE, cfg)


@app.post("/training/rf/stop")
def stop_rf_training():
    return _stop_trainer(RF_PID_FILE)


@app.get("/training/rf/status", response_model=schemas.TrainingStatus)
def rf_training_status():
    return _trainer_status(RF_PID_FILE, RF_LOG_PATH)


@app.get("/training/rf/logs")
def rf_training_logs():
    return _trainer_logs(RF_LOG_PATH)


@app.get("/training/rf/metrics")
def rf_training_metrics():
    return _trainer_metrics(RF_METRICS_PATH)


# ---------------------------------------------------------------------------
# Training — Interpretable Features + MLP
# ---------------------------------------------------------------------------

@app.post("/training/mlp/start")
def start_mlp_training(cfg: schemas.TrainingConfig = schemas.TrainingConfig(epochs=30, batch_size=32)):
    return _start_trainer("trainer_mlp.py", MLP_MODEL_DIR, MLP_LOG_PATH, MLP_PID_FILE, cfg)


@app.post("/training/mlp/stop")
def stop_mlp_training():
    return _stop_trainer(MLP_PID_FILE)


@app.get("/training/mlp/status", response_model=schemas.TrainingStatus)
def mlp_training_status():
    return _trainer_status(MLP_PID_FILE, MLP_LOG_PATH)


@app.get("/training/mlp/logs")
def mlp_training_logs():
    return _trainer_logs(MLP_LOG_PATH)


@app.get("/training/mlp/metrics")
def mlp_training_metrics():
    return _trainer_metrics(MLP_METRICS_PATH)


# ---------------------------------------------------------------------------
# Training — Interpretable + Deep Features Hybrid
# ---------------------------------------------------------------------------

@app.post("/training/hybrid/start")
def start_hybrid_training(cfg: schemas.TrainingConfig = schemas.TrainingConfig(epochs=30, batch_size=32)):
    return _start_trainer(
        "trainer_hybrid.py", HYBRID_MODEL_DIR, HYBRID_LOG_PATH, HYBRID_PID_FILE, cfg,
        extra_args=["--backbone", cfg.backbone],
    )


@app.post("/training/hybrid/stop")
def stop_hybrid_training():
    return _stop_trainer(HYBRID_PID_FILE)


@app.get("/training/hybrid/status", response_model=schemas.TrainingStatus)
def hybrid_training_status():
    return _trainer_status(HYBRID_PID_FILE, HYBRID_LOG_PATH)


@app.get("/training/hybrid/logs")
def hybrid_training_logs():
    return _trainer_logs(HYBRID_LOG_PATH)


@app.get("/training/hybrid/metrics")
def hybrid_training_metrics():
    return _trainer_metrics(HYBRID_METRICS_PATH)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _build_export_rows(db: Session):
    labels = db.query(models.Label).join(models.Clip).all()
    rows = []
    for lbl in labels:
        clip = lbl.clip
        rows.append({
            "clip_id": clip.id,
            "clip_path": clip.clip_path,
            "match_id": clip.match_id,
            "t_start": clip.t_start,
            "t_end": clip.t_end,
            "window_size": clip.window_size,
            "event_class": lbl.event_class,
            "highlight_score": lbl.highlight_score,
            "t_start_adjusted": lbl.t_start_adjusted,
            "t_end_adjusted": lbl.t_end_adjusted,
            "notes": lbl.notes,
        })
    return rows


@app.get("/export/json")
def export_json(db: Session = Depends(get_db)):
    rows = _build_export_rows(db)
    content = json.dumps(rows, indent=2)
    return StreamingResponse(
        iter([content]),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=labels.json"},
    )


@app.get("/export/csv")
def export_csv(db: Session = Depends(get_db)):
    rows = _build_export_rows(db)
    fieldnames = [
        "clip_id", "clip_path", "match_id", "t_start", "t_end",
        "window_size", "event_class", "highlight_score",
        "t_start_adjusted", "t_end_adjusted", "notes",
    ]
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        row["notes"] = row["notes"] or ""
        writer.writerow(row)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=labels.csv"},
    )


@app.get("/export/stats")
def export_stats(db: Session = Depends(get_db)):
    total    = db.query(models.Clip).count()
    labeled  = db.query(models.Clip).filter(models.Clip.status == "labeled").count()
    skipped  = db.query(models.Clip).filter(models.Clip.status == "skipped").count()
    unlabeled = total - labeled - skipped

    class_counts: dict = dict.fromkeys(HIGHLIGHT_CLASSES, 0)
    for lbl in db.query(models.Label).all():
        if lbl.event_class in class_counts:
            class_counts[lbl.event_class] += 1

    bins = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for lbl in db.query(models.Label).all():
        s = lbl.highlight_score
        if s < 0.2:
            bins["0.0-0.2"] += 1
        elif s < 0.4:
            bins["0.2-0.4"] += 1
        elif s < 0.6:
            bins["0.4-0.6"] += 1
        elif s < 0.8:
            bins["0.6-0.8"] += 1
        else:
            bins["0.8-1.0"] += 1

    return {
        "total": total, "labeled": labeled,
        "skipped": skipped, "unlabeled": unlabeled,
        "class_counts": class_counts, "score_bins": bins,
    }
