import csv
import json
import logging
import os
import signal
import subprocess
from datetime import datetime
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
    CLIPS_DIR, MODEL_DIR, EXPORT_DIR,
    TRAINING_LOG_PATH, PID_FILE,
    HIGHLIGHT_CLASSES, WINDOW_SIZES, WINDOW_CONFIG,
)
from database import Base, engine, get_db

logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)

for _d in [CLIPS_DIR, MODEL_DIR, EXPORT_DIR]:
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
        raise HTTPException(status_code=404, detail="Match not found")
    if match.status == "extracting":
        raise HTTPException(status_code=409, detail="Cannot delete while extraction is running")
    db.delete(match)
    db.commit()
    return {"message": "Match deleted"}


@app.post("/matches/{match_id}/extract")
def extract_clips(match_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    match = db.query(models.Match).filter(models.Match.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    if match.status == "extracting":
        raise HTTPException(status_code=409, detail="Extraction already in progress")
    background_tasks.add_task(run_extraction, match_id)
    return {"message": "Extraction started"}


@app.get("/matches/{match_id}/progress", response_model=schemas.ExtractionProgress)
def extraction_progress(match_id: int, db: Session = Depends(get_db)):
    match = db.query(models.Match).filter(models.Match.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

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
    db: Session = Depends(get_db),
):
    q = db.query(models.Clip).filter(models.Clip.status == "unlabeled")
    if match_id:
        q = q.filter(models.Clip.match_id == match_id)
    if after_clip_id:
        q = q.filter(models.Clip.id > after_clip_id)
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
        raise HTTPException(status_code=404, detail="Clip not found")
    return _clip_out(clip)


@app.post("/clips/{clip_id}/skip")
def skip_clip(clip_id: int, db: Session = Depends(get_db)):
    clip = db.query(models.Clip).filter(models.Clip.id == clip_id).first()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
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
        raise HTTPException(status_code=404, detail="Clip not found")

    existing = db.query(models.Label).filter(models.Label.clip_id == body.clip_id).first()
    if existing:
        existing.event_class = body.event_class
        existing.highlight_score = body.highlight_score
        existing.t_start_adjusted = body.t_start_adjusted
        existing.t_end_adjusted = body.t_end_adjusted
        existing.notes = body.notes
        existing.updated_at = datetime.utcnow()
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
    return [_label_out(l) for l in q.all()]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@app.post("/training/start")
def start_training():
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            return {"message": "Already running", "pid": pid}
        except (ProcessLookupError, ValueError, OSError):
            PID_FILE.unlink(missing_ok=True)

    script = Path(__file__).parent / "trainer.py"
    cmd = [
        "python", str(script),
        "--data_dir", str(CLIPS_DIR),
        "--output_dir", str(MODEL_DIR),
        "--epochs", "40",
        "--batch_size", "4",
        "--lr", "1e-3",
        "--device", "cuda",
    ]
    proc = subprocess.Popen(cmd, cwd=str(Path(__file__).parent))
    PID_FILE.write_text(str(proc.pid))
    return {"message": "Training started", "pid": proc.pid}


@app.post("/training/stop")
def stop_training():
    if not PID_FILE.exists():
        return {"message": "No training process found"}
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        PID_FILE.unlink(missing_ok=True)
        return {"message": "Stopped", "pid": pid}
    except (ProcessLookupError, ValueError, OSError):
        PID_FILE.unlink(missing_ok=True)
        return {"message": "Process not found, cleaned up"}


@app.get("/training/status", response_model=schemas.TrainingStatus)
def training_status():
    running = False
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            running = True
        except (ProcessLookupError, ValueError, OSError):
            PID_FILE.unlink(missing_ok=True)

    current_epoch = train_loss = val_loss = val_acc = None
    if TRAINING_LOG_PATH.exists():
        for line in reversed(TRAINING_LOG_PATH.read_text().splitlines()):
            if line.startswith("EPOCH"):
                parts = line.split()
                try:
                    current_epoch = int(parts[1])
                    train_loss = float(parts[3])
                    val_loss = float(parts[5])
                    val_acc = float(parts[7])
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


@app.get("/training/logs")
def training_logs():
    if not TRAINING_LOG_PATH.exists():
        return {"lines": []}
    lines = TRAINING_LOG_PATH.read_text().splitlines()
    return {"lines": lines[-100:]}


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


# ---------------------------------------------------------------------------
# Stats helper used by Export page
# ---------------------------------------------------------------------------

@app.get("/export/stats")
def export_stats(db: Session = Depends(get_db)):
    total = db.query(models.Clip).count()
    labeled = db.query(models.Clip).filter(models.Clip.status == "labeled").count()
    skipped = db.query(models.Clip).filter(models.Clip.status == "skipped").count()
    unlabeled = total - labeled - skipped

    class_counts: dict = {cls: 0 for cls in HIGHLIGHT_CLASSES}
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
        "total": total,
        "labeled": labeled,
        "skipped": skipped,
        "unlabeled": unlabeled,
        "class_counts": class_counts,
        "score_bins": bins,
    }
