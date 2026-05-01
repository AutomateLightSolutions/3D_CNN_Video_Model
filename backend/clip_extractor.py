import subprocess
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from config import BASE_DIR, CLIPS_DIR, WINDOW_CONFIG, CLIP_FPS, CLIP_RESOLUTION

# Local bundled binaries live here (populated by download_ffmpeg.py)
_BIN_DIR = Path(__file__).parent / "bin"


# --------------------------------------------------------------------------
# Tool resolution — local bin/ first, then system PATH
# --------------------------------------------------------------------------

def _tool(name: str) -> str:
    """Return the path to ffmpeg or ffprobe, preferring the local bin/ folder."""
    exe = name + (".exe" if sys.platform == "win32" else "")
    local = _BIN_DIR / exe
    if local.exists():
        return str(local)
    system = shutil.which(name)
    if system:
        return system
    raise RuntimeError(
        f"'{name}' not found in backend/bin/ or system PATH.\n"
        f"Run:  python backend/download_ffmpeg.py"
    )


# --------------------------------------------------------------------------
# Subprocess helper — forces UTF-8, prevents window pop-ups on Windows
# --------------------------------------------------------------------------

def _run(cmd: list) -> subprocess.CompletedProcess:
    kwargs = {
        "capture_output": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return subprocess.run(cmd, **kwargs)


# --------------------------------------------------------------------------
# Pre-flight checks
# --------------------------------------------------------------------------

def check_ffmpeg():
    """Raise RuntimeError with a helpful message if ffmpeg/ffprobe are missing."""
    for name in ("ffprobe", "ffmpeg"):
        _tool(name)   # raises with clear message if not found


# --------------------------------------------------------------------------
# Video inspection
# --------------------------------------------------------------------------

def get_video_info(file_path: str) -> dict:
    cmd = [
        _tool("ffprobe"), "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-select_streams", "v:0",
        "-show_format",
        file_path,
    ]
    r = _run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed (exit {r.returncode}): {r.stderr[:600]}")

    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ffprobe returned invalid JSON: {e}\nstdout={r.stdout[:300]}")

    # Duration — prefer stream, fallback to format container
    duration = 0.0
    streams = data.get("streams", [])
    if streams:
        duration = float(streams[0].get("duration") or 0)
    if not duration:
        duration = float(data.get("format", {}).get("duration") or 0)
    if not duration:
        raise RuntimeError("Could not determine video duration from ffprobe output.")

    # FPS
    fps = 25.0
    if streams:
        fps_str = streams[0].get("r_frame_rate", "25/1")
        try:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) else 25.0
        except Exception:
            pass

    return {"duration": duration, "fps": fps}


# --------------------------------------------------------------------------
# Single-clip extraction
# --------------------------------------------------------------------------

def extract_clip(input_path: str, output_path: str, t_start: float, window_size: int):
    cmd = [
        _tool("ffmpeg"), "-y",
        "-ss", f"{t_start:.3f}",
        "-i", input_path,
        "-t", f"{window_size:.3f}",
        "-vf", f"scale={CLIP_RESOLUTION}:{CLIP_RESOLUTION}:force_original_aspect_ratio=decrease,pad={CLIP_RESOLUTION}:{CLIP_RESOLUTION}:(ow-iw)/2:(oh-ih)/2",
        "-r", str(CLIP_FPS),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-an",
        str(output_path),
    ]
    r = _run(cmd)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-600:])


# --------------------------------------------------------------------------
# Per-match log helpers
# --------------------------------------------------------------------------

def _log_path(match_id: int) -> Path:
    return BASE_DIR / f"extraction_{match_id}.log"


def _make_logger(match_id: int):
    path = _log_path(match_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w", encoding="utf-8", buffering=1)

    def log(msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        fh.write(line + "\n")

    return log, fh


# --------------------------------------------------------------------------
# Extraction sub-steps (keep run_extraction complexity low)
# --------------------------------------------------------------------------

def _build_clip_jobs(duration: float) -> list:
    jobs = []
    for ws, cfg in WINDOW_CONFIG.items():
        stride = cfg["stride_s"]
        t = 0.0
        while t + ws <= duration + 0.01:
            jobs.append((ws, round(t, 3)))
            t = round(t + stride, 3)
    return jobs


def _extract_one(db, match, clip_path, t_start, ws, duration, log):
    """Extract a single clip and insert into DB. Returns True on success."""
    from models import Clip
    try:
        extract_clip(match.file_path, str(clip_path), t_start, ws)
        t_end = min(round(t_start + ws, 3), duration)
        db.add(Clip(
            match_id=match.id,
            clip_path=str(clip_path),
            t_start=t_start,
            t_end=t_end,
            window_size=ws,
            status="unlabeled",
        ))
        db.commit()
        return True
    except Exception as exc:
        log(f"WARN [{ws}s @ {t_start:.1f}s] {exc}")
        db.rollback()
        return False


def _run_clip_loop(db, match, clip_jobs, duration, log):
    """Iterate all clip jobs and return (done, errors) counts."""
    total = len(clip_jobs)
    done = errors = 0
    for ws, t_start in clip_jobs:
        out_dir = CLIPS_DIR / match.name / f"{ws}s"
        clip_path = out_dir / f"clip_{t_start:.1f}.mp4"
        if _extract_one(db, match, clip_path, t_start, ws, duration, log):
            done += 1
        else:
            errors += 1
        if (done + errors) % 10 == 0 or (done + errors) == total:
            pct = int((done + errors) / total * 100)
            log(f"Progress: {done + errors}/{total} clips ({pct}%)  ok={done}  errors={errors}")
    return done, errors


# --------------------------------------------------------------------------
# Main extraction task  (called as FastAPI BackgroundTask)
# --------------------------------------------------------------------------

def run_extraction(match_id: int):
    from database import SessionLocal
    from models import Match

    log, fh = _make_logger(match_id)
    db = SessionLocal()
    try:
        _do_extraction(match_id, db, log)
    except Exception as exc:
        log(f"FATAL: {exc}")
        _mark_error(match_id, db, log)
    finally:
        fh.close()
        db.close()


def _do_extraction(match_id: int, db, log):
    from models import Match

    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        log(f"ERROR: Match {match_id} not found.")
        return

    log(f"Starting extraction for: {match.name}")
    log(f"Source: {match.file_path}")

    try:
        check_ffmpeg()
        log("ffmpeg + ffprobe found in PATH.")
    except RuntimeError as e:
        log(f"ERROR: {e}")
        match.status = "error"
        db.commit()
        return

    match.status = "extracting"
    db.commit()

    log("Running ffprobe…")
    try:
        info = get_video_info(match.file_path)
    except RuntimeError as e:
        log(f"ERROR: {e}")
        match.status = "error"
        db.commit()
        return

    match.duration_seconds = info["duration"]
    match.fps = info["fps"]
    db.commit()
    log(f"Duration: {info['duration']:.2f}s  FPS: {info['fps']:.2f}")

    clip_jobs = _build_clip_jobs(info["duration"])
    log(f"Planned {len(clip_jobs)} clips  (window sizes: {list(WINDOW_CONFIG.keys())}s)")

    for ws in WINDOW_CONFIG:
        (CLIPS_DIR / match.name / f"{ws}s").mkdir(parents=True, exist_ok=True)

    done, errors = _run_clip_loop(db, match, clip_jobs, info["duration"], log)

    if errors == len(clip_jobs) and clip_jobs:
        log("ERROR: All clips failed — check the WARN lines above.")
        match.status = "error"
    else:
        match.status = "ready"
        log(f"Done. {done} clips extracted, {errors} skipped.")
    db.commit()


def _mark_error(match_id: int, db, log):
    from models import Match
    try:
        m = db.query(Match).filter(Match.id == match_id).first()
        if m:
            m.status = "error"
            db.commit()
    except Exception as e:
        log(f"Could not mark match as error: {e}")
