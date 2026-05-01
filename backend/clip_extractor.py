import subprocess
import json
import logging
from pathlib import Path

from config import CLIPS_DIR, WINDOW_SIZES, CLIP_FPS, CLIP_RESOLUTION

logger = logging.getLogger(__name__)


def get_video_info(file_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-select_streams", "v:0",
        file_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise RuntimeError("ffprobe returned no video streams")

    stream = data["streams"][0]
    duration = float(stream.get("duration", 0) or stream.get("tags", {}).get("DURATION", 0))

    if not duration:
        # fallback: query container duration
        cmd2 = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            file_path,
        ]
        r2 = subprocess.run(cmd2, capture_output=True, text=True)
        if r2.returncode == 0:
            fmt = json.loads(r2.stdout).get("format", {})
            duration = float(fmt.get("duration", 0))

    fps_str = stream.get("r_frame_rate", "25/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 25.0

    return {"duration": duration, "fps": fps}


def extract_clip(input_path: str, output_path: str, t_start: float, duration: float):
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{t_start:.3f}",
        "-i", input_path,
        "-t", f"{duration:.3f}",
        "-vf", f"scale={CLIP_RESOLUTION}:{CLIP_RESOLUTION}",
        "-r", str(CLIP_FPS),
        "-c:v", "libx264",
        "-crf", "23",
        "-an",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed for {output_path}: {result.stderr[-500:]}")


def run_extraction(match_id: int):
    from database import SessionLocal
    from models import Match, Clip

    db = SessionLocal()
    try:
        match = db.query(Match).filter(Match.id == match_id).first()
        if not match:
            logger.error(f"Match {match_id} not found")
            return

        match.status = "extracting"
        db.commit()

        info = get_video_info(match.file_path)
        match.duration_seconds = info["duration"]
        match.fps = info["fps"]
        db.commit()

        duration = info["duration"]

        # Build clip job list across all window sizes
        clip_jobs = []
        for window_size in WINDOW_SIZES:
            stride = window_size // 2
            t = 0.0
            while t + window_size <= duration + 0.001:
                t_end = min(t + window_size, duration)
                clip_jobs.append((window_size, round(t, 3), round(t_end, 3)))
                t += stride

        # Pre-create output directories
        for window_size in WINDOW_SIZES:
            out_dir = CLIPS_DIR / match.name / f"{window_size}s"
            out_dir.mkdir(parents=True, exist_ok=True)

        for window_size, t_start, t_end in clip_jobs:
            out_dir = CLIPS_DIR / match.name / f"{window_size}s"
            clip_filename = f"clip_{t_start:.1f}.mp4"
            clip_path = out_dir / clip_filename

            try:
                extract_clip(match.file_path, str(clip_path), t_start, window_size)
                clip = Clip(
                    match_id=match.id,
                    clip_path=str(clip_path),
                    t_start=t_start,
                    t_end=t_end,
                    window_size=window_size,
                    status="unlabeled",
                )
                db.add(clip)
                db.commit()
            except Exception as exc:
                logger.error(f"Clip at {t_start}s (window={window_size}s) failed: {exc}")
                db.rollback()

        match.status = "ready"
        db.commit()

    except Exception as exc:
        logger.error(f"Extraction failed for match {match_id}: {exc}")
        try:
            match = db.query(Match).filter(Match.id == match_id).first()
            if match:
                match.status = "error"
                db.commit()
        except Exception:
            pass
    finally:
        db.close()
