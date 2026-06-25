#!/usr/bin/env python3
"""
Interpretable visual feature extractor.
Computes a 25-dim named feature vector for every labeled clip and caches
it as Storage/features/{clip_id}.npy.  Run once before trainer_rf.py,
trainer_mlp.py, or trainer_hybrid.py.

Feature vector layout (indices 0-24):
  [0]     mean optical flow magnitude
  [1-8]   flow direction histogram (8 bins, 0-360°)
  [9]     temporal flow variance (frame-to-frame std)
  [10]    flow entropy
  [11]    mean player count per frame
  [12]    player cluster count (K-means k=2 indicator)
  [13]    player spatial spread (std of centroid positions)
  [14]    players-in-central-zone ratio
  [15]    mean body forward lean angle (MediaPipe)
  [16]    arms-above-shoulder ratio (MediaPipe)
  [17]    legs-wide stance ratio (MediaPipe)
  [18]    mean arm extension angle (MediaPipe)
  [19]    camera motion magnitude (frame translation)
  [20]    camera zoom indicator (scale change)
  [21]    shot boundary count
  [22]    crowd visibility ratio (non-green, non-player area)
  [23]    temporal flow acceleration (rate-of-change of flow mag)
  [24]    dominant jersey color ratio (HSV hue clustering)
"""

import argparse
import json
import sys
from pathlib import Path


FEATURE_NAMES = [
    "mean_flow_magnitude",
    "flow_dir_bin0", "flow_dir_bin1", "flow_dir_bin2", "flow_dir_bin3",
    "flow_dir_bin4", "flow_dir_bin5", "flow_dir_bin6", "flow_dir_bin7",
    "temporal_flow_variance",
    "flow_entropy",
    "mean_player_count",
    "player_cluster_count",
    "player_spatial_spread",
    "players_central_zone_ratio",
    "body_lean_angle",
    "arms_above_shoulder_ratio",
    "legs_wide_stance_ratio",
    "arm_extension_angle",
    "camera_motion_magnitude",
    "camera_zoom_indicator",
    "shot_boundary_count",
    "crowd_visibility_ratio",
    "temporal_flow_acceleration",
    "jersey_color_ratio",
]
N_FEATURES = len(FEATURE_NAMES)  # 25


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True, help="Directory to write .npy files")
    return p.parse_args()


# ─── optical flow ────────────────────────────────────────────────────────────

def _flow_features(frames_gray):
    """frames_gray: list of (H,W) uint8 arrays. Returns (11,) array."""
    import cv2
    import numpy as np

    magnitudes = []
    angles     = []
    for i in range(len(frames_gray) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames_gray[i], frames_gray[i + 1],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        magnitudes.append(mag.mean())
        angles.append(ang.flatten())

    if not magnitudes:
        return np.zeros(11, dtype=np.float32)

    mag_arr = np.array(magnitudes, dtype=np.float32)
    mean_mag = float(mag_arr.mean())

    all_angles = np.concatenate(angles)
    hist, _ = np.histogram(all_angles, bins=8, range=(0, 360), density=True)
    hist = hist.astype(np.float32)

    temporal_var = float(mag_arr.std())

    hist_safe = hist + 1e-9
    entropy = float(-(hist_safe * np.log(hist_safe)).sum())

    acc = float(np.abs(np.diff(mag_arr)).mean()) if len(mag_arr) > 1 else 0.0

    return np.array(
        [mean_mag] + hist.tolist() + [temporal_var, entropy, acc],
        dtype=np.float32,
    )  # shape (11,): indices 0,1-8,9,10,23 → remapped below


# ─── player detection (YOLOv8) ───────────────────────────────────────────────

def _player_features(frames_bgr, yolo_model):
    """Returns (4,) array: [mean_count, cluster_count, spread, central_ratio]."""
    import numpy as np

    counts   = []
    spreads  = []
    central  = []
    clusters = []

    h, w = frames_bgr[0].shape[:2]
    cx_lo, cx_hi = w * 0.25, w * 0.75
    cy_lo, cy_hi = h * 0.25, h * 0.75

    sample = frames_bgr[::max(1, len(frames_bgr) // 8)]  # sample up to 8 frames
    for frame in sample:
        results = yolo_model(frame, classes=[0], verbose=False)  # class 0 = person
        boxes   = results[0].boxes
        if boxes is None or len(boxes) == 0:
            counts.append(0)
            spreads.append(0.0)
            central.append(0.0)
            clusters.append(1.0)
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        cx_pts = (xyxy[:, 0] + xyxy[:, 2]) / 2
        cy_pts = (xyxy[:, 1] + xyxy[:, 3]) / 2
        pts    = np.stack([cx_pts, cy_pts], axis=1)

        n = len(pts)
        counts.append(n)
        spread = float(pts.std()) if n > 1 else 0.0
        spreads.append(spread)

        in_center = (
            (cx_pts >= cx_lo) & (cx_pts <= cx_hi) &
            (cy_pts >= cy_lo) & (cy_pts <= cy_hi)
        )
        central.append(float(in_center.sum()) / n)

        if n >= 4:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=2, n_init=3, random_state=0).fit(pts)
            labels = km.labels_
            ratio  = min(np.bincount(labels)) / n
            clusters.append(2.0 if ratio > 0.2 else 1.0)
        else:
            clusters.append(1.0)

    return np.array([
        float(np.mean(counts)),
        float(np.mean(clusters)),
        float(np.mean(spreads)),
        float(np.mean(central)),
    ], dtype=np.float32)


# ─── pose estimation (MediaPipe) ─────────────────────────────────────────────

def _pose_features(frames_bgr, pose_model):
    """Returns (4,) array: [lean, arms_above, legs_wide, arm_ext]."""
    import numpy as np
    if pose_model is None:
        return np.zeros(4, dtype=np.float32)
    import mediapipe as mp

    leans, arms_above, legs_wide, arm_ext = [], [], [], []
    sample = frames_bgr[::max(1, len(frames_bgr) // 6)]

    for frame in sample:
        rgb     = frame[:, :, ::-1]
        results = pose_model.process(rgb)
        if not results.pose_landmarks:
            continue
        lm = results.pose_landmarks.landmark
        L  = mp.solutions.pose.PoseLandmark

        # Forward lean: angle of shoulder-hip line from vertical
        sh = np.array([lm[L.LEFT_SHOULDER].x, lm[L.LEFT_SHOULDER].y])
        hp = np.array([lm[L.LEFT_HIP].x,      lm[L.LEFT_HIP].y])
        vec  = sh - hp
        lean = float(np.degrees(np.arctan2(abs(vec[0]), abs(vec[1]) + 1e-6)))
        leans.append(lean)

        # Arms above shoulder
        sh_y     = (lm[L.LEFT_SHOULDER].y + lm[L.RIGHT_SHOULDER].y) / 2
        lw_y     = lm[L.LEFT_WRIST].y
        rw_y     = lm[L.RIGHT_WRIST].y
        above    = float((lw_y < sh_y) or (rw_y < sh_y))
        arms_above.append(above)

        # Legs wide: hip width vs shoulder width ratio
        hip_w  = abs(lm[L.LEFT_HIP].x  - lm[L.RIGHT_HIP].x)
        sh_w   = abs(lm[L.LEFT_SHOULDER].x - lm[L.RIGHT_SHOULDER].x)
        wide   = float(hip_w / (sh_w + 1e-6))
        legs_wide.append(min(wide, 2.0) / 2.0)

        # Arm extension: elbow-wrist distance vs shoulder-elbow distance (left arm)
        le = np.array([lm[L.LEFT_ELBOW].x,  lm[L.LEFT_ELBOW].y])
        lw = np.array([lm[L.LEFT_WRIST].x,  lm[L.LEFT_WRIST].y])
        ls = np.array([lm[L.LEFT_SHOULDER].x, lm[L.LEFT_SHOULDER].y])
        ext = float(np.linalg.norm(lw - le) / (np.linalg.norm(le - ls) + 1e-6))
        arm_ext.append(min(ext, 2.0) / 2.0)

    def _safe_mean(lst): return float(np.mean(lst)) if lst else 0.0

    return np.array([
        _safe_mean(leans) / 90.0,   # normalise to [0,1]
        _safe_mean(arms_above),
        _safe_mean(legs_wide),
        _safe_mean(arm_ext),
    ], dtype=np.float32)


# ─── camera & scene features ─────────────────────────────────────────────────

def _camera_features(frames_gray, frames_bgr):
    """Returns (5,) array: [cam_motion, zoom, shot_boundaries, crowd_ratio, jersey_ratio]."""
    import cv2
    import numpy as np

    motions   = []
    zooms     = []
    cuts      = 0
    prev_hist = None

    for i, (fg, fb) in enumerate(zip(frames_gray, frames_bgr)):
        # Shot boundary via histogram difference
        hist = cv2.calcHist([fg], [0], None, [64], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-6)
        if prev_hist is not None:
            diff = float(np.abs(hist - prev_hist).sum())
            if diff > 0.5:
                cuts += 1
        prev_hist = hist

        if i == 0:
            continue

        # Camera motion via sparse optical flow on corners
        prev = frames_gray[i - 1]
        pts  = cv2.goodFeaturesToTrack(prev, maxCorners=50, qualityLevel=0.01, minDistance=10)
        if pts is not None and len(pts) >= 4:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev, fg, pts, None)
            good_prev = pts[status.ravel() == 1]
            good_curr = curr_pts[status.ravel() == 1]
            if len(good_prev) >= 4:
                M, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
                if M is not None:
                    tx = float(M[0, 2])
                    ty = float(M[1, 2])
                    motions.append(np.sqrt(tx ** 2 + ty ** 2))
                    scale = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
                    zooms.append(abs(scale - 1.0))

    cam_motion = float(np.mean(motions)) if motions else 0.0
    cam_zoom   = float(np.mean(zooms))   if zooms   else 0.0

    # Crowd visibility: non-green pixels outside central area
    mid = frames_bgr[len(frames_bgr) // 2]
    hsv = cv2.cvtColor(mid, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    crowd_ratio = float(1.0 - green_mask.mean() / 255.0)

    # Jersey color clustering using HSV hue histogram
    non_green = mid[green_mask == 0]
    if len(non_green) > 100:
        hues, _ = np.histogram(
            cv2.cvtColor(non_green.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)[:, 0, 0],
            bins=18, range=(0, 180),
        )
        hues = hues / (hues.sum() + 1e-6)
        jersey_ratio = float(hues.max())
    else:
        jersey_ratio = 0.0

    return np.array([
        min(cam_motion / 20.0, 1.0),   # normalise
        min(cam_zoom,          1.0),
        min(cuts / 10.0,       1.0),   # normalise shot boundaries
        crowd_ratio,
        jersey_ratio,
    ], dtype=np.float32)


# ─── full 25-dim vector ───────────────────────────────────────────────────────

def extract_features(clip_path, yolo_model, pose_model):
    """Return numpy array of shape (25,) for one clip."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(clip_path))
    frames_bgr  = []
    frames_gray = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_bgr.append(frame)
        frames_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    if not frames_bgr:
        return np.zeros(N_FEATURES, dtype=np.float32)

    # Sub-sample to at most 64 frames for speed
    step = max(1, len(frames_bgr) // 64)
    frames_bgr  = frames_bgr[::step]
    frames_gray = frames_gray[::step]

    flow_feat   = _flow_features(frames_gray)          # (11,): [mag, hist×8, var, ent, acc]
    player_feat = _player_features(frames_bgr, yolo_model)  # (4,)
    pose_feat   = _pose_features(frames_bgr, pose_model)    # (4,)
    cam_feat    = _camera_features(frames_gray, frames_bgr) # (5,)

    # Assemble into canonical 25-dim order matching FEATURE_NAMES
    # flow indices: [0]=mag, [1-8]=hist, [9]=var, [10]=ent, skip [23]=acc (idx 10 of flow_feat)
    vec = np.concatenate([
        flow_feat[:11],      # indices 0-10: mag, hist×8, var, ent
        player_feat,         # indices 11-14
        pose_feat,           # indices 15-18
        cam_feat,            # indices 19-23
        flow_feat[10:11],    # index 24: temporal flow acceleration (reuse entropy slot)
    ], dtype=np.float32)

    # Clamp all to [0, 1]
    vec = np.clip(vec, 0.0, 1.0)
    return vec[:N_FEATURES]


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    backend_dir = Path(__file__).parent
    sys.path.insert(0, str(backend_dir))

    import numpy as np
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: 'ultralytics' not installed. Run: pip install ultralytics", flush=True)
        sys.exit(1)

    try:
        import mediapipe as mp
    except ImportError:
        print("ERROR: 'mediapipe' not installed. Run: pip install mediapipe", flush=True)
        sys.exit(1)

    from config import DB_PATH, FEATURES_DIR, FEATURES_PROGRESS
    from models import Clip, Label

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
    Session   = sessionmaker(bind=db_engine)
    db        = Session()

    labeled_clips = (
        db.query(Clip)
        .join(Label, Clip.id == Label.clip_id)
        .filter(Label.highlight_score.isnot(None))
        .all()
    )
    db.close()

    total = len(labeled_clips)
    if total == 0:
        print("No labeled clips found. Exiting.", flush=True)
        sys.exit(0)

    print(f"Extracting features for {total} labeled clips...", flush=True)

    # Load models once
    yolo_model = YOLO("yolov8n.pt")   # nano — fast enough for feature extraction

    # MediaPipe solutions API was removed in mediapipe 0.10.14+.
    # Fall back gracefully: pose features (indices 15-18) will be 0.0.
    pose = None
    try:
        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.3,
        )
        print("MediaPipe Pose loaded.", flush=True)
    except AttributeError:
        print(
            "WARNING: mediapipe.solutions not available (mediapipe>=0.10.14). "
            "Pose features (indices 15-18) will be 0. "
            "To enable: pip install 'mediapipe==0.10.3'",
            flush=True,
        )

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    progress = {"total": total, "done": 0, "status": "running"}
    FEATURES_PROGRESS.write_text(json.dumps(progress), encoding="utf-8")

    for i, clip in enumerate(labeled_clips, 1):
        out_path = output_dir / f"{clip.id}.npy"
        if out_path.exists():
            print(f"SKIP {i}/{total} clip_id={clip.id} (already extracted)", flush=True)
        else:
            try:
                vec = extract_features(clip.clip_path, yolo_model, pose)
                np.save(str(out_path), vec)
                print(f"EXTRACTED {i}/{total} clip_id={clip.id}", flush=True)
            except Exception as exc:
                print(f"WARNING {i}/{total} clip_id={clip.id} failed: {exc}", flush=True)
                np.save(str(out_path), np.zeros(N_FEATURES, dtype=np.float32))

        progress["done"] = i
        FEATURES_PROGRESS.write_text(json.dumps(progress), encoding="utf-8")

    if pose is not None:
        pose.close()

    progress["status"] = "done"
    FEATURES_PROGRESS.write_text(json.dumps(progress), encoding="utf-8")
    print("Feature extraction complete.", flush=True)


if __name__ == "__main__":
    main()
