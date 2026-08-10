#!/usr/bin/env python3
"""Full-match multi-window prediction ("Predict" feature).

Given a registered Match and a chosen trained model_type, tiles the whole
match into non-overlapping 8s segments (TILE_SIZE), extracts an aligned
8s/16s/32s clip per tile (16s/32s centered on the tile, clamped at video
boundaries), runs the chosen model on all 3, and merges them into one
(event_class, highlight_score) per tile:

  - Score merge: fixed-weight average favoring the 8s window (MERGE_WEIGHTS).
  - Class merge: masked weighted vote using the same MERGE_WEIGHTS. Different
    window sizes have training-label coverage for different event classes
    (e.g. scrum only ever
    labeled at 16s) — class_support_sets() computes, from the live DB, which
    classes each window size actually has enough labeled examples for, and
    merge_class() zeroes out a window's vote for any class outside its
    support set before combining. This stops a window from ever "voting"
    for a class it has no training signal for.

Raw per-window predictions are kept (PredictionWindowResult) alongside the
merged result (PredictionSegment) for debugging/audit.

Mirrors clip_extractor.run_extraction's structure: a BackgroundTask
entrypoint with its own DB session, its own log file, and a progress.json
the status route polls.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import model_defs
import models
from clip_extractor import extract_clip, get_video_info
from config import (
    CLASS_SUPPORT_MIN_COUNT, HIGHLIGHT_CLASSES,
    HYBRID_MODEL_DIR, MERGE_WEIGHTS, MLP_MODEL_DIR, PREDICTIONS_DIR, R3D_MODEL_DIR,
    RF_MODEL_DIR, SLOWFAST_MODEL_DIR, TILE_SIZE,
    VIDEOMAE_MODEL_DIR, WINDOW_SIZES,
)


# ---------------------------------------------------------------------------
# Class support sets (dynamic, from live DB — not hardcoded)
# ---------------------------------------------------------------------------

def class_support_sets(db, floor: int = CLASS_SUPPORT_MIN_COUNT) -> dict:
    """{window_size: set(event_class, ...)} — which classes each window size
    has >= floor labeled examples for, right now, in the live DB."""
    from sqlalchemy import func

    rows = (
        db.query(models.Clip.window_size, models.Label.event_class, func.count(models.Label.id))
        .join(models.Label, models.Clip.id == models.Label.clip_id)
        .group_by(models.Clip.window_size, models.Label.event_class)
        .having(func.count(models.Label.id) >= floor)
        .all()
    )
    result = {ws: set() for ws in WINDOW_SIZES}
    for ws, event_class, _count in rows:
        result.setdefault(ws, set()).add(event_class)
    return result


def available_models() -> dict:
    """Per model_type: whether a usable checkpoint exists on disk."""
    return {
        "r3d": (R3D_MODEL_DIR / "best_model.pt").exists(),
        "videomae": (VIDEOMAE_MODEL_DIR / "best_model.pt").exists(),
        "slowfast": (SLOWFAST_MODEL_DIR / "best_model.pt").exists(),
        "rf": (RF_MODEL_DIR / "best_model.pkl").exists(),
        "mlp": (MLP_MODEL_DIR / "best_model.pt").exists(),
        "hybrid": (HYBRID_MODEL_DIR / "best_model.pt").exists(),
    }


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def build_tiles(duration: float, tile_size: int = TILE_SIZE) -> list:
    """Non-overlapping tiles covering [0, duration). Last tile is clamped short."""
    tiles = []
    t = 0.0
    idx = 0
    while t < duration - 1e-6:
        t_end = min(round(t + tile_size, 3), round(duration, 3))
        tiles.append({"tile_index": idx, "t_start": round(t, 3), "t_end": t_end})
        idx += 1
        t = round(t + tile_size, 3)
    return tiles


def tile_context_window(tile_start: float, tile_end: float, window_size: int, duration: float):
    """16s/32s clip centered on the tile, clamped (not shifted) at [0, duration] —
    a boundary tile's context window can end up shorter than the nominal size."""
    center = (tile_start + tile_end) / 2.0
    half = window_size / 2.0
    start = max(0.0, center - half)
    end = min(duration, center + half)
    return round(start, 3), round(end, 3)


def _extract_tile_clips(match_file_path: str, run_dir: Path, tile: dict, duration: float) -> dict:
    """Extract the 3 aligned clips for one tile. Returns
    {window_size: (clip_path, t_start, t_end)}."""
    paths = {}
    idx = tile["tile_index"]

    t_start, t_end = tile["t_start"], tile["t_end"]
    p8 = run_dir / f"tile_{idx:05d}_8s.mp4"
    extract_clip(match_file_path, str(p8), t_start, t_end - t_start)
    paths[8] = (p8, t_start, t_end)

    for ws in (16, 32):
        cstart, cend = tile_context_window(t_start, t_end, ws, duration)
        p = run_dir / f"tile_{idx:05d}_{ws}s.mp4"
        extract_clip(match_file_path, str(p), cstart, cend - cstart)
        paths[ws] = (p, cstart, cend)

    return paths


# ---------------------------------------------------------------------------
# Model loading — each called once per PredictionRun, not per clip
# ---------------------------------------------------------------------------

def _load_checkpoint_model(model_cls_factory, ckpt_path: Path, device, wrapped: bool):
    if not ckpt_path.exists():
        raise RuntimeError(f"No trained checkpoint found at {ckpt_path} — train this model first.")
    model = model_cls_factory()
    state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"] if wrapped else state)
    return model.to(device).eval()


def load_r3d(device):
    return _load_checkpoint_model(
        lambda: model_defs.R3DHighlightModel(len(HIGHLIGHT_CLASSES)),
        R3D_MODEL_DIR / "best_model.pt", device, wrapped=True,
    )


def load_videomae(device):
    return _load_checkpoint_model(
        lambda: model_defs.VideoMAEHighlightModel(len(HIGHLIGHT_CLASSES)),
        VIDEOMAE_MODEL_DIR / "best_model.pt", device, wrapped=True,
    )


def load_slowfast(device):
    return _load_checkpoint_model(
        lambda: model_defs.SlowFastHighlightModel(len(HIGHLIGHT_CLASSES)),
        SLOWFAST_MODEL_DIR / "best_model.pt", device, wrapped=True,
    )


def load_mlp(device):
    return _load_checkpoint_model(
        lambda: model_defs.InterpMLP(25, len(HIGHLIGHT_CLASSES)),
        MLP_MODEL_DIR / "best_model.pt", device, wrapped=False,
    )


def load_rf():
    import joblib
    ckpt_path = RF_MODEL_DIR / "best_model.pkl"
    if not ckpt_path.exists():
        raise RuntimeError(f"No trained checkpoint found at {ckpt_path} — train RF first.")
    return joblib.load(str(ckpt_path))


def load_hybrid(db, device):
    import training_history
    from config import BASE_DIR
    from trainer_hybrid import load_backbone

    latest = training_history.latest_completed_per_model(db)
    hybrid_run = latest.get("hybrid")
    if hybrid_run is None or not hybrid_run.backbone:
        raise RuntimeError("No completed Hybrid training run found — train Hybrid first.")
    backbone_name = hybrid_run.backbone

    backbone_model, transform_fn, feat_dim = load_backbone(backbone_name, BASE_DIR, device)

    ckpt_path = HYBRID_MODEL_DIR / "best_model.pt"
    if not ckpt_path.exists():
        raise RuntimeError(f"No trained checkpoint found at {ckpt_path} — train Hybrid first.")
    fusion = model_defs.HybridFusion(25, feat_dim, len(HIGHLIGHT_CLASSES))
    state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    fusion.load_state_dict(state)
    fusion = fusion.to(device).eval()

    return {"backbone": backbone_model, "transform_fn": transform_fn,
            "fusion": fusion, "backbone_name": backbone_name}


def _load_feature_extractors():
    """YOLO + MediaPipe Pose, loaded once and reused per clip — same
    principle as feature_extractor.py's own one-time load in main()."""
    from ultralytics import YOLO
    import mediapipe as mp

    yolo_model = YOLO("yolov8n.pt")
    pose = None
    try:
        pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=0, min_detection_confidence=0.3,
        )
    except AttributeError:
        pose = None  # mediapipe>=0.10.14 removed the solutions API; pose features become 0
    return yolo_model, pose


def load_model_bundle(model_type: str, db, device) -> dict:
    if model_type == "r3d":
        return {"kind": "r3d", "model": load_r3d(device)}
    if model_type == "videomae":
        return {"kind": "videomae", "model": load_videomae(device)}
    if model_type == "slowfast":
        return {"kind": "slowfast", "model": load_slowfast(device)}
    if model_type == "rf":
        yolo_model, pose_model = _load_feature_extractors()
        return {"kind": "rf", "yolo": yolo_model, "pose": pose_model, "rf": load_rf()}
    if model_type == "mlp":
        yolo_model, pose_model = _load_feature_extractors()
        return {"kind": "mlp", "yolo": yolo_model, "pose": pose_model, "mlp": load_mlp(device)}
    if model_type == "hybrid":
        yolo_model, pose_model = _load_feature_extractors()
        bundle = {"kind": "hybrid", "yolo": yolo_model, "pose": pose_model}
        bundle.update(load_hybrid(db, device))
        return bundle
    raise ValueError(f"Unknown model_type: {model_type}")


# ---------------------------------------------------------------------------
# Per-window forward pass
# ---------------------------------------------------------------------------

def predict_window(model_type: str, bundle: dict, clip_path: str, window_size_s: int, device):
    """Returns (probs: np.ndarray[10], score: float) for one clip. Trainers
    only ever store argmax — softmax is computed explicitly here."""
    kind = bundle["kind"]

    if kind == "r3d":
        x = model_defs.preprocess_r3d_clip(clip_path, window_size_s).to(device)
        with torch.no_grad():
            logits, score = bundle["model"](x)
        return F.softmax(logits, dim=1)[0].cpu().numpy(), float(score.item())

    if kind == "videomae":
        x = model_defs.preprocess_videomae_clip(clip_path).to(device)
        with torch.no_grad():
            logits, score = bundle["model"](x)
        return F.softmax(logits, dim=1)[0].cpu().numpy(), float(score.item())

    if kind == "slowfast":
        slow, fast = model_defs.preprocess_slowfast_clip(clip_path)
        slow, fast = slow.to(device), fast.to(device)
        with torch.no_grad():
            logits, score = bundle["model"](slow, fast)
        return F.softmax(logits, dim=1)[0].cpu().numpy(), float(score.item())

    if kind in ("rf", "mlp", "hybrid"):
        import feature_extractor
        feat = feature_extractor.extract_features(clip_path, bundle["yolo"], bundle["pose"])  # (25,)

        if kind == "rf":
            clf = bundle["rf"]["classifier"]
            reg = bundle["rf"]["regressor"]
            raw_probs = clf.predict_proba(feat.reshape(1, -1))[0]  # ordered by clf.classes_
            probs = np.zeros(len(HIGHLIGHT_CLASSES), dtype=np.float32)
            for i, cls_int in enumerate(clf.classes_):
                probs[cls_int] = raw_probs[i]
            score = float(np.clip(reg.predict(feat.reshape(1, -1))[0], 0.0, 1.0))
            return probs, score

        if kind == "mlp":
            x = torch.from_numpy(feat).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, score = bundle["mlp"](x)
            return F.softmax(logits, dim=1)[0].cpu().numpy(), float(score.item())

        # hybrid — deep embedding from the frozen backbone + the 25-dim features
        video = model_defs.read_video_compat(clip_path)
        inp = bundle["transform_fn"](video)
        with torch.no_grad():
            inp = [t.to(device) for t in inp] if isinstance(inp, list) else inp.to(device)
            deep_feat = bundle["backbone"](inp)  # already batched (1, feat_dim)
            interp = torch.from_numpy(feat).unsqueeze(0).to(device)
            logits, score = bundle["fusion"](interp, deep_feat)
        return F.softmax(logits, dim=1)[0].cpu().numpy(), float(score.item())

    raise ValueError(f"Unknown model kind: {kind}")


# ---------------------------------------------------------------------------
# Merges
# ---------------------------------------------------------------------------

def merge_score(scores: dict, weights: dict = MERGE_WEIGHTS) -> float:
    """Weighted average across whichever windows produced a score, renormalized."""
    avail = {ws: s for ws, s in scores.items() if s is not None}
    if not avail:
        return 0.0
    total_w = sum(weights.get(ws, 0.0) for ws in avail)
    if total_w <= 0:
        return float(np.mean(list(avail.values())))
    return float(sum(weights.get(ws, 0.0) * s for ws, s in avail.items()) / total_w)


def merge_class(class_probs: dict, support_sets: dict, weights: dict = MERGE_WEIGHTS,
                 classes: list = HIGHLIGHT_CLASSES):
    """Masked weighted vote: each window's softmax is zeroed outside its own
    class support set and renormalized before being combined, so a window
    can never vote for a class it has no training signal for. A window
    whose masked mass is ~0 is dropped and the remaining weights renormalized.
    If every window gets dropped, falls back to a straight unmasked average
    so the tile still gets a label."""
    n = len(classes)
    combined = np.zeros(n, dtype=np.float64)
    total_w = 0.0

    for ws, probs in class_probs.items():
        support = support_sets.get(ws, set())
        mask = np.array([1.0 if c in support else 0.0 for c in classes])
        masked = np.asarray(probs, dtype=np.float64) * mask
        mass = masked.sum()
        if mass <= 1e-9:
            continue
        w = weights.get(ws, 0.0)
        combined += w * (masked / mass)
        total_w += w

    if total_w <= 0:
        combined = np.mean([np.asarray(p, dtype=np.float64) for p in class_probs.values()], axis=0)
    else:
        combined = combined / total_w

    best_idx = int(np.argmax(combined))
    return classes[best_idx], combined.tolist()


# ---------------------------------------------------------------------------
# Merge-weight calibration (admin) — ground-truth CSV ingestion + fast re-merge
# ---------------------------------------------------------------------------

def ingest_ground_truth_csv(db, match_id: int, csv_text: str) -> int:
    """Parse a Start/End/Text/Event/Score CSV (dense, overlapping caption-
    level rows spanning a whole match) and aggregate it onto this match's
    non-overlapping TILE_SIZE tile grid — the same grid build_tiles()
    produces for any PredictionRun against this match, so it lines up with
    PredictionSegment.tile_index regardless of which model/run is being
    evaluated.

    For each tile, every CSV row overlapping it contributes its Event/Score
    weighted by the overlap duration; the tile's ground-truth event is the
    overlap-weighted majority class, and its score the overlap-weighted mean.
    A tile with no overlapping CSV row is left out entirely (no row written)
    rather than guessed.

    One CSV per match: this replaces any previously ingested ground truth
    for match_id. Returns the number of tiles written.
    """
    import csv as csv_mod
    import io

    match = db.query(models.Match).filter(models.Match.id == match_id).first()
    if not match:
        raise ValueError(f"Match {match_id} not found")
    if not match.duration_seconds:
        raise ValueError(f"Match {match_id} has no known duration")

    rows = []
    reader = csv_mod.DictReader(io.StringIO(csv_text))
    for r in reader:
        try:
            start, end = float(r["Start"]), float(r["End"])
            event, score = r["Event"].strip(), float(r["Score"])
        except (KeyError, ValueError):
            continue
        if end > start and event:
            rows.append((start, end, event, score))
    if not rows:
        raise ValueError("No usable rows found in CSV (expected columns: Start,End,Event,Score)")

    tiles = build_tiles(match.duration_seconds)

    db.query(models.GroundTruthSegment).filter(models.GroundTruthSegment.match_id == match_id).delete()

    n_written = 0
    for tile in tiles:
        t_start, t_end = tile["t_start"], tile["t_end"]
        class_weight = {}
        score_weight_sum = 0.0
        score_sum = 0.0
        for start, end, event, score in rows:
            overlap = min(t_end, end) - max(t_start, start)
            if overlap <= 0:
                continue
            class_weight[event] = class_weight.get(event, 0.0) + overlap
            score_weight_sum += overlap
            score_sum += overlap * score
        if not class_weight:
            continue
        best_event = max(class_weight.items(), key=lambda kv: kv[1])[0]
        mean_score = score_sum / score_weight_sum if score_weight_sum > 0 else 0.0
        db.add(models.GroundTruthSegment(
            match_id=match_id, tile_index=tile["tile_index"],
            t_start=t_start, t_end=t_end,
            event_class=best_event, score=round(mean_score, 4),
        ))
        n_written += 1

    db.commit()
    return n_written


def ground_truth_status(db, match_id: int) -> dict:
    n = (
        db.query(models.GroundTruthSegment)
        .filter(models.GroundTruthSegment.match_id == match_id)
        .count()
    )
    return {"uploaded": n > 0, "n_tiles": n}


def _load_eval_tiles(db, run_id: int) -> list:
    """Every tile of a PredictionRun that has ground truth, pre-loaded with
    its cached per-window predictions and true (event_class, score) — the
    weight-independent part of scoring, computed once and reused across
    however many weight configs get tried against this run."""
    run = db.query(models.PredictionRun).filter(models.PredictionRun.id == run_id).first()
    if not run:
        raise ValueError(f"PredictionRun {run_id} not found")

    gt_rows = (
        db.query(models.GroundTruthSegment)
        .filter(models.GroundTruthSegment.match_id == run.match_id)
        .all()
    )
    if not gt_rows:
        raise ValueError("No ground truth uploaded for this match yet")
    gt_by_tile = {g.tile_index: (g.event_class, g.score) for g in gt_rows}

    segments = (
        db.query(models.PredictionSegment)
        .filter(models.PredictionSegment.run_id == run_id)
        .all()
    )

    tiles = []
    for seg in segments:
        gt = gt_by_tile.get(seg.tile_index)
        if gt is None:
            continue
        true_class, true_score = gt

        window_probs, window_scores = {}, {}
        for win in seg.windows:
            window_probs[win.window_size] = np.asarray(json.loads(win.class_probs_json), dtype=np.float64)
            window_scores[win.window_size] = win.highlight_score
        if not window_probs:
            continue

        tiles.append((window_probs, window_scores, true_class, true_score))

    if not tiles:
        raise ValueError("No tiles overlap between this run's segments and the uploaded ground truth")
    return tiles


def _score_weights(tiles: list, support_sets: dict, weights: dict) -> dict:
    """Merge every pre-loaded tile under one weight dict and score against
    its ground truth. Pure in-memory — no DB access — so this is cheap
    enough to call once per grid point in evaluate_all_weights()."""
    from training_common import class_int_for, compute_full_metrics
    from config import HIGHLIGHT_CLASSES

    norm_weights = {int(k): float(v) for k, v in weights.items()}

    all_labels, all_preds, all_scores_true, all_scores_pred = [], [], [], []
    for window_probs, window_scores, true_class, true_score in tiles:
        pred_class, _ = merge_class(window_probs, support_sets, weights=norm_weights, classes=HIGHLIGHT_CLASSES)
        pred_score = merge_score(window_scores, weights=norm_weights)

        all_labels.append(class_int_for(true_class))
        all_preds.append(class_int_for(pred_class))
        all_scores_true.append(true_score)
        all_scores_pred.append(pred_score)

    metrics = compute_full_metrics(all_labels, all_preds, all_scores_true, all_scores_pred, HIGHLIGHT_CLASSES)
    return {"n_tiles": len(all_labels), "metrics": metrics}


def evaluate_merge_weights(db, run_id: int, weights: dict) -> dict:
    """Re-merge a completed PredictionRun's *cached* per-window predictions
    (PredictionWindowResult — no model re-inference) under a candidate weight
    dict — applied to both the class vote and the score merge, same as
    production — and score the result against this match's
    GroundTruthSegment rows. Weights only affect the merge step
    (merge_score/merge_class), never the per-window model outputs, so this
    reproduces exactly what a full re-run with those weights would have
    produced, at a fraction of the cost.

    Returns {"n_tiles": int, "metrics": {...}} — metrics is the same schema
    training_common.compute_full_metrics uses, on the merged class/score.
    """
    tiles = _load_eval_tiles(db, run_id)
    support_sets = class_support_sets(db)
    return _score_weights(tiles, support_sets, weights)


def _weights_key(weights: dict) -> tuple:
    """Canonical form for duplicate detection — rounds to 4dp so e.g. 0.30
    and 0.3000001 (a JSON round-trip artifact) compare equal, and accepts
    both {8: v} and {"8": v} key styles."""
    def _get(w, k):
        return w.get(k, w.get(str(k), 0.0))
    return tuple(round(float(_get(weights, ws)), 4) for ws in WINDOW_SIZES)


def find_existing_eval(db, run_id: int, weights: dict):
    """The MergeWeightEvalRun for this run whose weights match (within 4dp),
    if any — used to avoid ever storing two rows for the same weight config."""
    target = _weights_key(weights)
    existing = (
        db.query(models.MergeWeightEvalRun)
        .filter(models.MergeWeightEvalRun.prediction_run_id == run_id)
        .all()
    )
    for row in existing:
        if _weights_key(json.loads(row.weights_json)) == target:
            return row
    return None


def generate_weight_grid(step: float = 0.05) -> list:
    """Every (w8, w16, w32) triple, each a multiple of `step`, summing to 1
    — the full non-redundant search space (merge_score/merge_class already
    renormalize, so any grid without the sum-to-1 constraint would waste
    most of its points on ratios that merge identically to ones already
    covered). Computed in integer steps to avoid float drift."""
    n = round(1.0 / step)
    combos = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            combos.append({
                WINDOW_SIZES[0]: round(i * step, 4),
                WINDOW_SIZES[1]: round(j * step, 4),
                WINDOW_SIZES[2]: round(k * step, 4),
            })
    return combos


def evaluate_all_weights(db, run_id: int, step: float = 0.05) -> dict:
    """Evaluate every weight combo on the step-0.05 simplex grid against this
    run, skipping (not recomputing, not re-storing) any combo that already
    has a MergeWeightEvalRun row for this run — whether from a prior grid
    run or a manual single Evaluate. Tiles are loaded once and reused across
    every grid point."""
    tiles = _load_eval_tiles(db, run_id)
    support_sets = class_support_sets(db)

    existing = (
        db.query(models.MergeWeightEvalRun)
        .filter(models.MergeWeightEvalRun.prediction_run_id == run_id)
        .all()
    )
    seen = {_weights_key(json.loads(row.weights_json)) for row in existing}

    grid = generate_weight_grid(step)
    created = 0
    for weights in grid:
        key = _weights_key(weights)
        if key in seen:
            continue
        result = _score_weights(tiles, support_sets, weights)
        db.add(models.MergeWeightEvalRun(
            prediction_run_id=run_id,
            label=f"grid step={step}",
            weights_json=json.dumps(weights),
            n_tiles=result["n_tiles"],
            metrics_json=json.dumps(result["metrics"]),
        ))
        seen.add(key)
        created += 1

    db.commit()
    return {"created": created, "skipped_duplicate": len(grid) - created, "total_combos": len(grid)}


# ---------------------------------------------------------------------------
# VisualScore weight calibration (admin, sub-tab) — the training-label
# formula VisualScore = base_weight*BaseScore(event_class) + flow_weight*
# OpticalFlowMagnitude, not the merge weights above. Calibrated against the
# same uploaded ground-truth CSV, but only rows with score != 0 (a zero
# means no commentary signal for that tile, not "definitely zero
# highlight-worthiness") and against each tile's *own* optical flow, lazily
# computed from its kept 8s clip and cached on GroundTruthSegment.flow_mag.
# ---------------------------------------------------------------------------

def _load_visual_score_tiles(db, run_id: int) -> list:
    """Ground-truth tiles for this run's match with score != 0, each paired
    with its BaseScore(event_class) and (lazily computed, cached) flow_mag.
    Returns a list of (base_score, flow_mag, true_score) tuples."""
    import feature_extractor
    from config import BASE_SCORES

    run = db.query(models.PredictionRun).filter(models.PredictionRun.id == run_id).first()
    if not run:
        raise ValueError(f"PredictionRun {run_id} not found")

    gt_rows = (
        db.query(models.GroundTruthSegment)
        .filter(models.GroundTruthSegment.match_id == run.match_id, models.GroundTruthSegment.score != 0)
        .all()
    )
    if not gt_rows:
        raise ValueError("No ground truth with a nonzero score for this match yet")

    seg_by_tile = {
        s.tile_index: s.clip_path
        for s in db.query(models.PredictionSegment).filter(models.PredictionSegment.run_id == run_id).all()
    }

    tiles = []
    dirty = False
    for g in gt_rows:
        clip_path = seg_by_tile.get(g.tile_index)
        if not clip_path:
            continue

        if g.flow_mag is None:
            if not Path(clip_path).exists():
                continue
            g.flow_mag = feature_extractor.mean_flow_magnitude(clip_path)
            dirty = True

        base_score = BASE_SCORES.get(g.event_class, 0.1)
        tiles.append((base_score, g.flow_mag, g.score))

    if dirty:
        db.commit()

    if not tiles:
        raise ValueError("No usable tiles — their 8s clip files may have been deleted since the Predict run.")
    return tiles


def _score_visual_weights(tiles: list, weights: dict) -> dict:
    base_w = float(weights.get("base", weights.get("base_weight", 0.0)))
    flow_w = float(weights.get("flow", weights.get("flow_weight", 0.0)))

    preds = np.array([
        np.clip(base_w * base_score + flow_w * flow_mag, 0.0, 1.0)
        for base_score, flow_mag, _ in tiles
    ])
    true = np.array([t for _, _, t in tiles])

    mae = float(np.mean(np.abs(true - preds)))
    ss_res = float(np.sum((true - preds) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    mse = ss_res / len(true)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0

    return {"n_tiles": len(tiles), "metrics": {"mae": round(mae, 6), "mse": round(mse, 6), "r2": round(r2, 6)}}


def evaluate_visual_score_weights(db, run_id: int, weights: dict) -> dict:
    tiles = _load_visual_score_tiles(db, run_id)
    return _score_visual_weights(tiles, weights)


def _visual_weights_key(weights: dict) -> tuple:
    base_w = weights.get("base", weights.get("base_weight", 0.0))
    flow_w = weights.get("flow", weights.get("flow_weight", 0.0))
    return (round(float(base_w), 4), round(float(flow_w), 4))


def find_existing_visual_eval(db, run_id: int, weights: dict):
    target = _visual_weights_key(weights)
    existing = (
        db.query(models.VisualScoreEvalRun)
        .filter(models.VisualScoreEvalRun.prediction_run_id == run_id)
        .all()
    )
    for row in existing:
        if _visual_weights_key(json.loads(row.weights_json)) == target:
            return row
    return None


def generate_visual_weight_grid(step: float = 0.05) -> list:
    """(base_weight, flow_weight) pairs summing to 1 — a single free
    variable, unlike the 3-window merge grid, since flow_weight = 1 - base_weight."""
    n = round(1.0 / step)
    return [
        {"base": round(i * step, 4), "flow": round(1.0 - i * step, 4)}
        for i in range(n + 1)
    ]


def evaluate_all_visual_score_weights(db, run_id: int, step: float = 0.05) -> dict:
    tiles = _load_visual_score_tiles(db, run_id)

    existing = (
        db.query(models.VisualScoreEvalRun)
        .filter(models.VisualScoreEvalRun.prediction_run_id == run_id)
        .all()
    )
    seen = {_visual_weights_key(json.loads(row.weights_json)) for row in existing}

    grid = generate_visual_weight_grid(step)
    created = 0
    for weights in grid:
        key = _visual_weights_key(weights)
        if key in seen:
            continue
        result = _score_visual_weights(tiles, weights)
        db.add(models.VisualScoreEvalRun(
            prediction_run_id=run_id,
            label=f"grid step={step}",
            weights_json=json.dumps(weights),
            n_tiles=result["n_tiles"],
            metrics_json=json.dumps(result["metrics"]),
        ))
        seen.add(key)
        created += 1

    db.commit()
    return {"created": created, "skipped_duplicate": len(grid) - created, "total_combos": len(grid)}


# ---------------------------------------------------------------------------
# Log / progress helpers (mirrors clip_extractor.py's conventions)
# ---------------------------------------------------------------------------

def _run_dir(run_id: int) -> Path:
    return PREDICTIONS_DIR / str(run_id)


def _log_path(run_id: int) -> Path:
    return _run_dir(run_id) / "inference.log"


def _progress_path(run_id: int) -> Path:
    return _run_dir(run_id) / "progress.json"


def _make_logger(run_id: int):
    path = _log_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w", encoding="utf-8", buffering=1)

    def log(msg: str):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line)
        fh.write(line + "\n")

    return log, fh


def _write_progress(run_id: int, tiles_total: int, tiles_done: int, status: str):
    _progress_path(run_id).write_text(
        json.dumps({"tiles_total": tiles_total, "tiles_done": tiles_done, "status": status}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Orchestration (BackgroundTask entrypoint, mirrors clip_extractor.run_extraction)
# ---------------------------------------------------------------------------

def run_inference(run_id: int):
    from database import SessionLocal

    log, fh = _make_logger(run_id)
    db = SessionLocal()
    try:
        _do_run_inference(run_id, db, log)
    except Exception as exc:
        log(f"FATAL: {exc}")
        _mark_error(run_id, db, str(exc))
    finally:
        fh.close()
        db.close()


def _do_run_inference(run_id: int, db, log):
    run = db.query(models.PredictionRun).filter(models.PredictionRun.id == run_id).first()
    if not run:
        log(f"ERROR: PredictionRun {run_id} not found.")
        return
    match = db.query(models.Match).filter(models.Match.id == run.match_id).first()
    if not match:
        log("ERROR: Match not found.")
        run.status, run.error_message = "error", "Match not found"
        db.commit()
        return

    log(f"Starting prediction run {run_id}: model={run.model_type} match={match.name}")
    device = torch.device("cuda" if run.device == "cuda" and torch.cuda.is_available() else "cpu")

    info = get_video_info(match.file_path)
    duration = info["duration"]
    log(f"Duration: {duration:.2f}s")

    support_sets = class_support_sets(db)
    run.class_support_json = json.dumps({str(ws): sorted(c) for ws, c in support_sets.items()})
    db.commit()
    log(f"Class support sets: {({ws: sorted(c) for ws, c in support_sets.items()})}")

    try:
        bundle = load_model_bundle(run.model_type, db, device)
    except Exception as exc:
        log(f"ERROR loading model: {exc}")
        run.status, run.error_message = "error", str(exc)[:500]
        db.commit()
        return

    if bundle.get("backbone_name"):
        run.backbone = bundle["backbone_name"]
        db.commit()

    tiles = build_tiles(duration)
    total = len(tiles)
    log(f"Planned {total} tiles of {TILE_SIZE}s")

    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_progress(run_id, total, 0, "running")

    done = 0
    for tile in tiles:
        try:
            clip_info = _extract_tile_clips(match.file_path, run_dir, tile, duration)

            window_probs, window_scores = {}, {}
            for ws, (path, _cstart, _cend) in clip_info.items():
                probs, score = predict_window(run.model_type, bundle, str(path), ws, device)
                window_probs[ws] = probs
                window_scores[ws] = score

            merged_score = merge_score(window_scores)
            merged_class, _combined = merge_class(window_probs, support_sets)

            seg = models.PredictionSegment(
                run_id=run_id, tile_index=tile["tile_index"],
                global_start_time=tile["t_start"], global_end_time=tile["t_end"],
                predicted_event=merged_class, highlight_score=round(merged_score, 4),
                clip_path=str(clip_info[8][0]),
            )
            db.add(seg)
            db.flush()

            for ws, probs in window_probs.items():
                top_idx = int(np.argmax(probs))
                db.add(models.PredictionWindowResult(
                    segment_id=seg.id, window_size=ws,
                    event_class=HIGHLIGHT_CLASSES[top_idx], confidence=float(probs[top_idx]),
                    highlight_score=round(float(window_scores[ws]), 4),
                    class_probs_json=json.dumps([float(p) for p in probs]),
                ))
            db.commit()

            for ws in (16, 32):
                try:
                    os.remove(clip_info[ws][0])
                except OSError:
                    pass

        except Exception as exc:
            log(f"WARN tile {tile['tile_index']}: {exc}")
            db.rollback()

        done += 1
        if done % 10 == 0 or done == total:
            log(f"Progress: {done}/{total} tiles ({int(done / total * 100) if total else 100}%)")
            _write_progress(run_id, total, done, "running")

    run.status = "completed"
    run.completed_at = datetime.utcnow()
    db.commit()
    _write_progress(run_id, total, done, "completed")
    log("Prediction run complete.")


def _mark_error(run_id: int, db, message: str):
    try:
        run = db.query(models.PredictionRun).filter(models.PredictionRun.id == run_id).first()
        if run:
            run.status = "error"
            run.error_message = message[:500]
            run.completed_at = datetime.utcnow()
            db.commit()
        _write_progress(run_id, 0, 0, "error")
    except Exception:
        pass
