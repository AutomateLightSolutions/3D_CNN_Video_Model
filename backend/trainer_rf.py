#!/usr/bin/env python3
"""Interpretable Features + Random Forest trainer. Launched as subprocess by the API."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=10)   # unused — kept for CLI compatibility with other trainers
    p.add_argument("--batch_size", type=int, default=0)  # ignored
    p.add_argument("--lr", type=float, default=0.0)      # ignored
    p.add_argument("--device", default="cpu")            # RF is CPU-only
    return p.parse_args()


def log(msg: str):
    print(msg, flush=True)


def main():
    args = parse_args()

    # ── resolve paths ──────────────────────────────────────────────
    BASE_DIR = Path(args.data_dir).parent          # Storage/
    features_dir = BASE_DIR / "features"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── imports ────────────────────────────────────────────────────
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        import joblib
    except ImportError as e:
        log(f"ERROR: {e}")
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).parent))
    from database import engine
    from sqlalchemy.orm import sessionmaker
    from config import HIGHLIGHT_CLASSES
    from training_common import load_labeled_split, load_clip_filter, class_int_for, visual_score_for, compute_full_metrics

    # ── shared match-level split (same held-out matches as every other trainer) ──
    Session = sessionmaker(bind=engine)
    db = Session()
    clip_filter = load_clip_filter(output_dir)
    if clip_filter is not None:
        log(f"Clip filter active: training restricted to {len(clip_filter)} uploaded clip ids.")
    train_raw, val_raw = load_labeled_split(db, allowed_clip_ids=clip_filter)
    db.close()

    if not train_raw:
        log("ERROR: No labeled clips found. Annotate data first.")
        sys.exit(1)

    def build_xy(raw):
        X, y_class, y_score, missing = [], [], [], []
        for clip, label in raw:
            feat_path = features_dir / f"{clip.id}.npy"
            if not feat_path.exists():
                missing.append(clip.id)
                continue
            feat = np.load(str(feat_path))
            flow_mag = float(feat[0])
            X.append(feat)
            y_class.append(class_int_for(label.event_class))
            y_score.append(visual_score_for(label.event_class, flow_mag))
        X = np.array(X, dtype=np.float32) if X else np.zeros((0, 25), dtype=np.float32)
        return X, np.array(y_class, dtype=np.int64), np.array(y_score, dtype=np.float32), missing

    X_train, y_train_c, y_train_s, missing_train = build_xy(train_raw)
    X_val, y_val_c, y_val_s, missing_val = build_xy(val_raw)

    n_missing = len(missing_train) + len(missing_val)
    if n_missing:
        log(f"WARNING: {n_missing} clips missing features — run feature extraction first.")

    if len(X_train) == 0 or len(X_val) == 0:
        log("ERROR: No feature files found for train or val split. Run /features/extract first.")
        sys.exit(1)

    log(f"Train: {len(X_train)} | Val: {len(X_val)}, {X_train.shape[1]} features, {len(HIGHLIGHT_CLASSES)} classes.")

    # ── feature names (must match feature_extractor.py FEATURE_NAMES) ─
    FEATURE_NAMES = [
        "mean_flow_magnitude", "flow_dir_0", "flow_dir_1", "flow_dir_2",
        "flow_dir_3", "flow_dir_4", "flow_dir_5", "flow_dir_6", "flow_dir_7",
        "flow_temporal_variance", "flow_entropy", "flow_acceleration",
        "player_count", "player_cluster_count", "player_spatial_spread",
        "player_central_zone_ratio",
        "body_lean_angle", "arms_above_shoulder_ratio",
        "legs_wide_stance_ratio", "arm_extension_angle",
        "camera_motion_magnitude", "zoom_indicator",
        "shot_boundary_count", "crowd_visibility_ratio", "jersey_color_ratio",
    ]
    if len(FEATURE_NAMES) != X_train.shape[1]:
        FEATURE_NAMES = [f"feat_{i}" for i in range(X_train.shape[1])]

    metrics_path = output_dir / "metrics.json"
    metrics_payload = {
        "status": "training", "model": "Interpretable + RF",
        "current_epoch": 0, "total_epochs": 1,
        "best_epoch": 0, "best_val_loss": None,
        "current": {}, "best": {},
        "per_class_f1": {}, "confusion_matrix": [], "completed_at": None,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # ── fit once on the train split, evaluate once on the held-out val split ──
    # (previously this used StratifiedKFold CV over ALL clips, which let clips
    # from the same match land in both train and val folds — leaking match-
    # specific cues and making RF's numbers incomparable to R3D/VideoMAE/
    # SlowFast, which have always used a match-level held-out split.)
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=None, max_features="sqrt",
        min_samples_leaf=2, n_jobs=-1, random_state=0,
    )
    reg = RandomForestRegressor(
        n_estimators=200, max_depth=None, max_features=1.0,
        min_samples_leaf=2, n_jobs=-1, random_state=0,
    )
    clf.fit(X_train, y_train_c)
    reg.fit(X_train, y_train_s)

    train_acc = float((clf.predict(X_train) == y_train_c).mean())
    val_pred_c = clf.predict(X_val)
    val_pred_s = reg.predict(X_val)

    m = compute_full_metrics(y_val_c, val_pred_c, y_val_s, val_pred_s, HIGHLIGHT_CLASSES)
    train_loss = 1.0 - train_acc
    val_loss = 1.0 - m["val_accuracy"]

    log(
        f"EPOCH 1 "
        f"TRAIN_LOSS {train_loss:.4f} "
        f"VAL_LOSS {val_loss:.4f} "
        f"VAL_ACC {m['val_accuracy']:.4f} "
        f"MACRO_F1 {m['macro_f1']:.4f} "
        f"MAE {m['mae']:.4f} "
        f"R2 {m['r2']:.4f}"
    )

    joblib.dump(
        {"classifier": clf, "regressor": reg, "classes": HIGHLIGHT_CLASSES},
        str(output_dir / "best_model.pkl"),
    )

    # ── feature importance (single fit, no fold-averaging needed anymore) ──
    importances = clf.feature_importances_
    total_imp = importances.sum() or 1.0
    feature_importance = {
        FEATURE_NAMES[i]: float(importances[i] / total_imp)
        for i in range(len(FEATURE_NAMES))
    }
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)
    )

    cur = {
        "train_loss": round(train_loss, 6), "val_loss": round(val_loss, 6),
        "val_accuracy": m["val_accuracy"], "macro_f1": m["macro_f1"],
        "weighted_f1": m["weighted_f1"], "mae": m["mae"], "mse": m["mse"], "r2": m["r2"],
    }
    metrics_payload.update({
        "status": "done",
        "current_epoch": 1,
        "best_epoch": 1,
        "best_val_loss": round(val_loss, 6),
        "current": cur,
        "best": {**cur, "precision": m["precision"], "recall": m["recall"], "pearson": m["pearson"]},
        "per_class_f1": m["per_class_f1"],
        "confusion_matrix": m["confusion_matrix"],
        "feature_importance": feature_importance,
        "classes": HIGHLIGHT_CLASSES,
        "n_samples": len(X_train) + len(X_val),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    log(f"Training complete. Val Acc: {m['val_accuracy']:.4f}, Macro F1: {m['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
