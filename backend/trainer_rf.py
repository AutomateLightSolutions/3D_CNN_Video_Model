#!/usr/bin/env python3
"""Interpretable Features + Random Forest trainer. Launched as subprocess by the API."""

import argparse
import json
import sys
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=10)   # treated as n_folds
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
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
        from sklearn.preprocessing import LabelEncoder
        import joblib
    except ImportError as e:
        log(f"ERROR: {e}")
        sys.exit(1)

    # ── load feature files from DB ─────────────────────────────────
    sys.path.insert(0, str(Path(__file__).parent))
    from database import engine
    from sqlalchemy import text
    from config import BASE_SCORES

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT c.id, l.event_class "
            "FROM clips c JOIN labels l ON l.clip_id = c.id"
        )).fetchall()

    if not rows:
        log("ERROR: No labeled clips found. Annotate data first.")
        sys.exit(1)

    X, y_class, y_score, missing = [], [], [], []
    for clip_id, event_class in rows:
        feat_path = features_dir / f"{clip_id}.npy"
        if not feat_path.exists():
            missing.append(clip_id)
            continue
        feat = np.load(str(feat_path))
        flow_mag = float(feat[0])
        base = BASE_SCORES.get(event_class, 0.1)
        visual_score = float(np.clip(0.60 * base + 0.40 * flow_mag, 0.0, 1.0))
        X.append(feat)
        y_class.append(event_class)
        y_score.append(visual_score)

    if missing:
        log(f"WARNING: {len(missing)} clips missing features — run feature extraction first.")

    if not X:
        log("ERROR: No feature files found. Run /features/extract first.")
        sys.exit(1)

    X = np.array(X, dtype=np.float32)
    y_score = np.array(y_score, dtype=np.float32)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_class)
    classes = list(le.classes_)
    n_classes = len(classes)

    log(f"Loaded {len(X)} clips, {X.shape[1]} features, {n_classes} classes.")

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
    if len(FEATURE_NAMES) != X.shape[1]:
        FEATURE_NAMES = [f"feat_{i}" for i in range(X.shape[1])]

    # ── cross-validation ───────────────────────────────────────────
    n_folds = max(2, min(args.epochs, len(X)))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_metrics = []
    all_importances = np.zeros(X.shape[1])

    metrics_path = output_dir / "metrics.json"
    metrics_payload = {
        "status": "training", "model": "Interpretable + RF",
        "current_epoch": 0, "total_epochs": n_folds,
        "best_epoch": 0, "best_val_loss": None,
        "current": {}, "best": {},
        "per_class_f1": {}, "completed_at": None,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    best_val_loss = float("inf")
    best_metrics  = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_enc), start=1):
        x_tr, x_val = X[train_idx], X[val_idx]
        y_tr_c, y_val_c = y_enc[train_idx], y_enc[val_idx]
        y_tr_s, y_val_s = y_score[train_idx], y_score[val_idx]

        clf = RandomForestClassifier(
            n_estimators=200, max_depth=None, max_features="sqrt",
            min_samples_leaf=2, n_jobs=-1, random_state=fold,
        )
        reg = RandomForestRegressor(
            n_estimators=200, max_depth=None, max_features=1.0,
            min_samples_leaf=2, n_jobs=-1, random_state=fold,
        )

        clf.fit(x_tr, y_tr_c)
        reg.fit(x_tr, y_tr_s)

        all_importances += clf.feature_importances_

        train_acc = accuracy_score(y_tr_c, clf.predict(x_tr))
        val_pred = clf.predict(x_val)
        val_acc = accuracy_score(y_val_c, val_pred)
        macro_f1 = f1_score(y_val_c, val_pred, average="macro", zero_division=0)
        wf1_fold = float(f1_score(y_val_c, val_pred, average="weighted", zero_division=0))

        score_pred = reg.predict(x_val)
        mae = float(mean_absolute_error(y_val_s, score_pred))
        r2 = float(r2_score(y_val_s, score_pred))

        train_loss = 1.0 - train_acc
        val_loss = 1.0 - val_acc

        log(
            f"EPOCH {fold} "
            f"TRAIN_LOSS {train_loss:.4f} "
            f"VAL_LOSS {val_loss:.4f} "
            f"VAL_ACC {val_acc:.4f} "
            f"MACRO_F1 {macro_f1:.4f} "
            f"MAE {mae:.4f} "
            f"R2 {r2:.4f}"
        )

        fold_metrics.append({
            "fold": fold,
            "val_acc": float(val_acc),
            "macro_f1": float(macro_f1),
            "mae": mae,
            "r2": r2,
        })

        cur = {
            "train_loss": round(train_loss, 6), "val_loss": round(val_loss, 6),
            "val_accuracy": round(float(val_acc), 6), "macro_f1": round(float(macro_f1), 6),
            "weighted_f1": round(wf1_fold, 6), "mae": round(mae, 6), "r2": round(r2, 6),
        }
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics  = cur.copy()

        metrics_payload.update({
            "current_epoch": fold,
            "best_epoch": fold_metrics[[m["val_acc"] for m in fold_metrics].index(max(m["val_acc"] for m in fold_metrics))]["fold"],
            "best_val_loss": round(best_val_loss, 6),
            "current": cur,
            "best": best_metrics,
        })
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # ── train final model on all data ──────────────────────────────
    clf_final = RandomForestClassifier(
        n_estimators=200, max_depth=None, max_features="sqrt",
        min_samples_leaf=2, n_jobs=-1, random_state=0,
    )
    reg_final = RandomForestRegressor(
        n_estimators=200, max_depth=None, max_features=1.0,
        min_samples_leaf=2, n_jobs=-1, random_state=0,
    )
    clf_final.fit(X, y_enc)
    reg_final.fit(X, y_score)

    joblib.dump(
        {"classifier": clf_final, "regressor": reg_final, "label_encoder": le},
        str(output_dir / "best_model.pkl"),
    )

    # ── aggregate metrics ──────────────────────────────────────────
    avg_acc = float(sum(m["val_acc"] for m in fold_metrics) / len(fold_metrics))
    avg_f1 = float(sum(m["macro_f1"] for m in fold_metrics) / len(fold_metrics))
    avg_mae = float(sum(m["mae"] for m in fold_metrics) / len(fold_metrics))
    avg_r2 = float(sum(m["r2"] for m in fold_metrics) / len(fold_metrics))

    # Per-class F1 on full dataset (final model)
    y_pred_all = clf_final.predict(X)
    wf1 = float(f1_score(y_enc, y_pred_all, average="weighted", zero_division=0))
    per_class_f1 = f1_score(y_enc, y_pred_all, average=None, zero_division=0)
    per_class_dict = {cls: float(per_class_f1[i]) for i, cls in enumerate(classes)}

    # Normalise feature importance across folds
    all_importances /= n_folds
    total_imp = all_importances.sum() or 1.0
    feature_importance = {
        FEATURE_NAMES[i]: float(all_importances[i] / total_imp)
        for i in range(len(FEATURE_NAMES))
    }
    # Sort descending for readability
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)
    )

    best_fold = fold_metrics[[m["val_acc"] for m in fold_metrics].index(max(m["val_acc"] for m in fold_metrics))]
    metrics_payload.update({
        "status": "done",
        "current_epoch": n_folds,
        "best_epoch": best_fold["fold"],
        "best_val_loss": round(1.0 - best_fold["val_acc"], 6),
        "best": {
            "val_accuracy": round(avg_acc, 6), "macro_f1": round(avg_f1, 6),
            "weighted_f1": round(wf1, 6), "mae": round(avg_mae, 6), "r2": round(avg_r2, 6),
        },
        "per_class_f1": per_class_dict,
        "feature_importance": feature_importance,
        "fold_results": fold_metrics,
        "classes": classes,
        "n_samples": len(X),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    log(f"Training complete. Avg Val Acc: {avg_acc:.4f}, Avg Macro F1: {avg_f1:.4f}")


if __name__ == "__main__":
    main()
