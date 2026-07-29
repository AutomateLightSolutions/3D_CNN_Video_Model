#!/usr/bin/env python3
"""
Shared data-split and metric utilities used by every trainer (R3D, VideoMAE,
SlowFast, RF, MLP, Hybrid).

This module exists so all six models are evaluated on an identical,
leakage-free held-out set of matches, against an identical VisualScore
target and an identical class-index mapping. Without this, results across
model families are not directly comparable — see conversation history for
why (clip-level splits leak match-specific cues; per-model LabelEncoders can
assign different integers to the same class across runs).
"""

import numpy as np

SPLIT_SEED = 42
TRAIN_FRACTION = 0.85


def load_labeled_split(db):
    """Match-level train/val split: no match's clips appear in both sets.

    Deterministic and identical across every trainer given the same labeled
    dataset — match IDs are sorted before shuffling so the result does not
    depend on set/dict iteration order.

    Returns (train_raw, val_raw), each a list of (Clip, Label) tuples.
    """
    from models import Clip, Label, Match

    labeled = (
        db.query(Clip, Label, Match)
        .join(Label, Clip.id == Label.clip_id)
        .join(Match, Clip.match_id == Match.id)
        .filter(Label.highlight_score.isnot(None))
        .all()
    )
    if not labeled:
        return [], []

    match_ids = sorted({m.id for _, _, m in labeled})
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(match_ids)
    split = max(1, int(len(match_ids) * TRAIN_FRACTION))
    train_ids = set(match_ids[:split])
    val_ids = set(match_ids[split:])

    train_raw = [(c, l) for c, l, m in labeled if m.id in train_ids]
    val_raw = [(c, l) for c, l, m in labeled if m.id in val_ids]
    if not val_raw:
        val_raw = train_raw[: max(1, len(train_raw) // 10)]

    return train_raw, val_raw


def class_int_for(event_class: str) -> int:
    """Canonical class->int mapping, shared by every trainer so per-class F1
    and confusion matrices line up across all six models regardless of which
    classes happen to appear in a given train/val split."""
    from config import HIGHLIGHT_CLASSES

    return (
        HIGHLIGHT_CLASSES.index(event_class)
        if event_class in HIGHLIGHT_CLASSES
        else len(HIGHLIGHT_CLASSES) - 1
    )


def visual_score_for(event_class: str, flow_mag: float) -> float:
    """VisualScore = 0.60 * BaseScore(class) + 0.40 * OpticalFlowMagnitude_norm.

    Single definition point — previously this formula was hardcoded
    identically in five separate trainer files.
    """
    from config import BASE_SCORES

    base = BASE_SCORES.get(event_class, 0.1)
    return float(np.clip(0.60 * base + 0.40 * flow_mag, 0.0, 1.0))


def compute_full_metrics(all_labels, all_preds, all_scores_true, all_scores_pred,
                          highlight_classes=None):
    """Identical metrics schema for every model: val_accuracy, macro/weighted
    F1, precision/recall, MAE/R2/Pearson on the score head, per-class F1,
    confusion matrix."""
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, confusion_matrix, accuracy_score,
    )
    from scipy.stats import pearsonr

    if highlight_classes is None:
        from config import HIGHLIGHT_CLASSES
        highlight_classes = HIGHLIGHT_CLASSES

    val_accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    macro_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(
        all_labels, all_preds, average=None, zero_division=0,
        labels=list(range(len(highlight_classes))),
    )

    t = np.array(all_scores_true)
    p = np.array(all_scores_pred)
    mae = float(np.mean(np.abs(t - p)))
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    mse = ss_res / len(t)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0
    pearson = float(pearsonr(t, p)[0]) if len(t) > 1 else 0.0

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(highlight_classes))))
    return {
        "val_accuracy": round(float(val_accuracy), 6),
        "macro_f1": round(float(macro_f1), 6),
        "weighted_f1": round(float(weighted_f1), 6),
        "precision": round(float(macro_precision), 6),
        "recall": round(float(macro_recall), 6),
        "mae": round(mae, 6),
        "mse": round(mse, 6),
        "r2": round(r2, 6),
        "pearson": round(pearson, 6),
        "per_class_f1": {highlight_classes[i]: round(float(v), 6) for i, v in enumerate(per_class_f1)},
        "confusion_matrix": cm.tolist(),
    }
