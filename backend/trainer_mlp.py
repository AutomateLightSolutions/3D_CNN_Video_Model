#!/usr/bin/env python3
"""Interpretable Features + MLP trainer. Launched as subprocess by the API."""

import argparse
import json
import signal
import sys
import os
from collections import Counter
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


_stop = False


def _handle_signal(sig, frame):
    global _stop
    _stop = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def log(msg: str):
    print(msg, flush=True)


def main():
    global _stop
    args = parse_args()

    BASE_DIR = Path(args.data_dir).parent
    features_dir = BASE_DIR / "features"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader, random_split
        from sklearn.metrics import f1_score, mean_absolute_error, r2_score
        from sklearn.preprocessing import LabelEncoder
    except ImportError as e:
        log(f"ERROR: {e}")
        sys.exit(1)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    # ── load features ──────────────────────────────────────────────
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
        log("ERROR: No labeled clips found.")
        sys.exit(1)

    X, y_class, y_score = [], [], []
    for clip_id, event_class in rows:
        feat_path = features_dir / f"{clip_id}.npy"
        if not feat_path.exists():
            continue
        feat = np.load(str(feat_path)).astype(np.float32)
        flow_mag = float(feat[0])
        base = BASE_SCORES.get(event_class, 0.1)
        visual_score = float(np.clip(0.60 * base + 0.40 * flow_mag, 0.0, 1.0))
        X.append(feat)
        y_class.append(event_class)
        y_score.append(visual_score)

    if not X:
        log("ERROR: No feature files found. Run /features/extract first.")
        sys.exit(1)

    X = np.array(X, dtype=np.float32)
    y_score = np.array(y_score, dtype=np.float32)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_class).astype(np.int64)
    classes = list(le.classes_)
    n_classes = len(classes)
    in_dim = X.shape[1]

    log(f"Loaded {len(X)} clips, {in_dim} features, {n_classes} classes.")

    # ── dataset ────────────────────────────────────────────────────
    class FeatDataset(Dataset):
        def __init__(self, feats, labels, scores):
            self.feats = torch.from_numpy(feats)
            self.labels = torch.from_numpy(labels)
            self.scores = torch.from_numpy(scores)

        def __len__(self):
            return len(self.feats)

        def __getitem__(self, i):
            return self.feats[i], self.labels[i], self.scores[i]

    full_ds = FeatDataset(X, y_enc, y_score)
    val_n = max(1, int(0.2 * len(full_ds)))
    train_n = len(full_ds) - val_n
    train_ds, val_ds = random_split(full_ds, [train_n, val_n],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── model ──────────────────────────────────────────────────────
    class InterpMLP(nn.Module):
        def __init__(self, in_dim, n_classes):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 64), nn.ReLU(),
            )
            self.class_head = nn.Linear(64, n_classes)
            self.score_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

        def forward(self, x):
            feat = self.backbone(x)
            return self.class_head(feat), self.score_head(feat).squeeze(1)

    model = InterpMLP(in_dim, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()

    best_f1, best_epoch = -1.0, 0
    best_val_loss = float("inf")
    best_metrics  = {}

    metrics_path = output_dir / "metrics.json"
    metrics_payload = {
        "status": "training", "model": "Interpretable + MLP",
        "current_epoch": 0, "total_epochs": args.epochs,
        "best_epoch": 0, "best_val_loss": None,
        "current": {}, "best": {},
        "per_class_f1": {}, "completed_at": None,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        if _stop:
            log(f"Stopped at epoch {epoch}.")
            break

        # train
        model.train()
        t_loss_sum, t_steps = 0.0, 0
        for feats, labels, scores in train_loader:
            feats, labels, scores = feats.to(device), labels.to(device), scores.to(device)
            logits, score_pred = model(feats)
            loss = cls_criterion(logits, labels) + 0.3 * reg_criterion(score_pred, scores)
            opt.zero_grad(); loss.backward(); opt.step()
            t_loss_sum += loss.item(); t_steps += 1

        train_loss = t_loss_sum / max(t_steps, 1)

        # validate
        model.eval()
        v_loss_sum, v_steps = 0.0, 0
        all_preds, all_labels = [], []
        all_scores_pred, all_scores_true = [], []
        with torch.no_grad():
            for feats, labels, scores in val_loader:
                feats, labels, scores = feats.to(device), labels.to(device), scores.to(device)
                logits, score_pred = model(feats)
                loss = cls_criterion(logits, labels) + 0.3 * reg_criterion(score_pred, scores)
                v_loss_sum += loss.item(); v_steps += 1
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores_pred.extend(score_pred.cpu().numpy())
                all_scores_true.extend(scores.cpu().numpy())

        val_loss = v_loss_sum / max(v_steps, 1)
        val_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(len(all_labels), 1)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        wf1_cur  = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
        mae = float(mean_absolute_error(all_scores_true, all_scores_pred))
        r2 = float(r2_score(all_scores_true, all_scores_pred))

        log(
            f"EPOCH {epoch} "
            f"TRAIN_LOSS {train_loss:.4f} "
            f"VAL_LOSS {val_loss:.4f} "
            f"VAL_ACC {val_acc:.4f} "
            f"MACRO_F1 {macro_f1:.4f} "
            f"MAE {mae:.4f} "
            f"R2 {r2:.4f}"
        )

        torch.save(model.state_dict(), str(output_dir / "last_model.pt"))
        if macro_f1 >= best_f1:
            best_f1, best_epoch = macro_f1, epoch
            torch.save(model.state_dict(), str(output_dir / "best_model.pt"))

        cur = {
            "train_loss": round(train_loss, 6), "val_loss": round(val_loss, 6),
            "val_accuracy": round(float(val_acc), 6), "macro_f1": round(float(macro_f1), 6),
            "weighted_f1": round(wf1_cur, 6), "mae": round(mae, 6), "r2": round(r2, 6),
        }
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics  = cur.copy()

        metrics_payload.update({
            "current_epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 6),
            "current": cur,
            "best": best_metrics,
        })
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # ── final metrics ──────────────────────────────────────────────
    model.load_state_dict(torch.load(str(output_dir / "best_model.pt"), map_location=device, weights_only=True))
    model.eval()
    all_preds, all_labels = [], []
    all_scores_pred, all_scores_true = [], []
    with torch.no_grad():
        for feats, labels, scores in val_loader:
            feats, labels, scores = feats.to(device), labels.to(device), scores.to(device)
            logits, score_pred = model(feats)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores_pred.extend(score_pred.cpu().numpy())
            all_scores_true.extend(scores.cpu().numpy())

    import numpy as np
    val_acc  = sum(p == l for p, l in zip(all_preds, all_labels)) / max(len(all_labels), 1)
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    wf1      = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
    mae      = float(mean_absolute_error(all_scores_true, all_scores_pred))
    r2       = float(r2_score(all_scores_true, all_scores_pred))

    per_class_f1  = f1_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_dict = {classes[i]: float(per_class_f1[i]) for i in range(len(classes)) if i < len(per_class_f1)}

    from datetime import timezone
    metrics_payload.update({
        "status": "done",
        "best": {
            "val_accuracy": round(float(val_acc), 6), "macro_f1": round(macro_f1, 6),
            "weighted_f1": round(wf1, 6), "mae": round(mae, 6), "r2": round(r2, 6),
        },
        "per_class_f1": per_class_dict,
        "classes": classes,
        "n_samples": len(X),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    log(f"Training complete. Best epoch: {best_epoch}, Val Acc: {val_acc:.4f}, Macro F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
