#!/usr/bin/env python3
"""Interpretable Features + MLP trainer. Launched as subprocess by the API."""

import argparse
import json
import signal
import sys
from datetime import datetime, timezone
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
        from torch.utils.data import Dataset, DataLoader
    except ImportError as e:
        log(f"ERROR: {e}")
        sys.exit(1)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    # ── load features via the shared match-level split ──────────────
    sys.path.insert(0, str(Path(__file__).parent))
    from database import engine
    from sqlalchemy.orm import sessionmaker
    from config import HIGHLIGHT_CLASSES
    from training_common import load_labeled_split, class_int_for, visual_score_for, compute_full_metrics

    Session = sessionmaker(bind=engine)
    db = Session()
    train_raw, val_raw = load_labeled_split(db)
    db.close()

    if not train_raw:
        log("ERROR: No labeled clips found.")
        sys.exit(1)

    def build_xy(raw):
        X, y_class, y_score = [], [], []
        for clip, label in raw:
            feat_path = features_dir / f"{clip.id}.npy"
            if not feat_path.exists():
                continue
            feat = np.load(str(feat_path)).astype(np.float32)
            flow_mag = float(feat[0])
            X.append(feat)
            y_class.append(class_int_for(label.event_class))
            y_score.append(visual_score_for(label.event_class, flow_mag))
        X = np.array(X, dtype=np.float32) if X else np.zeros((0, 25), dtype=np.float32)
        return X, np.array(y_class, dtype=np.int64), np.array(y_score, dtype=np.float32)

    X_train, y_train_c, y_train_s = build_xy(train_raw)
    X_val, y_val_c, y_val_s = build_xy(val_raw)

    if len(X_train) == 0 or len(X_val) == 0:
        log("ERROR: No feature files found. Run /features/extract first.")
        sys.exit(1)

    n_classes = len(HIGHLIGHT_CLASSES)
    in_dim = X_train.shape[1]
    log(f"Train: {len(X_train)} | Val: {len(X_val)}, {in_dim} features, {n_classes} classes.")

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

    train_ds = FeatDataset(X_train, y_train_c, y_train_s)
    val_ds   = FeatDataset(X_val,   y_val_c,   y_val_s)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

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
        "per_class_f1": {}, "confusion_matrix": [], "completed_at": None,
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
        m = compute_full_metrics(all_labels, all_preds, all_scores_true, all_scores_pred, HIGHLIGHT_CLASSES)

        log(
            f"EPOCH {epoch} "
            f"TRAIN_LOSS {train_loss:.4f} "
            f"VAL_LOSS {val_loss:.4f} "
            f"VAL_ACC {m['val_accuracy']:.4f} "
            f"MACRO_F1 {m['macro_f1']:.4f} "
            f"MAE {m['mae']:.4f} "
            f"R2 {m['r2']:.4f}"
        )

        torch.save(model.state_dict(), str(output_dir / "last_model.pt"))
        if m["macro_f1"] >= best_f1:
            best_f1, best_epoch = m["macro_f1"], epoch
            torch.save(model.state_dict(), str(output_dir / "best_model.pt"))

        cur = {
            "train_loss": round(train_loss, 6), "val_loss": round(val_loss, 6),
            "val_accuracy": m["val_accuracy"], "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"], "mae": m["mae"], "mse": m["mse"], "r2": m["r2"],
        }
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics  = {**cur, "precision": m["precision"], "recall": m["recall"], "pearson": m["pearson"]}

        metrics_payload.update({
            "current_epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 6),
            "current": cur,
            "best": best_metrics,
            "per_class_f1":     m["per_class_f1"]     if val_loss <= best_val_loss else metrics_payload.get("per_class_f1", {}),
            "confusion_matrix": m["confusion_matrix"]  if val_loss <= best_val_loss else metrics_payload.get("confusion_matrix", []),
        })
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # ── final metrics (reload best checkpoint) ──────────────────────
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

    m = compute_full_metrics(all_labels, all_preds, all_scores_true, all_scores_pred, HIGHLIGHT_CLASSES)

    metrics_payload.update({
        "status": "done",
        "best": {
            "val_accuracy": m["val_accuracy"], "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "precision": m["precision"], "recall": m["recall"],
            "mae": m["mae"], "mse": m["mse"], "r2": m["r2"],
            "pearson": m["pearson"],
        },
        "per_class_f1": m["per_class_f1"],
        "confusion_matrix": m["confusion_matrix"],
        "classes": HIGHLIGHT_CLASSES,
        "n_samples": len(X_train) + len(X_val),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    log(f"Training complete. Best epoch: {best_epoch}, Val Acc: {m['val_accuracy']:.4f}, Macro F1: {m['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
