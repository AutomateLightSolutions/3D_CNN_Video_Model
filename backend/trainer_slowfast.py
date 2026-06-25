#!/usr/bin/env python3
"""SlowFast R50 trainer. Launched as subprocess by the API."""

import argparse
import json
import signal
import sys
from datetime import datetime
from pathlib import Path
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# SlowFast constants
SLOW_FRAMES = 8
FAST_FRAMES = 32   # alpha=4: fast = 4 × slow
IMG_SIZE    = 224


def sample_pathway(total_frames, n_frames):
    import numpy as np
    if total_frames <= n_frames:
        idx = list(range(total_frames)) + [total_frames - 1] * (n_frames - total_frames)
        return idx
    return list(map(int, np.linspace(0, total_frames - 1, n_frames)))


def compute_class_weights(labels_list, highlight_classes):
    import torch
    counts  = Counter(labels_list)
    total   = sum(counts.values())
    weights = [total / (len(highlight_classes) * counts.get(cls, 1)) for cls in highlight_classes]
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(all_labels, all_preds, all_scores_true, all_scores_pred, highlight_classes):
    import numpy as np
    from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
    from scipy.stats import pearsonr

    val_accuracy = accuracy_score(all_labels, all_preds)
    macro_f1     = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    weighted_f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None,       zero_division=0,
                            labels=list(range(len(highlight_classes))))

    t = np.array(all_scores_true)
    p = np.array(all_scores_pred)
    mae     = float(np.mean(np.abs(t - p)))
    ss_res  = float(np.sum((t - p) ** 2))
    ss_tot  = float(np.sum((t - t.mean()) ** 2))
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0
    pearson = float(pearsonr(t, p)[0]) if len(t) > 1 else 0.0

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(highlight_classes))))
    return {
        "val_accuracy": round(float(val_accuracy), 6),
        "macro_f1":     round(float(macro_f1),     6),
        "weighted_f1":  round(float(weighted_f1),  6),
        "mae":          round(mae,     6),
        "r2":           round(r2,      6),
        "pearson":      round(pearson, 6),
        "per_class_f1": {highlight_classes[i]: round(float(v), 6) for i, v in enumerate(per_class_f1)},
        "confusion_matrix": cm.tolist(),
    }


def write_metrics(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_video_compat(clip_path):
    """Read video with cv2. Returns uint8 tensor (T, H, W, C) in RGB."""
    import cv2
    import numpy as np
    import torch
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {clip_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from: {clip_path}")
    return torch.from_numpy(np.stack(frames))


def main():
    args = parse_args()

    backend_dir = Path(__file__).parent
    sys.path.insert(0, str(backend_dir))

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    import torchvision.transforms.functional as TF
    import numpy as np
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    try:
        from pytorchvideo.models import create_slowfast
        _USE_PYTORCHVIDEO = True
    except ImportError:
        _USE_PYTORCHVIDEO = False
        print("WARNING: pytorchvideo not installed. Falling back to torch.hub for SlowFast.", flush=True)

    from config import DB_PATH, SLOWFAST_MODEL_DIR, HIGHLIGHT_CLASSES, WINDOW_CONFIG, SLOWFAST_METRICS_PATH, FEATURES_DIR, BASE_SCORES
    from models import Clip, Label, Match

    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = FEATURES_DIR
    metrics_path = SLOWFAST_METRICS_PATH
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    db_engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
    Session   = sessionmaker(bind=db_engine)
    db        = Session()

    labeled = (
        db.query(Clip, Label, Match)
        .join(Label, Clip.id == Label.clip_id)
        .join(Match, Clip.match_id == Match.id)
        .filter(Label.highlight_score.isnot(None))
        .all()
    )

    if not labeled:
        print("No labeled clips found. Exiting.")
        db.close()
        sys.exit(0)

    match_ids = list({m.id for _, _, m in labeled})
    rng = np.random.default_rng(42)
    rng.shuffle(match_ids)
    split     = max(1, int(len(match_ids) * 0.85))
    train_ids = set(match_ids[:split])
    val_ids   = set(match_ids[split:])

    def make_item(clip, label):
        class_int = (
            HIGHLIGHT_CLASSES.index(label.event_class)
            if label.event_class in HIGHLIGHT_CLASSES
            else len(HIGHLIGHT_CLASSES) - 1
        )
        feat_path = features_dir / f"{clip.id}.npy"
        flow_mag = float(np.load(str(feat_path))[0]) if feat_path.exists() else 0.0
        base = BASE_SCORES.get(label.event_class, 0.1)
        visual_score = float(np.clip(0.60 * base + 0.40 * flow_mag, 0.0, 1.0))
        return (clip.clip_path, class_int, visual_score)

    train_raw  = [(c, l) for c, l, m in labeled if m.id in train_ids]
    val_raw    = [(c, l) for c, l, m in labeled if m.id in val_ids]
    if not val_raw:
        val_raw = train_raw[: max(1, len(train_raw) // 10)]

    train_data = [make_item(c, l) for c, l in train_raw]
    val_data   = [make_item(c, l) for c, l in val_raw]
    db.close()
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    # Kinetics normalisation (same as R3D-18)
    MEAN = torch.tensor([0.45, 0.45, 0.45])
    STD  = torch.tensor([0.225, 0.225, 0.225])

    class SlowFastDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            clip_path, class_int, score = self.data[idx]
            score_t = torch.tensor(score, dtype=torch.float32)

            try:
                video = read_video_compat(clip_path)
            except Exception as exc:
                print(f"Warning: cannot read {clip_path}: {exc}", flush=True)
                slow = torch.zeros(3, SLOW_FRAMES, IMG_SIZE, IMG_SIZE)
                fast = torch.zeros(3, FAST_FRAMES, IMG_SIZE, IMG_SIZE)
                return slow, fast, class_int, score_t

            T = video.shape[0]
            if T == 0:
                slow = torch.zeros(3, SLOW_FRAMES, IMG_SIZE, IMG_SIZE)
                fast = torch.zeros(3, FAST_FRAMES, IMG_SIZE, IMG_SIZE)
                return slow, fast, class_int, score_t

            slow_idx = sample_pathway(T, SLOW_FRAMES)
            fast_idx = sample_pathway(T, FAST_FRAMES)

            def build_tensor(indices):
                frames = video[indices].permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
                frames = torch.stack([TF.resize(f, [IMG_SIZE, IMG_SIZE]) for f in frames])
                frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
                for c_idx in range(3):
                    frames[c_idx] = (frames[c_idx] - MEAN[c_idx]) / STD[c_idx]
                return frames

            return build_tensor(slow_idx), build_tensor(fast_idx), class_int, score_t

    train_ds = SlowFastDataset(train_data)
    val_ds   = SlowFastDataset(val_data)

    int_counts     = Counter(ci for _, ci, _ in train_data)
    sample_weights = [1.0 / max(int_counts[ci], 1) for _, ci, _ in train_data]
    sampler        = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,   num_workers=0, pin_memory=(str(device) == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    train_class_names = [l.event_class for c, l in train_raw]
    class_weights     = compute_class_weights(train_class_names, HIGHLIGHT_CLASSES).to(device)
    num_classes       = len(HIGHLIGHT_CLASSES)

    class SlowFastHighlightModel(nn.Module):
        """SlowFast R50 backbone with dual classification + regression head."""

        def __init__(self, num_classes):
            super().__init__()
            loaded = False

            if _USE_PYTORCHVIDEO:
                try:
                    from pytorchvideo.models.hub import slowfast_r50
                    base    = slowfast_r50(pretrained=True)
                    # Remove final projection head; keep blocks up to the pooling stage
                    self.backbone = nn.Sequential(*list(base.blocks[:-1]))
                    feat_dim = 2304  # SlowFast R50 concatenated feat dim
                    loaded = True
                    print("Loaded pretrained SlowFast R50 (pytorchvideo).", flush=True)
                except Exception as e:
                    print(f"pytorchvideo load failed: {e}", flush=True)

            if not loaded:
                # torch.hub fallback
                try:
                    base = torch.hub.load(
                        "facebookresearch/pytorchvideo",
                        "slowfast_r50",
                        pretrained=True,
                        verbose=False,
                    )
                    self.backbone = nn.Sequential(*list(base.blocks[:-1]))
                    feat_dim = 2304
                    loaded = True
                    print("Loaded pretrained SlowFast R50 (torch.hub).", flush=True)
                except Exception as e:
                    print(f"WARNING: could not load pretrained SlowFast weights ({e}). Training from random init.", flush=True)
                    base = torch.hub.load(
                        "facebookresearch/pytorchvideo",
                        "slowfast_r50",
                        pretrained=False,
                        verbose=False,
                    )
                    self.backbone = nn.Sequential(*list(base.blocks[:-1]))
                    feat_dim = 2304

            self.pool       = nn.AdaptiveAvgPool3d(1)
            self.class_head = nn.Linear(feat_dim, num_classes)
            self.score_head = nn.Sequential(nn.Linear(feat_dim, 1), nn.Sigmoid())

        def forward(self, slow, fast):
            # SlowFast backbone expects a list [slow, fast]
            feat = self.backbone([slow, fast])
            # feat is a list from the last block; take the concatenated output
            if isinstance(feat, (list, tuple)):
                feat = torch.cat([self.pool(f).flatten(1) for f in feat], dim=1)
            else:
                feat = self.pool(feat).flatten(1)
            return self.class_head(feat), self.score_head(feat).squeeze(1)

    model    = SlowFastHighlightModel(num_classes).to(device)
    ce_loss  = nn.CrossEntropyLoss(weight=class_weights)
    mse_loss = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch    = 0
    best_metrics  = {}
    current_epoch = [0]

    def save_checkpoint(epoch, val_loss, is_best=False):
        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss}
        torch.save(ckpt, output_dir / "last_model.pt")
        if is_best:
            torch.save(ckpt, output_dir / "best_model.pt")

    metrics_payload = {
        "status": "training", "model": "SlowFast-R50",
        "current_epoch": 0, "total_epochs": args.epochs,
        "best_epoch": 0, "best_val_loss": None,
        "current": {}, "best": {},
        "per_class_f1": {}, "confusion_matrix": [],
        "completed_at": None,
    }

    def handle_signal(sig, frame):
        print("\nInterrupted — saving checkpoint...", flush=True)
        save_checkpoint(current_epoch[0], best_val_loss)
        write_metrics(metrics_path, {**metrics_payload, "status": "stopped"})
        sys.exit(0)

    signal.signal(signal.SIGINT,  handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    write_metrics(metrics_path, metrics_payload)

    # 3-phase training scaled to 30 epochs
    PHASES = [
        {"start": 1,  "end": 8,           "freeze": "all",     "lr": 1e-3},
        {"start": 9,  "end": 22,          "freeze": "partial", "lr": 1e-4},
        {"start": 23, "end": args.epochs, "freeze": "none",    "lr": 1e-5},
    ]

    def get_phase(epoch):
        for ph in PHASES:
            if ph["start"] <= epoch <= ph["end"]:
                return ph
        return PHASES[-1]

    def apply_phase(epoch):
        ph = get_phase(epoch)
        for param in model.backbone.parameters():
            param.requires_grad = False

        if ph["freeze"] == "none":
            for param in model.backbone.parameters():
                param.requires_grad = True
        elif ph["freeze"] == "partial":
            # Unfreeze last two backbone blocks (res4 + res5 equivalent)
            backbone_blocks = list(model.backbone.children())
            for block in backbone_blocks[-2:]:
                for param in block.parameters():
                    param.requires_grad = True

        for param in model.pool.parameters():
            param.requires_grad = True
        for param in model.class_head.parameters():
            param.requires_grad = True
        for param in model.score_head.parameters():
            param.requires_grad = True

        trainable = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable, lr=ph["lr"])

    optimizer  = apply_phase(1)
    prev_phase = get_phase(1)

    for epoch in range(1, args.epochs + 1):
        current_epoch[0] = epoch

        cur_phase = get_phase(epoch)
        if cur_phase is not prev_phase:
            optimizer  = apply_phase(epoch)
            prev_phase = cur_phase

        model.train()
        train_loss_sum = 0.0
        for slow, fast, cls_idx, scores in train_loader:
            slow    = slow.to(device)
            fast    = fast.to(device)
            cls_idx = cls_idx.to(device)
            scores  = scores.to(device)
            optimizer.zero_grad()
            logits, pred_scores = model(slow, fast)
            loss = 1.0 * ce_loss(logits, cls_idx) + 2.0 * mse_loss(pred_scores, scores)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
        train_loss = train_loss_sum / len(train_loader)

        model.eval()
        val_loss_sum    = 0.0
        all_labels      = []
        all_preds       = []
        all_scores_true = []
        all_scores_pred = []

        with torch.no_grad():
            for slow, fast, cls_idx, scores in val_loader:
                slow    = slow.to(device)
                fast    = fast.to(device)
                cls_idx = cls_idx.to(device)
                scores  = scores.to(device)
                logits, pred_scores = model(slow, fast)
                loss = 1.0 * ce_loss(logits, cls_idx) + 2.0 * mse_loss(pred_scores, scores)
                val_loss_sum += loss.item()
                all_labels.extend(cls_idx.cpu().tolist())
                all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                all_scores_true.extend(scores.cpu().tolist())
                all_scores_pred.extend(pred_scores.cpu().tolist())

        val_loss = val_loss_sum / len(val_loader)
        m = compute_metrics(all_labels, all_preds, all_scores_true, all_scores_pred, HIGHLIGHT_CLASSES)

        print(
            f"EPOCH {epoch} TRAIN_LOSS {train_loss:.4f} VAL_LOSS {val_loss:.4f} "
            f"VAL_ACC {m['val_accuracy']:.4f} MACRO_F1 {m['macro_f1']:.4f} "
            f"MAE {m['mae']:.4f} R2 {m['r2']:.4f}",
            flush=True,
        )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch    = epoch
            best_metrics  = {k: v for k, v in m.items() if k not in ("per_class_f1", "confusion_matrix")}
            save_checkpoint(epoch, val_loss, is_best=True)

        save_checkpoint(epoch, val_loss, is_best=False)

        metrics_payload.update({
            "current_epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 6),
            "current": {
                "train_loss":   round(train_loss, 6),
                "val_loss":     round(val_loss,   6),
                "val_accuracy": m["val_accuracy"],
                "macro_f1":     m["macro_f1"],
                "weighted_f1":  m["weighted_f1"],
                "mae":          m["mae"],
                "r2":           m["r2"],
            },
            "best": best_metrics,
            "per_class_f1":     m["per_class_f1"]     if is_best else metrics_payload.get("per_class_f1", {}),
            "confusion_matrix": m["confusion_matrix"]  if is_best else metrics_payload.get("confusion_matrix", []),
        })
        write_metrics(metrics_path, metrics_payload)

    metrics_payload.update({"status": "done", "completed_at": datetime.utcnow().isoformat()})
    write_metrics(metrics_path, metrics_payload)
    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
