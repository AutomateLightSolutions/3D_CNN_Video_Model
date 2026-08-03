#!/usr/bin/env python3
"""R3D-18 trainer. Launched as subprocess by the API."""

import argparse
import json
import signal
import sys
import os
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def sample_frame_indices(total_frames: int, window_size_s: int, n_frames: int):
    """
    Sample n_frames indices from a clip.
    Short windows (8s): uniform sampling — action can be anywhere.
    Long windows (16s, 32s): dense-end sampling — action resolves in second half.
    """
    import numpy as np

    if total_frames <= n_frames:
        return np.arange(total_frames)

    if window_size_s <= 8:
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    else:
        # Dense-end: 1/3 frames from first half, 2/3 from second half
        n_first = n_frames // 3
        n_second = n_frames - n_first
        first_half  = np.linspace(0, total_frames // 2, n_first,  dtype=int)
        second_half = np.linspace(total_frames // 2, total_frames - 1, n_second, dtype=int)
        indices = np.concatenate([first_half, second_half])

    return indices


def compute_class_weights(labels_list: list, highlight_classes: list):
    import torch
    counts = Counter(labels_list)
    total  = sum(counts.values())
    weights = []
    for cls in highlight_classes:
        count = counts.get(cls, 1)
        weights.append(total / (len(highlight_classes) * count))
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(all_labels, all_preds, all_scores_true, all_scores_pred, highlight_classes):
    from training_common import compute_full_metrics
    return compute_full_metrics(all_labels, all_preds, all_scores_true, all_scores_pred, highlight_classes)


def write_metrics(metrics_path, payload):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
STD  = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)


class RugbyClipDataset(Dataset):
    """Module-level (not nested in main()) so Windows' spawn-based DataLoader
    workers can pickle/import it — a class defined inside a function can't be
    resolved by name in a worker process, which silently breaks num_workers>0."""

    def __init__(self, data, window_config, n_frames):
        self.data = data  # list of (clip_path, window_size_s, class_int, score)
        self.window_config = window_config
        self.n_frames = n_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip_path, window_size_s, class_int, score = self.data[idx]
        score_t = torch.tensor(score, dtype=torch.float32)

        ws = window_size_s if window_size_s in self.window_config else 8

        try:
            video = read_video_compat(clip_path)
        except Exception as exc:
            print(f"Warning: cannot read {clip_path}: {exc}", flush=True)
            return torch.zeros(3, self.n_frames, 112, 112), class_int, score_t

        T = video.shape[0]
        if T == 0:
            return torch.zeros(3, self.n_frames, 112, 112), class_int, score_t

        indices = sample_frame_indices(T, ws, self.n_frames)

        # Pad if fewer frames than needed
        if len(indices) < self.n_frames:
            pad = np.full(self.n_frames - len(indices), len(indices) - 1, dtype=int)
            indices = np.concatenate([indices, pad])
        indices = indices[:self.n_frames]

        frames = video[indices].permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
        frames = torch.stack([TF.resize(f, [112, 112]) for f in frames])  # (T, C, 112, 112)
        frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
        frames = (frames - MEAN) / STD
        return frames, class_int, score_t


def main():
    args = parse_args()

    backend_dir = Path(__file__).parent
    sys.path.insert(0, str(backend_dir))

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    import torchvision.transforms.functional as TF
    from torchvision.models.video import r3d_18, R3D_18_Weights
    import numpy as np
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from config import DB_PATH, R3D_MODEL_DIR, HIGHLIGHT_CLASSES, WINDOW_CONFIG, R3D_METRICS_PATH, FEATURES_DIR
    from training_common import load_labeled_split, class_int_for, visual_score_for, compute_full_metrics
    from model_defs import R3DHighlightModel, R3D_N_FRAMES

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (requested: {args.device}, cuda_available: {torch.cuda.is_available()})", flush=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = FEATURES_DIR
    metrics_path = R3D_METRICS_PATH
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    db_engine = create_engine(
        f"sqlite:///{DB_PATH}",
        connect_args={"check_same_thread": False},
    )
    Session = sessionmaker(bind=db_engine)
    db = Session()

    train_raw, val_raw = load_labeled_split(db)

    if not train_raw:
        print("No labeled clips found. Exiting.")
        db.close()
        sys.exit(0)

    def make_item(clip, label):
        class_int = class_int_for(label.event_class)
        feat_path = features_dir / f"{clip.id}.npy"
        flow_mag = float(np.load(str(feat_path))[0]) if feat_path.exists() else 0.0
        visual_score = visual_score_for(label.event_class, flow_mag)
        return (clip.clip_path, clip.window_size, class_int, visual_score)

    train_data = [make_item(c, l) for c, l in train_raw]
    val_data   = [make_item(c, l) for c, l in val_raw]

    db.close()
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    train_ds = RugbyClipDataset(train_data, WINDOW_CONFIG, R3D_N_FRAMES)
    val_ds   = RugbyClipDataset(val_data, WINDOW_CONFIG, R3D_N_FRAMES)

    # Class-weighted sampler
    int_counts = Counter(ci for _, _, ci, _ in train_data)
    sample_weights = [1.0 / max(int_counts[ci], 1) for _, _, ci, _ in train_data]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # num_workers>0 overlaps CPU video decoding (cv2, single-clip-at-a-time)
    # with GPU compute instead of blocking on it every batch — this is what
    # was starving the GPU even after CUDA was working correctly.
    NUM_WORKERS = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=NUM_WORKERS, persistent_workers=(NUM_WORKERS > 0),
                              pin_memory=(str(device) == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, persistent_workers=(NUM_WORKERS > 0))

    # Class weights for CE loss
    train_class_names = [l.event_class for c, l in train_raw]
    class_weights = compute_class_weights(train_class_names, HIGHLIGHT_CLASSES).to(device)

    # Model with dual head
    num_classes = len(HIGHLIGHT_CLASSES)  # 23

    model    = R3DHighlightModel(num_classes).to(device)
    ce_loss  = nn.CrossEntropyLoss(weight=class_weights)
    mse_loss = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch    = 0
    best_metrics  = {}
    current_epoch = [0]

    def save_checkpoint(epoch: int, val_loss: float, is_best: bool = False):
        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss}
        torch.save(ckpt, output_dir / "last_model.pt")
        if is_best:
            torch.save(ckpt, output_dir / "best_model.pt")

    metrics_payload = {
        "status": "training", "model": "R3D-18",
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

    def log(msg: str):
        print(msg, flush=True)

    # Phase configuration
    PHASES = [
        {"start": 1,  "end": 10,          "freeze": "all",     "lr": 1e-3},
        {"start": 11, "end": 30,          "freeze": "partial", "lr": 1e-4},
        {"start": 31, "end": args.epochs, "freeze": "none",    "lr": 1e-5},
    ]

    def get_phase(epoch: int) -> dict:
        for ph in PHASES:
            if ph["start"] <= epoch <= ph["end"]:
                return ph
        return PHASES[-1]

    def apply_phase(epoch: int):
        ph = get_phase(epoch)
        backbone_children = list(model.backbone.children())

        for param in model.backbone.parameters():
            param.requires_grad = False

        if ph["freeze"] == "none":
            for param in model.backbone.parameters():
                param.requires_grad = True
        elif ph["freeze"] == "partial":
            for child in backbone_children[3:5]:
                for param in child.parameters():
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

        # Train
        model.train()
        train_loss_sum = 0.0
        for frames, cls_idx, scores in train_loader:
            frames  = frames.to(device)
            cls_idx = cls_idx.to(device)
            scores  = scores.to(device)
            optimizer.zero_grad()
            logits, pred_scores = model(frames)
            loss = 1.0 * ce_loss(logits, cls_idx) + 2.0 * mse_loss(pred_scores, scores)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        train_loss = train_loss_sum / len(train_loader)

        # Validate
        model.eval()
        val_loss_sum    = 0.0
        all_labels      = []
        all_preds       = []
        all_scores_true = []
        all_scores_pred = []

        with torch.no_grad():
            for frames, cls_idx, scores in val_loader:
                frames  = frames.to(device)
                cls_idx = cls_idx.to(device)
                scores  = scores.to(device)
                logits, pred_scores = model(frames)
                loss = 1.0 * ce_loss(logits, cls_idx) + 2.0 * mse_loss(pred_scores, scores)
                val_loss_sum += loss.item()
                all_labels.extend(cls_idx.cpu().tolist())
                all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                all_scores_true.extend(scores.cpu().tolist())
                all_scores_pred.extend(pred_scores.cpu().tolist())

        val_loss = val_loss_sum / len(val_loader)
        m = compute_metrics(all_labels, all_preds, all_scores_true, all_scores_pred, HIGHLIGHT_CLASSES)

        log(f"EPOCH {epoch} TRAIN_LOSS {train_loss:.4f} VAL_LOSS {val_loss:.4f} "
            f"VAL_ACC {m['val_accuracy']:.4f} MACRO_F1 {m['macro_f1']:.4f} "
            f"MAE {m['mae']:.4f} R2 {m['r2']:.4f}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch    = epoch
            best_metrics  = {k: v for k, v in m.items() if k not in ("per_class_f1", "confusion_matrix")}
            save_checkpoint(epoch, val_loss, is_best=True)

        save_checkpoint(epoch, val_loss, is_best=False)

        metrics_payload.update({
            "current_epoch": epoch,
            "best_epoch":    best_epoch,
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
