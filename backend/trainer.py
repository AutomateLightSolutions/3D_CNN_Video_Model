#!/usr/bin/env python3
"""Standalone training script. Launched as subprocess by the API."""

import argparse
import signal
import sys
import os
from pathlib import Path
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def sample_frame_indices(total_frames: int, window_size_s: int, window_config: dict):
    """
    Sample frame indices from a clip.
    Short windows (8s): uniform sampling — action can be anywhere.
    Long windows (16s, 32s): dense-end sampling — action resolves in second half.
    """
    import numpy as np
    n_frames = window_config[window_size_s]["frames"]

    if total_frames <= n_frames:
        return np.arange(total_frames)

    if window_size_s <= 8:
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    else:
        # Dense-end: 1/3 frames from first half, 2/3 from second half
        n_first = n_frames // 3
        n_second = n_frames - n_first
        first_half = np.linspace(0, total_frames // 2, n_first, dtype=int)
        second_half = np.linspace(total_frames // 2, total_frames - 1, n_second, dtype=int)
        indices = np.concatenate([first_half, second_half])

    return indices


def compute_class_weights(labels_list: list, highlight_classes: list):
    """
    Compute inverse frequency weights per class.
    Rare classes (try, line_break) get high weight; common ones (normal_play) get low weight.
    """
    import torch
    counts = Counter(labels_list)
    total = sum(counts.values())
    weights = []
    for cls in highlight_classes:
        count = counts.get(cls, 1)
        weights.append(total / (len(highlight_classes) * count))
    return torch.tensor(weights, dtype=torch.float32)


def main():
    args = parse_args()

    backend_dir = Path(__file__).parent
    sys.path.insert(0, str(backend_dir))

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    import torchvision.io as tvio
    import torchvision.transforms.functional as TF
    from torchvision.models.video import r3d_18, R3D_18_Weights
    import numpy as np
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from config import DB_PATH, MODEL_DIR, HIGHLIGHT_CLASSES, WINDOW_CONFIG
    from models import Clip, Label, Match

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_engine = create_engine(
        f"sqlite:///{DB_PATH}",
        connect_args={"check_same_thread": False},
    )
    Session = sessionmaker(bind=db_engine)
    db = Session()

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

    # Match-level train/val split
    match_ids = list({m.id for _, _, m in labeled})
    rng = np.random.default_rng(42)
    rng.shuffle(match_ids)
    split = max(1, int(len(match_ids) * 0.85))
    train_ids = set(match_ids[:split])
    val_ids = set(match_ids[split:])

    def make_item(clip, label):
        class_int = (
            HIGHLIGHT_CLASSES.index(label.event_class)
            if label.event_class in HIGHLIGHT_CLASSES
            else len(HIGHLIGHT_CLASSES) - 1
        )
        return (clip.clip_path, clip.window_size, class_int, label.highlight_score)

    train_raw = [(c, l) for c, l, m in labeled if m.id in train_ids]
    val_raw   = [(c, l) for c, l, m in labeled if m.id in val_ids]
    if not val_raw:
        val_raw = train_raw[: max(1, len(train_raw) // 10)]

    train_data = [make_item(c, l) for c, l in train_raw]
    val_data   = [make_item(c, l) for c, l in val_raw]

    db.close()
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
    STD  = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)

    class RugbyClipDataset(Dataset):
        def __init__(self, data):
            self.data = data  # list of (clip_path, window_size_s, class_int, score)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            clip_path, window_size_s, class_int, score = self.data[idx]
            score_t = torch.tensor(score, dtype=torch.float32)

            # Clamp window_size_s to a known key; default to 8 if unknown
            ws = window_size_s if window_size_s in WINDOW_CONFIG else 8
            n_frames = WINDOW_CONFIG[ws]["frames"]

            try:
                video, _, _ = tvio.read_video(clip_path, output_format="THWC", pts_unit="sec")
            except Exception as exc:
                print(f"Warning: cannot read {clip_path}: {exc}")
                return torch.zeros(3, n_frames, 112, 112), class_int, score_t

            T = video.shape[0]
            if T == 0:
                return torch.zeros(3, n_frames, 112, 112), class_int, score_t

            indices = sample_frame_indices(T, ws, WINDOW_CONFIG)

            # Pad if fewer frames than needed
            if len(indices) < n_frames:
                pad = np.full(n_frames - len(indices), len(indices) - 1, dtype=int)
                indices = np.concatenate([indices, pad])

            frames = video[indices].permute(0, 3, 1, 2).float() / 255.0  # (n, C, H, W)
            frames = torch.stack([TF.resize(f, [112, 112]) for f in frames])  # (n, C, 112, 112)
            frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
            frames = (frames - MEAN) / STD
            return frames, class_int, score_t

    train_ds = RugbyClipDataset(train_data)
    val_ds   = RugbyClipDataset(val_data)

    # Class-weighted sampler
    int_counts = Counter(ci for _, _, ci, _ in train_data)
    sample_weights = [1.0 / max(int_counts[ci], 1) for _, _, ci, _ in train_data]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=0, pin_memory=(str(device) == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Class weights for CE loss (inverse frequency, computed from training labels)
    train_class_names = [l.event_class for c, l in train_raw]
    class_weights = compute_class_weights(train_class_names, HIGHLIGHT_CLASSES).to(device)

    # Model with dual head
    num_classes = len(HIGHLIGHT_CLASSES)  # 23

    class R3DHighlightModel(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            try:
                backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
                print("Loaded pretrained R3D-18 weights from cache.", flush=True)
            except OSError as e:
                # URLError (no internet / DNS failure) is a subclass of OSError
                print(f"WARNING: could not load pretrained weights ({e})."
                      " Training from random initialisation.", flush=True)
                backbone = r3d_18(weights=None)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.class_head = nn.Linear(512, num_classes)
            self.score_head = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

        def forward(self, x):
            feat = self.backbone(x).flatten(1)
            return self.class_head(feat), self.score_head(feat).squeeze(1)

    model = R3DHighlightModel(num_classes).to(device)
    ce_loss  = nn.CrossEntropyLoss(weight=class_weights)
    mse_loss = nn.MSELoss()

    best_val_loss = float("inf")
    current_epoch = [0]

    def save_checkpoint(epoch: int, val_loss: float, is_best: bool = False):
        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss}
        torch.save(ckpt, output_dir / "last_model.pt")
        if is_best:
            torch.save(ckpt, output_dir / "best_model.pt")

    def handle_signal(sig, frame):
        print("\nInterrupted — saving checkpoint...", flush=True)
        save_checkpoint(current_epoch[0], best_val_loss)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

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
            # Unfreeze layer3 (idx 3) and layer4 (idx 4)
            for child in backbone_children[3:5]:
                for param in child.parameters():
                    param.requires_grad = True

        for param in model.class_head.parameters():
            param.requires_grad = True
        for param in model.score_head.parameters():
            param.requires_grad = True

        trainable = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable, lr=ph["lr"])

    optimizer = apply_phase(1)
    prev_phase = get_phase(1)

    for epoch in range(1, args.epochs + 1):
        current_epoch[0] = epoch

        cur_phase = get_phase(epoch)
        if cur_phase is not prev_phase:
            optimizer = apply_phase(epoch)
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
        val_loss_sum = 0.0
        correct = total = 0
        with torch.no_grad():
            for frames, cls_idx, scores in val_loader:
                frames  = frames.to(device)
                cls_idx = cls_idx.to(device)
                scores  = scores.to(device)
                logits, pred_scores = model(frames)
                loss = 1.0 * ce_loss(logits, cls_idx) + 2.0 * mse_loss(pred_scores, scores)
                val_loss_sum += loss.item()
                correct += (logits.argmax(dim=1) == cls_idx).sum().item()
                total   += cls_idx.size(0)

        val_loss = val_loss_sum / len(val_loader)
        val_acc  = correct / total if total > 0 else 0.0

        log(f"EPOCH {epoch} TRAIN_LOSS {train_loss:.4f} VAL_LOSS {val_loss:.4f} VAL_ACC {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(epoch, val_loss, is_best=True)

    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
