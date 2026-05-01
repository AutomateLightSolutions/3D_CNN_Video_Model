#!/usr/bin/env python3
"""Standalone training script. Launched as subprocess by the API."""

import argparse
import signal
import sys
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


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

    from config import DB_PATH, TRAINING_LOG_PATH, MODEL_DIR, EVENT_CLASSES
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

    train_data = [(c, l) for c, l, m in labeled if m.id in train_ids]
    val_data = [(c, l) for c, l, m in labeled if m.id in val_ids]
    if not val_data:
        val_data = train_data[: max(1, len(train_data) // 10)]

    db.close()
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
    STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)

    class VideoDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            clip, label = self.data[idx]
            class_idx = EVENT_CLASSES.index(label.event_class) if label.event_class in EVENT_CLASSES else len(EVENT_CLASSES) - 1
            score = torch.tensor(label.highlight_score, dtype=torch.float32)

            try:
                video, _, _ = tvio.read_video(clip.clip_path, output_format="THWC", pts_unit="sec")
            except Exception as exc:
                print(f"Warning: cannot read {clip.clip_path}: {exc}")
                return torch.zeros(3, 16, 112, 112), class_idx, score

            T = video.shape[0]
            if T == 0:
                return torch.zeros(3, 16, 112, 112), class_idx, score

            if T >= 16:
                indices = torch.linspace(0, T - 1, 16).long()
            else:
                pad = torch.full((16 - T,), T - 1, dtype=torch.long)
                indices = torch.cat([torch.arange(T), pad])

            frames = video[indices].permute(0, 3, 1, 2).float() / 255.0  # (16, C, H, W)
            frames = torch.stack([TF.resize(f, [112, 112]) for f in frames])  # (16, C, 112, 112)
            frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
            frames = (frames - MEAN) / STD
            return frames, class_idx, score

    train_ds = VideoDataset(train_data)
    val_ds = VideoDataset(val_data)

    # Inverse-frequency weighted sampler
    class_counts: dict = {}
    for _, lbl in train_data:
        class_counts[lbl.event_class] = class_counts.get(lbl.event_class, 0) + 1
    sample_weights = [1.0 / max(class_counts.get(lbl.event_class, 1), 1) for _, lbl in train_data]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0,
                              pin_memory=(str(device) == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model with dual head
    class R3DHighlightModel(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.class_head = nn.Linear(512, num_classes)
            self.score_head = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

        def forward(self, x):
            feat = self.backbone(x).flatten(1)
            return self.class_head(feat), self.score_head(feat).squeeze(1)

    model = R3DHighlightModel(len(EVENT_CLASSES)).to(device)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    best_val_loss = float("inf")
    current_epoch = [0]

    def save_checkpoint(epoch: int, val_loss: float, is_best: bool = False):
        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss}
        torch.save(ckpt, output_dir / "last_model.pt")
        if is_best:
            torch.save(ckpt, output_dir / "best_model.pt")

    def handle_signal(sig, frame):
        print("\nInterrupted — saving checkpoint...")
        save_checkpoint(current_epoch[0], best_val_loss)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    log_path = Path(TRAINING_LOG_PATH)
    log_file = open(log_path, "w", buffering=1)

    def log(msg: str):
        print(msg)
        log_file.write(msg + "\n")

    # Phase configuration
    PHASES = [
        {"start": 1,  "end": 10,         "freeze": "all",     "lr": 1e-3},
        {"start": 11, "end": 30,         "freeze": "partial", "lr": 1e-4},
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
            frames = frames.to(device)
            cls_idx = cls_idx.to(device)
            scores = scores.to(device)
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
                frames = frames.to(device)
                cls_idx = cls_idx.to(device)
                scores = scores.to(device)
                logits, pred_scores = model(frames)
                loss = 1.0 * ce_loss(logits, cls_idx) + 2.0 * mse_loss(pred_scores, scores)
                val_loss_sum += loss.item()
                correct += (logits.argmax(dim=1) == cls_idx).sum().item()
                total += cls_idx.size(0)

        val_loss = val_loss_sum / len(val_loader)
        val_acc = correct / total if total > 0 else 0.0

        log(f"EPOCH {epoch} TRAIN_LOSS {train_loss:.4f} VAL_LOSS {val_loss:.4f} VAL_ACC {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(epoch, val_loss, is_best=True)

    log_file.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
