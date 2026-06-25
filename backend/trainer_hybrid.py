#!/usr/bin/env python3
"""Interpretable + Deep Features Hybrid trainer. Launched as subprocess by the API."""

import argparse
import json
import signal
import sys
import os
from collections import Counter
from datetime import datetime
from pathlib import Path


# Backbone feature dimensions for the penultimate layer of each model
BACKBONE_DIMS = {
    "r3d": 512,
    "videomae": 768,
    "slowfast": 2304,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--backbone", default="r3d", choices=["r3d", "videomae", "slowfast"])
    return p.parse_args()


_stop = False


def _handle_signal(sig, frame):
    global _stop
    _stop = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def log(msg: str):
    print(msg, flush=True)


def read_video_compat(clip_path):
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


def load_backbone(backbone_name: str, model_dir: Path, device):
    """Load a frozen backbone and return (model, transform_fn, feat_dim)."""
    import torch
    import torch.nn as nn

    if backbone_name == "r3d":
        import torchvision.models.video as vm
        model = vm.r3d_18(weights=None)
        feat_dim = 512
        ckpt = model_dir / "models" / "r3d" / "best_model.pt"
        if ckpt.exists():
            state = torch.load(str(ckpt), map_location=device)
            # Strip classifier head keys if present
            state = {k: v for k, v in state.items() if "fc" not in k}
            model.load_state_dict(state, strict=False)
            log(f"Loaded R3D weights from {ckpt}")
        else:
            log("WARNING: R3D best_model.pt not found — using random init.")
        # Remove final FC
        model.fc = nn.Identity()
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        def transform_r3d(video_tensor):
            # video_tensor: (T, H, W, C) uint8 → (1, C, T, H, W) float
            import torchvision.transforms.functional as F
            import torch
            frames = video_tensor.permute(0, 3, 1, 2).float() / 255.0
            frames = torch.stack([F.resize(f, [112, 112]) for f in frames])
            mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1)
            std  = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1)
            frames = (frames.permute(1, 0, 2, 3) - mean) / std
            return frames.unsqueeze(0)

        return model, transform_r3d, feat_dim

    elif backbone_name == "videomae":
        try:
            from transformers import VideoMAEModel, VideoMAEFeatureExtractor
        except ImportError:
            log("ERROR: transformers not installed.")
            sys.exit(1)
        import torch
        ckpt_dir = model_dir / "models" / "videomae"
        if (ckpt_dir / "config.json").exists():
            backbone = VideoMAEModel.from_pretrained(str(ckpt_dir))
        else:
            backbone = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        feat_dim = backbone.config.hidden_size
        backbone = backbone.to(device).eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")

        def transform_vmae(video_tensor):
            import numpy as np
            frames = video_tensor.numpy()
            if frames.shape[0] < 16:
                idx = list(range(frames.shape[0]))
                idx += [idx[-1]] * (16 - len(idx))
            else:
                step = frames.shape[0] / 16
                idx = [int(i * step) for i in range(16)]
            sampled = [frames[i] for i in idx]
            inputs = extractor(sampled, return_tensors="pt")
            return inputs["pixel_values"]

        def vmae_forward(pixel_values):
            out = backbone(pixel_values=pixel_values.to(device))
            return out.last_hidden_state[:, 0, :]

        class WrapVMAE:
            def __call__(self, pv):
                return vmae_forward(pv)

        return WrapVMAE(), transform_vmae, feat_dim

    elif backbone_name == "slowfast":
        import torch
        import torch.nn as nn
        try:
            model = torch.hub.load(
                "facebookresearch/pytorchvideo", "slowfast_r50",
                pretrained=False, verbose=False,
            )
        except Exception as e:
            log(f"ERROR loading SlowFast: {e}")
            sys.exit(1)

        ckpt = model_dir / "models" / "slowfast" / "best_model.pt"
        if ckpt.exists():
            state = torch.load(str(ckpt), map_location=device)
            state = {k: v for k, v in state.items() if "head" not in k}
            model.load_state_dict(state, strict=False)
            log(f"Loaded SlowFast weights from {ckpt}")
        else:
            log("WARNING: SlowFast best_model.pt not found — using random init.")

        # Remove head
        model.blocks[-1].proj = nn.Identity()
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        feat_dim = 2304

        def transform_sf(video_tensor):
            import torch
            import torchvision.transforms.functional as F
            frames = video_tensor.permute(0, 3, 1, 2).float() / 255.0
            frames = torch.stack([F.resize(f, [256, 256]) for f in frames])
            mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1)
            std  = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1)
            frames = (frames.permute(1, 0, 2, 3) - mean) / std
            T = frames.shape[1]
            slow_idx = [int(i * T / 8) for i in range(8)]
            fast_idx = list(range(T))[:32]
            while len(fast_idx) < 32:
                fast_idx.append(fast_idx[-1])
            slow = frames[:, slow_idx, :, :].unsqueeze(0)
            fast = frames[:, fast_idx, :, :].unsqueeze(0)
            return [slow, fast]

        def sf_forward(inp):
            out = model(inp)
            return out if out.ndim == 2 else out.mean(dim=[2, 3, 4])

        class WrapSF:
            def __call__(self, inp):
                return sf_forward(inp)

        return WrapSF(), transform_sf, feat_dim

    raise ValueError(f"Unknown backbone: {backbone_name}")


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
        from sklearn.metrics import f1_score, mean_absolute_error, r2_score
        from sklearn.preprocessing import LabelEncoder
    except ImportError as e:
        log(f"ERROR: {e}")
        sys.exit(1)

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    log(f"Using device: {device}, backbone: {args.backbone}")

    # ── load features and clip paths ───────────────────────────────
    sys.path.insert(0, str(Path(__file__).parent))
    from database import engine
    from sqlalchemy import text
    from config import BASE_SCORES

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT c.id, c.clip_path, l.event_class "
            "FROM clips c JOIN labels l ON l.clip_id = c.id"
        )).fetchall()

    if not rows:
        log("ERROR: No labeled clips found.")
        sys.exit(1)

    clip_ids, clip_paths, y_class, y_score, interp_feats = [], [], [], [], []
    for clip_id, clip_path, event_class in rows:
        feat_path = features_dir / f"{clip_id}.npy"
        if not feat_path.exists():
            continue
        feat = np.load(str(feat_path)).astype(np.float32)
        flow_mag = float(feat[0])
        base = BASE_SCORES.get(event_class, 0.1)
        visual_score = float(np.clip(0.60 * base + 0.40 * flow_mag, 0.0, 1.0))
        clip_ids.append(clip_id)
        clip_paths.append(clip_path)
        interp_feats.append(feat)
        y_class.append(event_class)
        y_score.append(visual_score)

    if not interp_feats:
        log("ERROR: No feature files found. Run /features/extract first.")
        sys.exit(1)

    interp_feats = np.array(interp_feats, dtype=np.float32)
    y_score_arr = np.array(y_score, dtype=np.float32)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_class).astype(np.int64)
    classes = list(le.classes_)
    n_classes = len(classes)
    in_dim = interp_feats.shape[1]

    log(f"Loaded {len(interp_feats)} clips, {in_dim} interp features, {n_classes} classes.")

    # ── load backbone ──────────────────────────────────────────────
    backbone_model, transform_fn, feat_dim = load_backbone(args.backbone, BASE_DIR, device)

    # ── pre-extract deep features (one-time, all clips) ───────────
    log("Pre-extracting deep features from backbone...")
    deep_feats = []
    for i, (cid, cpath) in enumerate(zip(clip_ids, clip_paths)):
        try:
            video = read_video_compat(cpath)
            inp = transform_fn(video)
            with torch.no_grad():
                if isinstance(inp, list):
                    inp = [x.to(device) for x in inp]
                else:
                    inp = inp.to(device)
                feat = backbone_model(inp)
            deep_feats.append(feat.squeeze(0).cpu().numpy().astype(np.float32))
        except Exception as e:
            log(f"WARNING: clip {cid} deep extraction failed: {e} — using zeros")
            deep_feats.append(np.zeros(feat_dim, dtype=np.float32))

        if (i + 1) % 20 == 0:
            log(f"  {i+1}/{len(clip_ids)} clips done")

    deep_feats = np.array(deep_feats, dtype=np.float32)
    log(f"Deep features: {deep_feats.shape}")

    # ── dataset ────────────────────────────────────────────────────
    class HybridDataset(Dataset):
        def __init__(self, interp, deep, labels, scores):
            self.interp = torch.from_numpy(interp)
            self.deep   = torch.from_numpy(deep)
            self.labels = torch.from_numpy(labels)
            self.scores = torch.from_numpy(scores)

        def __len__(self):
            return len(self.interp)

        def __getitem__(self, i):
            return self.interp[i], self.deep[i], self.labels[i], self.scores[i]

    from torch.utils.data import random_split
    full_ds = HybridDataset(interp_feats, deep_feats, y_enc, y_score_arr)
    val_n = max(1, int(0.2 * len(full_ds)))
    train_n = len(full_ds) - val_n
    train_ds, val_ds = random_split(full_ds, [train_n, val_n],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── fusion model ───────────────────────────────────────────────
    class HybridFusion(nn.Module):
        def __init__(self, interp_dim, deep_dim, n_classes):
            super().__init__()
            self.interp_proj = nn.Sequential(
                nn.Linear(interp_dim, 128), nn.ReLU(),
            )
            self.deep_proj = nn.Sequential(
                nn.Linear(deep_dim, 128), nn.ReLU(),
            )
            self.fusion = nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            )
            self.class_head = nn.Linear(128, n_classes)
            self.score_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

        def forward(self, interp, deep):
            i_feat = self.interp_proj(interp)
            d_feat = self.deep_proj(deep)
            fused  = self.fusion(torch.cat([i_feat, d_feat], dim=1))
            return self.class_head(fused), self.score_head(fused).squeeze(1)

    model = HybridFusion(in_dim, feat_dim, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()

    best_f1, best_epoch = -1.0, 0
    best_val_loss = float("inf")
    best_metrics  = {}

    metrics_path = output_dir / "metrics.json"
    metrics_payload = {
        "status": "training", "model": f"Interpretable + Deep ({args.backbone})",
        "backbone_used": args.backbone,
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

        model.train()
        t_loss_sum, t_steps = 0.0, 0
        for interp, deep, labels, scores in train_loader:
            interp, deep, labels, scores = (
                interp.to(device), deep.to(device),
                labels.to(device), scores.to(device),
            )
            logits, score_pred = model(interp, deep)
            loss = cls_criterion(logits, labels) + 0.3 * reg_criterion(score_pred, scores)
            opt.zero_grad(); loss.backward(); opt.step()
            t_loss_sum += loss.item(); t_steps += 1

        train_loss = t_loss_sum / max(t_steps, 1)

        model.eval()
        v_loss_sum, v_steps = 0.0, 0
        all_preds, all_labels = [], []
        all_scores_pred, all_scores_true = [], []
        with torch.no_grad():
            for interp, deep, labels, scores in val_loader:
                interp, deep, labels, scores = (
                    interp.to(device), deep.to(device),
                    labels.to(device), scores.to(device),
                )
                logits, score_pred = model(interp, deep)
                loss = cls_criterion(logits, labels) + 0.3 * reg_criterion(score_pred, scores)
                v_loss_sum += loss.item(); v_steps += 1
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores_pred.extend(score_pred.cpu().numpy())
                all_scores_true.extend(scores.cpu().numpy())

        val_loss = v_loss_sum / max(v_steps, 1)
        val_acc  = sum(p == l for p, l in zip(all_preds, all_labels)) / max(len(all_labels), 1)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        wf1_cur  = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
        mae      = float(mean_absolute_error(all_scores_true, all_scores_pred))
        r2       = float(r2_score(all_scores_true, all_scores_pred))

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
    model.load_state_dict(torch.load(str(output_dir / "best_model.pt"), map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    all_scores_pred, all_scores_true = [], []
    with torch.no_grad():
        for interp, deep, labels, scores in val_loader:
            interp, deep, labels, scores = (
                interp.to(device), deep.to(device),
                labels.to(device), scores.to(device),
            )
            logits, score_pred = model(interp, deep)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores_pred.extend(score_pred.cpu().numpy())
            all_scores_true.extend(scores.cpu().numpy())

    val_acc  = sum(p == l for p, l in zip(all_preds, all_labels)) / max(len(all_labels), 1)
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    wf1      = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
    mae      = float(mean_absolute_error(all_scores_true, all_scores_pred))
    r2       = float(r2_score(all_scores_true, all_scores_pred))

    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_dict = {
        classes[i]: float(per_class_f1[i])
        for i in range(len(classes)) if i < len(per_class_f1)
    }

    from datetime import timezone
    metrics_payload.update({
        "status": "done",
        "best": {
            "val_accuracy": round(float(val_acc), 6), "macro_f1": round(macro_f1, 6),
            "weighted_f1": round(wf1, 6), "mae": round(mae, 6), "r2": round(r2, 6),
        },
        "per_class_f1": per_class_dict,
        "classes": classes,
        "n_samples": len(interp_feats),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    log(
        f"Training complete. Backbone: {args.backbone}, "
        f"Best epoch: {best_epoch}, Val Acc: {val_acc:.4f}, Macro F1: {macro_f1:.4f}"
    )


if __name__ == "__main__":
    main()
