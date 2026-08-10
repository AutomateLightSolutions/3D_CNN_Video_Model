#!/usr/bin/env python3
"""Interpretable + Deep Features Hybrid trainer. Launched as subprocess by the API."""

import argparse
import json
import signal
import sys
from datetime import datetime, timezone
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

    # Registered here (not at module level) because this module is also
    # imported by inference.py's load_hybrid() to reuse load_backbone() —
    # that import happens inside a FastAPI BackgroundTask worker thread, and
    # signal.signal() raises ValueError outside the main thread. Only the
    # standalone `python trainer_hybrid.py` subprocess path needs this.
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

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

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    log(f"Using device: {device}, backbone: {args.backbone}")

    # ── load features via the shared match-level split ──────────────
    sys.path.insert(0, str(Path(__file__).parent))
    from database import engine
    from sqlalchemy.orm import sessionmaker
    from config import HIGHLIGHT_CLASSES
    from training_common import load_labeled_split, load_clip_filter, class_int_for, visual_score_for, compute_full_metrics
    from model_defs import HybridFusion

    Session = sessionmaker(bind=engine)
    db = Session()
    clip_filter = load_clip_filter(output_dir)
    if clip_filter is not None:
        log(f"Clip filter active: training restricted to {len(clip_filter)} uploaded clip ids.")
    train_raw, val_raw = load_labeled_split(db, allowed_clip_ids=clip_filter)
    db.close()

    if not train_raw:
        log("ERROR: No labeled clips found.")
        sys.exit(1)

    def build_interp(raw):
        clip_ids, clip_paths, interp_feats, y_class, y_score = [], [], [], [], []
        for clip, label in raw:
            feat_path = features_dir / f"{clip.id}.npy"
            if not feat_path.exists():
                continue
            feat = np.load(str(feat_path)).astype(np.float32)
            flow_mag = float(feat[0])
            clip_ids.append(clip.id)
            clip_paths.append(clip.clip_path)
            interp_feats.append(feat)
            y_class.append(class_int_for(label.event_class))
            y_score.append(visual_score_for(label.event_class, flow_mag))
        return clip_ids, clip_paths, interp_feats, y_class, y_score

    train_ids, train_paths, train_interp, train_y_class, train_y_score = build_interp(train_raw)
    val_ids, val_paths, val_interp, val_y_class, val_y_score = build_interp(val_raw)

    if not train_interp or not val_interp:
        log("ERROR: No feature files found. Run /features/extract first.")
        sys.exit(1)

    n_classes = len(HIGHLIGHT_CLASSES)
    in_dim = len(train_interp[0])
    log(f"Train: {len(train_interp)} | Val: {len(val_interp)}, {in_dim} interp features, {n_classes} classes.")

    # ── load backbone ──────────────────────────────────────────────
    backbone_model, transform_fn, feat_dim = load_backbone(args.backbone, BASE_DIR, device)

    # ── pre-extract deep features (one-time, per split) ────────────
    def extract_deep(clip_ids, clip_paths, split_name):
        log(f"Pre-extracting deep features from backbone ({split_name})...")
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
                log(f"  {i+1}/{len(clip_ids)} {split_name} clips done")

        return np.array(deep_feats, dtype=np.float32)

    train_deep = extract_deep(train_ids, train_paths, "train")
    val_deep   = extract_deep(val_ids, val_paths, "val")
    log(f"Deep features: train {train_deep.shape}, val {val_deep.shape}")

    interp_train_arr = np.array(train_interp, dtype=np.float32)
    interp_val_arr   = np.array(val_interp, dtype=np.float32)
    y_train_c = np.array(train_y_class, dtype=np.int64)
    y_val_c   = np.array(val_y_class, dtype=np.int64)
    y_train_s = np.array(train_y_score, dtype=np.float32)
    y_val_s   = np.array(val_y_score, dtype=np.float32)

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

    train_ds = HybridDataset(interp_train_arr, train_deep, y_train_c, y_train_s)
    val_ds   = HybridDataset(interp_val_arr,   val_deep,   y_val_c,   y_val_s)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── fusion model ───────────────────────────────────────────────
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
        "per_class_f1": {}, "confusion_matrix": [], "completed_at": None,
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

    # ── final metrics ──────────────────────────────────────────────
    model.load_state_dict(torch.load(str(output_dir / "best_model.pt"), map_location=device, weights_only=True))
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
        "n_samples": len(train_interp) + len(val_interp),
        "n_train": len(train_interp),
        "n_val": len(val_interp),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    log(
        f"Training complete. Backbone: {args.backbone}, "
        f"Best epoch: {best_epoch}, Val Acc: {m['val_accuracy']:.4f}, Macro F1: {m['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
