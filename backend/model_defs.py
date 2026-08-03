"""Model architecture definitions, lifted out of each trainer's main() so
they're importable by both the training scripts and inference.py without
duplicating class bodies. Layer names/order are unchanged from the original
nested versions, so existing best_model.pt/last_model.pt checkpoints load
without modification.

Also provides single-clip preprocessing helpers for inference — training's
Dataset.__getitem__ bodies are batched-with-label and not directly reusable
for a one-off prediction.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


def read_video_compat(clip_path):
    """Read video with cv2. Returns uint8 tensor (T, H, W, C) in RGB."""
    import cv2
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


# ---------------------------------------------------------------------------
# R3D-18
# ---------------------------------------------------------------------------

R3D_N_FRAMES = 16
R3D_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
R3D_STD  = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)


def sample_frame_indices(total_frames: int, window_size_s: int, n_frames: int):
    """Short windows (8s): uniform sampling — action can be anywhere.
    Long windows (16s, 32s): dense-end sampling — action resolves in second half."""
    if total_frames <= n_frames:
        return np.arange(total_frames)
    if window_size_s <= 8:
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    else:
        n_first = n_frames // 3
        n_second = n_frames - n_first
        first_half  = np.linspace(0, total_frames // 2, n_first,  dtype=int)
        second_half = np.linspace(total_frames // 2, total_frames - 1, n_second, dtype=int)
        indices = np.concatenate([first_half, second_half])
    return indices


class R3DHighlightModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        from torchvision.models.video import r3d_18, R3D_18_Weights
        try:
            backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        except OSError:
            backbone = r3d_18(weights=None)
        self.backbone   = nn.Sequential(*list(backbone.children())[:-1])
        self.class_head = nn.Linear(512, num_classes)
        self.score_head = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return self.class_head(feat), self.score_head(feat).squeeze(1)


def preprocess_r3d_clip(clip_path, window_size_s: int) -> torch.Tensor:
    """Single clip -> (1, 3, 16, 112, 112) normalized tensor."""
    video = read_video_compat(clip_path)
    T = video.shape[0]
    if T == 0:
        return torch.zeros(1, 3, R3D_N_FRAMES, 112, 112)
    indices = sample_frame_indices(T, window_size_s, R3D_N_FRAMES)
    if len(indices) < R3D_N_FRAMES:
        pad = np.full(R3D_N_FRAMES - len(indices), len(indices) - 1, dtype=int)
        indices = np.concatenate([indices, pad])
    indices = indices[:R3D_N_FRAMES]
    frames = video[indices].permute(0, 3, 1, 2).float() / 255.0
    frames = torch.stack([TF.resize(f, [112, 112]) for f in frames])
    frames = frames.permute(1, 0, 2, 3)
    frames = (frames - R3D_MEAN) / R3D_STD
    return frames.unsqueeze(0)


# ---------------------------------------------------------------------------
# VideoMAE-base
# ---------------------------------------------------------------------------

VIDEOMAE_N_FRAMES = 16
VIDEOMAE_IMG_SIZE = 224
VIDEOMAE_MEAN = torch.tensor([0.5, 0.5, 0.5])
VIDEOMAE_STD  = torch.tensor([0.5, 0.5, 0.5])


def sample_frames(total_frames, n_frames):
    """Uniform sampling — VideoMAE always uses a fixed 16-frame window."""
    if total_frames <= n_frames:
        return list(range(total_frames)) + [total_frames - 1] * (n_frames - total_frames)
    return list(map(int, np.linspace(0, total_frames - 1, n_frames)))


class VideoMAEHighlightModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        from transformers import VideoMAEModel, VideoMAEConfig
        model_name = "MCG-NJU/videomae-base"
        try:
            self.backbone = VideoMAEModel.from_pretrained(model_name)
        except Exception:
            cfg = VideoMAEConfig()
            self.backbone = VideoMAEModel(cfg)
        hidden_size = self.backbone.config.hidden_size  # 768 for base; auto-detected
        self.class_head = nn.Linear(hidden_size, num_classes)
        self.score_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, pixel_values):
        # pixel_values: (B, T, C, H, W) — VideoMAE expects this format
        out  = self.backbone(pixel_values=pixel_values)
        feat = out.last_hidden_state.mean(dim=1)  # (B, hidden) — temporal mean pooling
        return self.class_head(feat), self.score_head(feat).squeeze(1)


def preprocess_videomae_clip(clip_path) -> torch.Tensor:
    """Single clip -> (1, 16, 3, 224, 224) normalized tensor."""
    video = read_video_compat(clip_path)
    T = video.shape[0]
    if T == 0:
        return torch.zeros(1, VIDEOMAE_N_FRAMES, 3, VIDEOMAE_IMG_SIZE, VIDEOMAE_IMG_SIZE)
    indices = sample_frames(T, VIDEOMAE_N_FRAMES)
    frames = video[indices].permute(0, 3, 1, 2).float() / 255.0
    frames = torch.stack([TF.resize(f, [VIDEOMAE_IMG_SIZE, VIDEOMAE_IMG_SIZE]) for f in frames])
    for c_idx in range(3):
        frames[:, c_idx] = (frames[:, c_idx] - VIDEOMAE_MEAN[c_idx]) / VIDEOMAE_STD[c_idx]
    return frames.unsqueeze(0)


# ---------------------------------------------------------------------------
# SlowFast R50
# ---------------------------------------------------------------------------

SLOW_FRAMES = 8
FAST_FRAMES = 32   # alpha=4: fast = 4 x slow
SLOWFAST_IMG_SIZE = 224
SLOWFAST_MEAN = torch.tensor([0.45, 0.45, 0.45])
SLOWFAST_STD  = torch.tensor([0.225, 0.225, 0.225])

try:
    from pytorchvideo.models import create_slowfast  # noqa: F401
    _USE_PYTORCHVIDEO = True
except ImportError:
    _USE_PYTORCHVIDEO = False


def sample_pathway(total_frames, n_frames):
    if total_frames <= n_frames:
        idx = list(range(total_frames)) + [total_frames - 1] * (n_frames - total_frames)
        return idx
    return list(map(int, np.linspace(0, total_frames - 1, n_frames)))


class SlowFastHighlightModel(nn.Module):
    """SlowFast R50 backbone with dual classification + regression head."""

    def __init__(self, num_classes):
        super().__init__()
        loaded = False

        if _USE_PYTORCHVIDEO:
            try:
                from pytorchvideo.models.hub import slowfast_r50
                base = slowfast_r50(pretrained=True)
                self.backbone = nn.Sequential(*list(base.blocks[:-1]))
                feat_dim = 2304
                loaded = True
            except Exception:
                pass

        if not loaded:
            try:
                base = torch.hub.load(
                    "facebookresearch/pytorchvideo", "slowfast_r50",
                    pretrained=True, verbose=False,
                )
                self.backbone = nn.Sequential(*list(base.blocks[:-1]))
                feat_dim = 2304
                loaded = True
            except Exception:
                base = torch.hub.load(
                    "facebookresearch/pytorchvideo", "slowfast_r50",
                    pretrained=False, verbose=False,
                )
                self.backbone = nn.Sequential(*list(base.blocks[:-1]))
                feat_dim = 2304

        self.pool       = nn.AdaptiveAvgPool3d(1)
        self.class_head = nn.Linear(feat_dim, num_classes)
        self.score_head = nn.Sequential(nn.Linear(feat_dim, 1), nn.Sigmoid())

    def forward(self, slow, fast):
        # SlowFast backbone expects a list [slow, fast]
        feat = self.backbone([slow, fast])
        if isinstance(feat, (list, tuple)):
            feat = torch.cat([self.pool(f).flatten(1) for f in feat], dim=1)
        else:
            feat = self.pool(feat).flatten(1)
        return self.class_head(feat), self.score_head(feat).squeeze(1)


def preprocess_slowfast_clip(clip_path):
    """Single clip -> (slow, fast) tensors, each batched (1, 3, T, 224, 224)."""
    video = read_video_compat(clip_path)
    T = video.shape[0]
    if T == 0:
        slow = torch.zeros(1, 3, SLOW_FRAMES, SLOWFAST_IMG_SIZE, SLOWFAST_IMG_SIZE)
        fast = torch.zeros(1, 3, FAST_FRAMES, SLOWFAST_IMG_SIZE, SLOWFAST_IMG_SIZE)
        return slow, fast

    slow_idx = sample_pathway(T, SLOW_FRAMES)
    fast_idx = sample_pathway(T, FAST_FRAMES)

    def build_tensor(indices):
        frames = video[indices].permute(0, 3, 1, 2).float() / 255.0
        frames = torch.stack([TF.resize(f, [SLOWFAST_IMG_SIZE, SLOWFAST_IMG_SIZE]) for f in frames])
        frames = frames.permute(1, 0, 2, 3)
        for c_idx in range(3):
            frames[c_idx] = (frames[c_idx] - SLOWFAST_MEAN[c_idx]) / SLOWFAST_STD[c_idx]
        return frames.unsqueeze(0)

    return build_tensor(slow_idx), build_tensor(fast_idx)


# ---------------------------------------------------------------------------
# Interpretable Features + MLP
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Interpretable + Deep Features Hybrid fusion head
# ---------------------------------------------------------------------------

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
