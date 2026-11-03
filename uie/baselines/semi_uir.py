"""Semi-UIR (Huang et al., CVPR 2023) adapter.

Official code: https://github.com/Huang-ShiRui/Semi-UIR, pinned to COMMIT and
cloned into baselines/semi_uir (not vendored: the repo has no license file).
Compatibility patches applied at import time, without editing the clone:
  - mmcv.ops.modulated_deform_conv2d -> torchvision (uie/shims/mmcv)
  - PIL.Image.ANTIALIAS (removed in Pillow 10) -> Image.LANCZOS
  - torchvision vgg16/vgg19(pretrained=True) -> weights=DEFAULT

The network takes (image, illumination map) and returns (image, gradient).
Its sparse attention draws random hash rotations every forward pass, so
inference is seeded for reproducible outputs.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

NAME = "semi_uir"
REPO_URL = "https://github.com/Huang-ShiRui/Semi-UIR.git"
COMMIT = "14eb75d78a405cdc1f259a5ccc84dc90a63e8c13"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = ROOT / "baselines" / NAME
SHIMS = ROOT / "uie" / "shims"
MULTIPLE = 8  # input sides must be divisible by this (two 2x downsamples + margin)

_activated: Path | None = None


def ensure_repo(dest: Path = DEFAULT_REPO) -> Path:
    if not (dest / "model.py").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", REPO_URL, str(dest)], check=True)
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != COMMIT:
        subprocess.run(["git", "-C", str(dest), "checkout", "-q", COMMIT], check=True)
    return dest


def activate(repo: Path | None = None) -> Path:
    """Apply compatibility patches and put the official code on sys.path."""
    global _activated
    if _activated:
        return _activated
    repo = ensure_repo(repo or DEFAULT_REPO)

    from PIL import Image
    if not hasattr(Image, "ANTIALIAS"):
        Image.ANTIALIAS = Image.LANCZOS

    import torchvision.models as tvm

    def _compat(fn, weights):
        def wrapped(*args, pretrained=False, **kw):
            if pretrained and "weights" not in kw:
                kw["weights"] = weights
            return fn(*args, **kw)
        return wrapped

    tvm.vgg16 = _compat(tvm.vgg16, tvm.VGG16_Weights.DEFAULT)
    tvm.vgg19 = _compat(tvm.vgg19, tvm.VGG19_Weights.DEFAULT)

    for p in (str(repo), str(SHIMS)):
        if p not in sys.path:
            sys.path.insert(0, p)
    _activated = repo
    return repo


def luminance_estimation(img: np.ndarray) -> np.ndarray:
    """Illumination map, identical to the official estimate_illumination.py
    (which cannot be imported: it runs a script at module level)."""
    import cv2

    sigma_list = [15, 60, 90]
    img = np.uint8(np.array(img))
    illuminance = np.ones_like(img).astype(np.float32)
    for sigma in sigma_list:
        illuminance1 = np.log10(cv2.GaussianBlur(img, (0, 0), sigma) + 1e-8)
        illuminance1 = np.clip(illuminance1, 0, 255)
        illuminance = illuminance + illuminance1
    illuminance = illuminance / 3
    L = (illuminance - np.min(illuminance)) / (np.max(illuminance) - np.min(illuminance) + 1e-6)
    return np.uint8(L * 255)


def build_model():
    activate()
    from model import AIMnet  # official
    return AIMnet()


def extract_state_dict(ckpt) -> dict:
    """Accepts official {'state_dict': ...} checkpoints, our checkpoints, or a
    bare state dict; strips DataParallel 'module.' prefixes."""
    if isinstance(ckpt, dict):
        for key in ("student", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    return {k.removeprefix("module."): v for k, v in ckpt.items()}


def load_weights(model, path, map_location="cpu"):
    import torch
    # weights_only: never unpickle arbitrary objects from checkpoint files
    sd = extract_state_dict(torch.load(path, map_location=map_location, weights_only=True))
    model.load_state_dict(sd, strict=True)
    return model


class Enhancer:
    """Inference wrapper: uint8 RGB in, uint8 RGB out."""

    def __init__(self, weights, device: str = "cpu", seed: int = 0):
        import torch
        self.torch, self.device, self.seed = torch, torch.device(device), seed
        self.model = load_weights(build_model(), weights).to(self.device).eval()

    def __call__(self, img: np.ndarray, size: int | None = 256, max_side: int | None = None) -> np.ndarray:
        """size: square resize (benchmark protocol). max_side: keep aspect,
        cap the longer side (demo). The illumination map is computed on the
        original image, then resized alongside it, as in the official pipeline."""
        from PIL import Image
        torch = self.torch

        la = luminance_estimation(img)
        h, w = img.shape[:2]
        if size:
            th, tw = size, size
        else:
            s = min(1.0, (max_side or max(h, w)) / max(h, w))
            th, tw = (max(MULTIPLE, int(round(d * s / MULTIPLE)) * MULTIPLE) for d in (h, w))
        rs = lambda a: np.asarray(Image.fromarray(a).resize((tw, th), Image.LANCZOS))
        x, l = rs(img), rs(la)

        t = lambda a: torch.from_numpy(a).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
        with torch.no_grad():
            torch.manual_seed(self.seed)
            out, _ = self.model(t(x), t(l))
        out = out[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return (out * 255).round().astype(np.uint8)
