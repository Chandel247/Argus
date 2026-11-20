"""Adapter for our own enhancers (Raghav): the FUnIE-GAN-style U-Net generator,
trained twice on the same 750 UIEB pairs, once as a conditional GAN
(LSGAN + 0.7 L1 + 0.3 VGG-19) and once with L1 only (the ablation).

Both runs share this generator, so one adapter serves the names "gan" and
"l1" in the registry; only the weights differ. Training code: enhance/train.py.

Checkpoints from enhance/train.py are dicts {"G": state_dict, "epoch", "val_psnr", "mode"};
a bare state_dict is accepted too.

    from uie import baselines
    enhance = baselines.get("gan").Enhancer("gan_best.pt", device="cuda")
    out = enhance(img)                    # 256x256, the benchmark protocol (same as Semi-UIR)
    out = enhance(img, size=None)         # native resolution: pad to a multiple of 32, crop back
    out = enhance(img, size=None, max_side=512)   # keep aspect, longer side <= 512 (demo)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def _down(cin, cout, bn=True):
    layers = [nn.Conv2d(cin, cout, 4, 2, 1)]
    if bn:
        layers.append(nn.BatchNorm2d(cout))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _up(cin, cout):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Generator(nn.Module):
    """U-Net: 5 down / 5 up stages, skip connections, tanh output in [-1, 1]. 7.02 M parameters.
    Layer names must stay as they are, or the trained checkpoints will not load."""

    def __init__(self):
        super().__init__()
        self.d1 = _down(3, 32, bn=False); self.d2 = _down(32, 128); self.d3 = _down(128, 256)
        self.d4 = _down(256, 256); self.d5 = _down(256, 256)
        self.u1 = _up(256, 256); self.u2 = _up(512, 256); self.u3 = _up(512, 128); self.u4 = _up(256, 32)
        self.out = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3); d5 = self.d5(d4)
        u = torch.cat([self.u1(d5), d4], 1); u = torch.cat([self.u2(u), d3], 1)
        u = torch.cat([self.u3(u), d2], 1); u = torch.cat([self.u4(u), d1], 1)
        return self.out(u)


def build_model():
    return Generator()


def extract_state_dict(ckpt) -> dict:
    if isinstance(ckpt, dict) and isinstance(ckpt.get("G"), dict):
        ckpt = ckpt["G"]
    return {k.removeprefix("module."): v for k, v in ckpt.items()}


def load_weights(model, path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(extract_state_dict(ckpt))
    return model


class Enhancer:
    """RGB uint8 HxWx3 in, RGB uint8 out (same interface as the Semi-UIR adapter)."""

    def __init__(self, weights, device: str = "cpu", seed: int = 0):
        torch.manual_seed(seed)
        self.device = device
        self.model = load_weights(build_model(), weights, map_location=device).to(device).eval()

    @torch.no_grad()
    def __call__(self, img: np.ndarray, size: int | None = 256, max_side: int | None = None) -> np.ndarray:
        """size: square resize (benchmark protocol, as in enhance/train.py). With size=None the
        whole frame is kept: optionally resized so the longer side is <= max_side, then padded
        (reflect) to a multiple of 32 for the five stride-2 stages and cropped back, so the output
        has exactly the input's (or the resized) shape and detection boxes stay aligned."""
        im = Image.fromarray(img).convert("RGB")
        if size:
            im = im.resize((size, size), Image.BICUBIC)
        elif max_side and max(im.size) > max_side:
            s = max_side / max(im.size)
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BICUBIC)
        a = np.asarray(im)
        h, w = a.shape[:2]
        x = torch.from_numpy(a.copy()).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).to(self.device)
        x = F.pad(x, (0, (32 - w % 32) % 32, 0, (32 - h % 32) % 32), mode="reflect")
        y = self.model(x)[:, :, :h, :w].clamp(-1, 1)
        return ((y[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
