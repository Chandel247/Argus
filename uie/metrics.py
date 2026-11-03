"""Shared image-quality metrics for the UIE benchmark.

Every method is scored with this one module so numbers are comparable across
methods (survey arXiv:2405.19684 notes metric values shift noticeably between
toolboxes). All functions take uint8 RGB arrays of shape (H, W, 3).

Full-reference: psnr, ssim (scikit-image); lpips (pyiqa).
No-reference:   uiqm, uciqe (implemented here); musiq, niqe, uranker (pyiqa).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.color import rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

FULL_REFERENCE = ("psnr", "ssim", "lpips")
NO_REFERENCE = ("uiqm", "uciqe", "musiq", "niqe", "uranker")
DEEP = ("lpips", "musiq", "niqe", "uranker")  # computed through pyiqa / torch
HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "lpips": False, "uiqm": True,
                    "uciqe": True, "musiq": True, "niqe": False, "uranker": True}


def _check(img: np.ndarray) -> None:
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected uint8 RGB (H, W, 3), got {img.dtype} {img.shape}")


# --------------------------------------------------------------------------
# Full-reference
# --------------------------------------------------------------------------
def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    _check(pred), _check(gt)
    return float(peak_signal_noise_ratio(gt, pred, data_range=255))


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    _check(pred), _check(gt)
    return float(structural_similarity(gt, pred, data_range=255, channel_axis=2))


# --------------------------------------------------------------------------
# UIQM  (Panetta et al., IEEE JOE 2016)
#
# Vectorized port of the widely used Python implementation in FUnIE-GAN
# (xahidbuffon/FUnIE-GAN, utils/uqim_utils.py), including its conventions
# (10x10 blocks, Sobel magnitude rescaled to 255, trimmed-mean indexing), so
# values line up with the many papers that use that code.
# --------------------------------------------------------------------------
_UIQM_C = (0.0282, 0.2953, 3.5753)
_WINDOW = 10


def _trimmed_mean(x: np.ndarray, alpha_l: float = 0.1, alpha_r: float = 0.1) -> float:
    x = np.sort(x)
    k = x.size
    t_l, t_r = int(np.ceil(alpha_l * k)), int(np.floor(alpha_r * k))
    return float(x[t_l + 1:k - t_r].sum() / (k - t_l - t_r))


def _uicm(img: np.ndarray) -> float:
    # float64: summing ~65k float32 values is order-dependent at the 1e-4 level
    r, g, b = (img[..., i].ravel().astype(np.float64) for i in range(3))
    rg, yb = r - g, (r + g) / 2 - b
    mu_rg, mu_yb = _trimmed_mean(rg), _trimmed_mean(yb)
    var_rg, var_yb = np.mean((rg - mu_rg) ** 2), np.mean((yb - mu_yb) ** 2)
    return float(-0.0268 * np.hypot(mu_rg, mu_yb) + 0.1586 * np.sqrt(var_rg + var_yb))


def _blocks(x: np.ndarray, w: int) -> np.ndarray:
    """Crop to whole w x w blocks; return (n_blocks, w*w*C) array."""
    k2, k1 = x.shape[0] // w, x.shape[1] // w
    x = x[:k2 * w, :k1 * w]
    c = 1 if x.ndim == 2 else x.shape[2]
    x = x.reshape(k2, w, k1, w, c).transpose(0, 2, 1, 3, 4)
    return x.reshape(k2 * k1, -1)


def _eme(x: np.ndarray, w: int = _WINDOW) -> float:
    k1, k2 = x.shape[1] / w, x.shape[0] / w
    b = _blocks(x, w)
    mx, mn = b.max(1), b.min(1)
    ok = (mx > 0) & (mn > 0)
    return float(2.0 / (k1 * k2) * np.log(mx[ok] / mn[ok]).sum())


def _sobel_mag(ch: np.ndarray) -> np.ndarray:
    mag = np.hypot(ndimage.sobel(ch, 0), ndimage.sobel(ch, 1))
    peak = mag.max()
    return mag * (255.0 / peak) if peak > 0 else mag


def _uism(img: np.ndarray) -> float:
    lam = (0.299, 0.587, 0.114)
    return float(sum(l * _eme(_sobel_mag(img[..., i]) * img[..., i]) for i, l in enumerate(lam)))


def _uiconm(img: np.ndarray, w: int = _WINDOW) -> float:
    k1, k2 = img.shape[1] / w, img.shape[0] / w
    b = _blocks(img, w)
    mx, mn = b.max(1), b.min(1)
    top, bot = mx - mn, mx + mn
    ok = (top != 0) & (bot != 0)
    ratio = top[ok] / bot[ok]
    return float(-1.0 / (k1 * k2) * (ratio * np.log(ratio)).sum())


def uiqm(img: np.ndarray) -> float:
    _check(img)
    x = img.astype(np.float32)
    c1, c2, c3 = _UIQM_C
    return c1 * _uicm(x) + c2 * _uism(x) + c3 * _uiconm(x)


# --------------------------------------------------------------------------
# UCIQE  (Yang & Sowmya, IEEE TIP 2015)
#
# UCIQE = 0.4680*sigma_c + 0.2745*con_l + 0.2576*mu_s in CIELab, with
#   L normalized to [0, 1] (L/100) and a, b scaled by 1/255 (the scale used
#   by the common OpenCV-uint8 implementations, but correctly zero-centered —
#   those implementations leave the +128 offset in a/b, which inflates chroma),
#   sigma_c = std of chroma, con_l = 99th - 1st percentile of L,
#   mu_s = mean of chroma / L over pixels with L > 0.
# Absolute values differ from other toolboxes; compare methods only within
# this benchmark.
# --------------------------------------------------------------------------
_UCIQE_C = (0.4680, 0.2745, 0.2576)


def uciqe(img: np.ndarray) -> float:
    _check(img)
    lab = rgb2lab(img)
    lum = lab[..., 0] / 100.0
    chroma = np.hypot(lab[..., 1], lab[..., 2]) / 255.0
    sigma_c = chroma.std()
    con_l = np.percentile(lum, 99) - np.percentile(lum, 1)
    sat = np.divide(chroma, lum, out=np.zeros_like(chroma), where=lum > 0)
    c1, c2, c3 = _UCIQE_C
    return float(c1 * sigma_c + c2 * con_l + c3 * sat.mean())


# --------------------------------------------------------------------------
# Deep metrics via pyiqa (lazy: torch is only needed if these are requested)
# --------------------------------------------------------------------------
class DeepMetrics:
    def __init__(self, names, device: str | None = None):
        import pyiqa
        import torch

        if device is None:
            # not MPS: pyiqa's NIQE needs float64, which MPS does not support
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch, self.device = torch, torch.device(device)
        self.models = {n: pyiqa.create_metric(n, device=self.device, as_loss=False) for n in names}

    def _t(self, img: np.ndarray):
        t = self.torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
        return t.unsqueeze(0).to(self.device)

    def __call__(self, pred: np.ndarray, gt: np.ndarray | None = None) -> dict[str, float]:
        out = {}
        with self.torch.no_grad():
            p = self._t(pred)
            for name, m in self.models.items():
                if name in FULL_REFERENCE:
                    if gt is None:
                        continue
                    out[name] = float(m(p, self._t(gt)).item())
                else:
                    out[name] = float(m(p).item())
        return out
