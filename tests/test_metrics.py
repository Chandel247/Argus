import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from uie import metrics as M  # noqa: E402

RNG = np.random.default_rng(0)


def underwater_like(h=256, w=256, seed=0):
    """Smooth, blue-green-cast image with texture — more realistic than pure noise."""
    rng = np.random.default_rng(seed)
    base = ndimage.gaussian_filter(rng.random((h, w)), 6)
    base = (base - base.min()) / (np.ptp(base) + 1e-9)
    tex = rng.random((h, w)) * 0.15
    r = 0.25 * base + tex
    g = 0.75 * base + 0.15 + tex
    b = 0.65 * base + 0.25 + tex
    return (np.clip(np.stack([r, g, b], -1), 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Loop-based transcription of FUnIE-GAN utils/uqim_utils.py (the reference
# that M.uiqm vectorizes). Kept deliberately literal.
# ---------------------------------------------------------------------------
def ref_mu_a(x, alpha_L=0.1, alpha_R=0.1):
    x = sorted(x)
    K = len(x)
    T_a_L = math.ceil(alpha_L * K)
    T_a_R = math.floor(alpha_R * K)
    weight = 1 / (K - T_a_L - T_a_R)
    s = int(T_a_L + 1)
    e = int(K - T_a_R)
    return weight * sum(x[s:e])


def ref_s_a(x, mu):
    return sum((p - mu) ** 2 for p in x) / len(x)


def ref_uicm(x):
    R, G, B = x[:, :, 0].flatten(), x[:, :, 1].flatten(), x[:, :, 2].flatten()
    RG = R - G
    YB = ((R + G) / 2) - B
    mu_a_RG, mu_a_YB = ref_mu_a(RG), ref_mu_a(YB)
    s_a_RG, s_a_YB = ref_s_a(RG, mu_a_RG), ref_s_a(YB, mu_a_YB)
    l = math.sqrt(mu_a_RG ** 2 + mu_a_YB ** 2)
    r = math.sqrt(s_a_RG + s_a_YB)
    return (-0.0268 * l) + (0.1586 * r)


def ref_sobel(x):
    dx = ndimage.sobel(x, 0)
    dy = ndimage.sobel(x, 1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)
    return mag


def ref_eme(x, window_size):
    k1 = x.shape[1] / window_size
    k2 = x.shape[0] / window_size
    w = 2.0 / (k1 * k2)
    bs = window_size
    x = x[:bs * int(k2), :bs * int(k1)]
    val = 0
    for l in range(int(k1)):
        for k in range(int(k2)):
            block = x[k * bs:bs * (k + 1), l * bs:bs * (l + 1)]
            max_, min_ = np.max(block), np.min(block)
            if min_ == 0.0 or max_ == 0.0:
                val += 0
            else:
                val += math.log(max_ / min_)
    return w * val


def ref_uism(x):
    R, G, B = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    r = ref_eme(np.multiply(ref_sobel(R), R), 10)
    g = ref_eme(np.multiply(ref_sobel(G), G), 10)
    b = ref_eme(np.multiply(ref_sobel(B), B), 10)
    return 0.299 * r + 0.587 * g + 0.114 * b


def ref_uiconm(x, window_size):
    k1 = x.shape[1] / window_size
    k2 = x.shape[0] / window_size
    w = -1.0 / (k1 * k2)
    bs = window_size
    x = x[:bs * int(k2), :bs * int(k1)]
    alpha = 1
    val = 0
    for l in range(int(k1)):
        for k in range(int(k2)):
            block = x[k * bs:bs * (k + 1), l * bs:bs * (l + 1), :]
            max_, min_ = np.max(block), np.min(block)
            top, bot = max_ - min_, max_ + min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0:
                val += 0.0
            else:
                val += alpha * math.pow((top / bot), alpha) * math.log(top / bot)
    return w * val


def ref_uiqm(x):
    x = x.astype(np.float32)
    return 0.0282 * ref_uicm(x) + 0.2953 * ref_uism(x) + 3.5753 * ref_uiconm(x, 10)


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(256, 256), (240, 320), (97, 131)])
def test_uiqm_matches_reference(shape):
    img = underwater_like(*shape, seed=shape[0])
    assert M.uiqm(img) == pytest.approx(ref_uiqm(img), rel=1e-5)


@pytest.mark.parametrize("fn", [M._uicm, M._uism])
def test_uiqm_components_match_reference(fn):
    img = underwater_like(seed=3).astype(np.float32)
    ref = {M._uicm: ref_uicm, M._uism: ref_uism}[fn]
    # UICM is computed in float64; compare against the reference run in float64
    ref_in = img.astype(np.float64) if fn is M._uicm else img
    assert fn(img) == pytest.approx(ref(ref_in), rel=1e-6)


def test_uiqm_handles_flat_image():
    flat = np.full((64, 64, 3), 128, np.uint8)
    assert np.isfinite(M.uiqm(flat))
    assert np.isfinite(M.uiqm(np.zeros((64, 64, 3), np.uint8)))


def test_psnr_ssim_identity_and_ordering():
    gt = underwater_like(seed=1)
    assert M.ssim(gt, gt) == pytest.approx(1.0)
    with pytest.warns(RuntimeWarning):
        assert M.psnr(gt, gt) == np.inf
    mild = np.clip(gt.astype(int) + RNG.integers(-5, 6, gt.shape), 0, 255).astype(np.uint8)
    heavy = np.clip(gt.astype(int) + RNG.integers(-40, 41, gt.shape), 0, 255).astype(np.uint8)
    assert M.psnr(mild, gt) > M.psnr(heavy, gt)
    assert M.ssim(mild, gt) > M.ssim(heavy, gt)


def test_uciqe_range_and_colour_sensitivity():
    img = underwater_like(seed=2)
    gray = np.repeat(img.mean(2, keepdims=True), 3, 2).astype(np.uint8)
    u_img, u_gray = M.uciqe(img), M.uciqe(gray)
    assert 0 < u_gray < u_img < 1.5
    # a grayscale image has ~zero chroma, so only the contrast term remains
    lum = M.rgb2lab(gray)[..., 0] / 100
    con = np.percentile(lum, 99) - np.percentile(lum, 1)
    assert u_gray == pytest.approx(0.2745 * con, abs=1e-3)


def test_rejects_non_uint8():
    with pytest.raises(ValueError):
        M.uiqm(np.zeros((8, 8, 3), np.float32))
