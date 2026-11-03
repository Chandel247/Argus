import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "uie" / "shims"))
from mmcv.ops import modulated_deform_conv2d  # noqa: E402


def test_zero_offset_unit_mask_equals_conv2d():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 17, 23)
    w = torch.randn(6, 8, 3, 3)
    b = torch.randn(6)
    offset = torch.zeros(2, 2 * 3 * 3, 17, 23)
    mask = torch.ones(2, 3 * 3, 17, 23)
    out = modulated_deform_conv2d(x, offset, mask, w, b, 1, 1, 1, 1, 1)
    assert torch.allclose(out, F.conv2d(x, w, b, padding=1), atol=1e-5)


def test_mask_scales_output():
    torch.manual_seed(1)
    x, w = torch.randn(1, 4, 9, 9), torch.randn(4, 4, 3, 3)
    offset = torch.zeros(1, 18, 9, 9)
    half = modulated_deform_conv2d(x, offset, torch.full((1, 9, 9, 9), 0.5), w, None, 1, 1, 1, 1, 1)
    full = modulated_deform_conv2d(x, offset, torch.ones(1, 9, 9, 9), w, None, 1, 1, 1, 1, 1)
    assert torch.allclose(half * 2, full, atol=1e-5)


def test_integer_offset_shifts_sampling():
    # offset (dy=0, dx=+1) on every tap == conv over the input shifted left by one
    torch.manual_seed(2)
    x, w = torch.randn(1, 3, 10, 10), torch.randn(5, 3, 3, 3)
    offset = torch.zeros(1, 18, 10, 10)
    offset[:, 1::2] = 1.0  # (y, x) pairs per tap: x component
    out = modulated_deform_conv2d(x, offset, torch.ones(1, 9, 10, 10), w, None, 1, 1, 1, 1, 1)
    shifted = torch.cat([x[..., 1:], torch.zeros(1, 3, 10, 1)], dim=-1)
    ref = F.conv2d(shifted, w, padding=1)
    # border columns differ (deformable sampling reads real pixels where the
    # shifted reference reads padding); compare the interior
    assert torch.allclose(out[..., 1:-1], ref[..., 1:-1], atol=1e-4)
    # and the x component really is the odd channel: swapping must break it
    swapped = torch.zeros_like(offset)
    swapped[:, 0::2] = 1.0
    out_y = modulated_deform_conv2d(x, swapped, torch.ones(1, 9, 10, 10), w, None, 1, 1, 1, 1, 1)
    assert not torch.allclose(out_y[..., 1:-1], ref[..., 1:-1], atol=1e-4)
