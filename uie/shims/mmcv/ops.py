from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d


def modulated_deform_conv2d(input, offset, mask, weight, bias=None, stride=1, padding=0,
                            dilation=1, groups=1, deform_groups=1):
    """mmcv.ops.modulated_deform_conv2d via torchvision (same DCNv2 kernel layout).

    torchvision infers `groups` from weight.shape[1] and `deform_groups` from
    offset.shape[1], so those two arguments only need to be consistent.
    """
    if weight.shape[1] * groups != input.shape[1]:
        raise ValueError("groups inconsistent with weight/input channels")
    if offset.shape[1] != 2 * deform_groups * weight.shape[2] * weight.shape[3]:
        raise ValueError("deform_groups inconsistent with offset channels")
    return deform_conv2d(input, offset, weight, bias, stride=_pair(stride), padding=_pair(padding),
                         dilation=_pair(dilation), mask=mask)
