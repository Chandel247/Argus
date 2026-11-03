"""Minimal stand-in for mmcv, providing only what baseline repos import.

mmcv-full 1.x does not build against current PyTorch; torchvision ships the
same modulated deformable convolution (DCNv2) op.
"""
