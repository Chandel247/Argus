"""Adapters that run third-party baselines (cloned at a pinned commit into
baselines/<name>/) behind one interface: build_model / load_weights / Enhancer.
"""
from importlib import import_module

REGISTRY = {
    "semi_uir": "uie.baselines.semi_uir",
    "gan": "uie.baselines.funie_gan",   # our enhancer, GAN run (enhance/train.py --mode gan)
    "l1": "uie.baselines.funie_gan",    # same generator, L1-only ablation (--mode l1)
}


def get(name: str):
    if name not in REGISTRY:
        raise KeyError(f"unknown baseline {name!r}; known: {sorted(REGISTRY)}")
    return import_module(REGISTRY[name])
