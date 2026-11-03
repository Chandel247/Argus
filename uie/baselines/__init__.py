"""Adapters that run third-party baselines (cloned at a pinned commit into
baselines/<name>/) behind one interface: build_model / load_weights / Enhancer.
"""
from importlib import import_module

REGISTRY = {"semi_uir": "uie.baselines.semi_uir"}


def get(name: str):
    if name not in REGISTRY:
        raise KeyError(f"unknown baseline {name!r}; known: {sorted(REGISTRY)}")
    return import_module(REGISTRY[name])
