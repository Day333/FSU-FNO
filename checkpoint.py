"""Checkpoint I/O.

A checkpoint stores tensors only -- a model state_dict plus the normalizer
statistics and the architecture arguments needed to rebuild the network. It
never pickles live objects, so a checkpoint keeps loading after the module
layout or the class names change.

Layout:
    {"state_dict": {...},
     "arch":  {"modes1", "modes2", "modes3", "width", "in_channels"},
     "x_mean", "x_std", "y_mean", "y_std",   # tensors; x_* may be per-channel
     "meta":  {...}}                          # free-form provenance
"""

import os

import torch

from data import Normalizer
from fsu_fno import FSUFNO


def save_checkpoint(path, model, x_norm, y_norm, meta=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    b = model.block
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "arch": {"modes1": b.modes1, "modes2": b.modes2, "modes3": b.modes3,
                 "width": b.width, "in_channels": b.in_channels},
        "x_mean": x_norm.mean.detach().cpu(), "x_std": x_norm.std.detach().cpu(),
        "y_mean": y_norm.mean.detach().cpu(), "y_std": y_norm.std.detach().cpu(),
        "meta": meta or {},
    }, path)


def load_checkpoint(path, device="cpu"):
    """-> (x_normalizer, model, y_normalizer), all on `device`, model in eval mode."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in obj:
        raise ValueError(
            f"{path} is not an FSU-FNO checkpoint (no 'state_dict' key). "
            f"Checkpoints written before the tensor-only format was introduced "
            f"must be migrated once before they can be loaded.")
    model = FSUFNO(**obj["arch"])
    model.load_state_dict(obj["state_dict"])
    return (Normalizer.from_stats(obj["x_mean"], obj["x_std"]).to(device),
            model.to(device).eval(),
            Normalizer.from_stats(obj["y_mean"], obj["y_std"]).to(device))
