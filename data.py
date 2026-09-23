"""Dataset loading, splitting and normalization -- self-contained, no project deps.

.mat convention (HDF5, key 'data'):
  input.mat :  (B, P, Z, Y, X)  float, P physical input channels
  output.mat:  (B, Z, Y, X)     float, temperature field in kelvin
Both are transposed on load into the tensor contract used by the model,
(B, X, Y, Z, P) and (B, X, Y, Z).
"""
import os

import h5py
import numpy as np
import torch
import torch.nn as nn


def _find_single_mat(data_dir, keyword):
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
             if f.endswith(".mat") and keyword in f.lower()]
    if len(files) != 1:
        raise ValueError(f"expected exactly 1 .mat matching '{keyword}' in {data_dir}, "
                         f"found {len(files)}: {files}")
    return files[0]


def load_mat_pair(folder_path):
    """Read the input/output .mat pair.
    Returns x_all (B, X, Y, Z, P) and y_all (B, X, Y, Z)."""
    with h5py.File(_find_single_mat(folder_path, "input"), "r") as f:
        input_np = f["data"][()]
    with h5py.File(_find_single_mat(folder_path, "output"), "r") as f:
        output_np = f["data"][()]
    if input_np.ndim != 5:
        raise ValueError(f"input must be 5D [B,P,Z,Y,X], got {input_np.shape}")
    if output_np.ndim != 4:
        raise ValueError(f"output must be 4D [B,Z,Y,X], got {output_np.shape}")
    input_np = np.transpose(input_np, (0, 4, 3, 2, 1))   # [B,P,Z,Y,X] -> [B,X,Y,Z,P]
    output_np = np.transpose(output_np, (0, 3, 2, 1))    # [B,Z,Y,X]  -> [B,X,Y,Z]
    return (torch.tensor(input_np, dtype=torch.float32),
            torch.tensor(output_np, dtype=torch.float32))


def split_train_val_test(x_all, y_all, train_ratio=0.8):
    """The leading train_ratio of the set becomes train+val (split 9:1), the rest is test.
    Index-based and deterministic -- the split never depends on a random seed.
    Returns (x_train, y_train, x_val, y_val, x_test, y_test)."""
    total = x_all.shape[0]
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must lie in (0,1), got {train_ratio}")
    trainval = max(2, min(int(total * train_ratio), total - 1))
    n_train = max(1, min(int(trainval * 0.9), trainval - 1))
    return (x_all[:n_train], y_all[:n_train],
            x_all[n_train:trainval], y_all[n_train:trainval],
            x_all[trainval:], y_all[trainval:])


class Normalizer(nn.Module):
    """Standardization whose statistics are estimated on the training split only.

    per_channel=False: one global scalar mean/std (the default).
    per_channel=True with a 5D input (B, X, Y, Z, P): mean/std per input channel;
    a 4D label tensor has no channel axis and falls back to the global scalar.

    Per-channel is mandatory whenever channel magnitudes differ by orders of
    magnitude (the S4/S5 boundary-condition channels, e.g. R_conv against
    power): a single global scale collapses the small-magnitude channels into
    numerical noise.
    """
    def __init__(self, x0=None, trainable=False, per_channel=False,
                 mean=None, std=None):
        super().__init__()
        if mean is None or std is None:
            if x0 is None:
                raise ValueError("Normalizer needs either a sample tensor or mean/std")
            if per_channel and x0.dim() == 5:
                dims = (0, 1, 2, 3)
                mean, std = x0.mean(dim=dims), x0.std(dim=dims)
                std = torch.where(std == 0, torch.ones_like(std), std)
            else:
                mean, std = x0.mean(), x0.std()
        self.mean = nn.Parameter(torch.as_tensor(mean), requires_grad=trainable)
        self.std = nn.Parameter(torch.as_tensor(std), requires_grad=trainable)

    @classmethod
    def from_stats(cls, mean, std):
        """Rebuild a normalizer from stored statistics (used when loading a checkpoint)."""
        return cls(mean=mean, std=std)

    def forward(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean
