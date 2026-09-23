"""Loss terms shared by pre-training and few-shot adaptation.

Keeping the frequency term in one place is what lets the few-shot curriculum
claim to reuse the pre-training objective: train.py and finetune.py call the
same function, so "the adaptation loss is the training loss with a scheduled
weight" is true by construction rather than by convention.
"""
import torch


def freq_l1(pred, target, dim="xy"):
    """L1 distance between the rFFT spectra of pred and target.

    Tensors are (B, X, Y, Z) in normalized space; dim="xy" transforms the two
    lateral axes (the released setting), dim="x" only the first.

    By Parseval a spatial MSE is already an L2 penalty in the frequency domain,
    so the L1 here is not redundant: it raises the relative weight of the
    small-amplitude high-frequency content that the mode-truncated spectral
    convolution under-resolves.
    """
    if dim == "x":
        d = torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)
    elif dim == "xy":
        d = torch.fft.rfft2(pred, dim=(1, 2)) - torch.fft.rfft2(target, dim=(1, 2))
    else:
        raise ValueError(f"freq_dim must be 'x' or 'xy', got {dim!r}")
    return d.abs().mean()
