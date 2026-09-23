"""Evaluation metrics -- computed per sample and per channel, then averaged;
R^2 uses the pooled definition (over all pixels of the evaluation set).

preds/labels are (B, C, H, W) or (B, H, W) (a channel axis is added), in kelvin.
"""
import math

import numpy as np


def _r2(pred, label):
    ss_res = float(np.sum((pred - label) ** 2))
    ss_tot = float(np.sum((label - label.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def compute_metrics(preds, labels, topk=50):
    """Return the six benchmark metrics:
      rmse, mae, r2 (pooled), max_absolute_error,
      max_temperature_error = mean over samples of |max(pred) - max(label)|
                              (peak / hot-spot temperature error),
      top_mae               = mean |error| over the topk hottest ground-truth
                              pixels (accuracy where the design actually cares).
    """
    preds, labels = np.asarray(preds, np.float64), np.asarray(labels, np.float64)
    if preds.ndim == 3:                       # (B,H,W) -> (B,1,H,W)
        preds, labels = preds[:, None], labels[:, None]
    B, C = preds.shape[0], preds.shape[1]
    pf = preds.reshape(B, C, -1)
    lf = labels.reshape(B, C, -1)

    rmse = mae = maxae = maxterr = topmae = 0.0
    for b in range(B):
        for c in range(C):
            p, l = pf[b, c], lf[b, c]
            d = p - l
            rmse += math.sqrt(np.mean(d ** 2))
            mae += float(np.mean(np.abs(d)))
            maxae += float(np.max(np.abs(d)))
            maxterr += abs(float(np.max(p)) - float(np.max(l)))
            k = min(topk, p.size)
            if k > 0:
                idx = np.argpartition(l, -k)[-k:]
                topmae += float(np.mean(np.abs(d[idx])))
    n = B * C
    r2 = float(np.mean([_r2(pf[:, c].ravel(), lf[:, c].ravel()) for c in range(C)]))
    return {
        "rmse": rmse / n,
        "mae": mae / n,
        "r2": r2,
        "max_absolute_error": maxae / n,
        "max_temperature_error": maxterr / n,
        "top_mae": topmae / n,
    }
