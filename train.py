"""FSU-FNO training and evaluation (deterministic evaluation path).

Recipe -- identical to the U-FNO baseline of the benchmark, so last-epoch
metrics are directly comparable:
  Adam(lr=1e-3, weight_decay=1e-4), batch 20, 100 epochs, StepLR(step=2,
  gamma=0.9), MSE in normalized space, inputs and labels standardized with
  statistics taken from the training split only.

The one intended difference is the frequency term (--freq_w), which is what
the "FS" in FSU-FNO refers to:

    loss = MSE(pred, y) + w * | rfft2(pred) - rfft2(y) |.mean()

By Parseval, a spatial MSE is equivalent to an L2 penalty in the frequency
domain. Switching the auxiliary term to an L1 in frequency raises the relative
weight of small-amplitude high-frequency components -- the part that the
spectral convolution, truncated at modes=10, systematically under-resolves.

Calibrating w: at convergence this model reaches MSE = 3.501e-4 and
freq-L1(xy) = 4.793e-1, a ratio of 1369, so w = 7.3e-4 puts the two terms at
equal magnitude. The released checkpoints instead use w = 0.1, where the
frequency term is ~137x the MSE and therefore dominates the objective. That is
a deliberate, empirically chosen setting, not an equal-magnitude one -- with
w = 0.1 the frequency term is effectively the primary loss and the MSE acts as
a regularizer.

Evaluation caveat: run-to-run std on this benchmark is 0.086 K RMSE (n = 7,
range 0.5165-0.7577). No single-run comparison is meaningful; report mean +/-
std over seeds.

Usage:
  python train.py --data <dataset_dir> --out <ckpt_dir> [--seed N] [--val_select]
<dataset_dir> must contain input.mat and output.mat (see data.py).
The checkpoint is written by checkpoint.save_checkpoint (tensors only).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from checkpoint import save_checkpoint
from data import load_mat_pair, split_train_val_test, Normalizer
from fsu_fno import FSUFNO
from losses import freq_l1
from metrics import compute_metrics


def _evaluate(model, xn_eval, y_eval, yn, batch_size, device):
    """Deterministic inference -> denormalize back to kelvin -> six metrics."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, xn_eval.shape[0], batch_size):
            preds.append(yn.inverse(model(xn_eval[i:i + batch_size].to(device)).cpu()))
    out = torch.cat(preds, 0)
    pred = out.permute(0, 3, 1, 2).numpy()              # (B,X,Y,Z) -> (B,Z,X,Y)
    label = y_eval.permute(0, 3, 1, 2).numpy()
    return compute_metrics(pred, label, topk=50)


def _val_loss(model, xn_va, yn_va, batch_size, device):
    """MSE in normalized space -- same scale as the training loss."""
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, xn_va.shape[0], batch_size):
            xb = xn_va[i:i + batch_size].to(device)
            yb = yn_va[i:i + batch_size].to(device)
            tot += F.mse_loss(model(xb), yb, reduction="sum").item()
            n += yb.numel()
    return tot / max(n, 1)


def train(data_dir, out_dir, epochs=100, width=36, modes=10, batch_size=20,
          lr=1e-3, weight_decay=1e-4, train_ratio=0.8, device="cuda", seed=0,
          val_select=False, lr_step=2, lr_gamma=0.9, freq_w=0.0, freq_dim="xy",
          per_channel_norm=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_all, y_all = load_mat_pair(data_dir)
    xtr, ytr, xva, yva, xte, yte = split_train_val_test(x_all, y_all, train_ratio)
    P, Z = int(x_all.shape[-1]), int(x_all.shape[-2])
    modes3 = min(modes, Z // 2 + 1)

    xn, yn = Normalizer(xtr, per_channel=per_channel_norm), Normalizer(ytr)
    model = FSUFNO(modes1=modes, modes2=modes, modes3=modes3, width=width,
                   in_channels=P).to(device)
    print(f"[FSU-FNO] P={P} Z={Z} modes3={modes3} train={xtr.shape[0]} val={xva.shape[0]} "
          f"test={xte.shape[0]} seed={seed} params={model.count_params()}", flush=True)
    print(f"[FSU-FNO] epochs={epochs} StepLR(step={lr_step}, gamma={lr_gamma}) "
          f"final_lr={lr * lr_gamma ** (epochs // lr_step):.3e} val_select={val_select} "
          f"freq_w={freq_w} freq_dim={freq_dim} per_channel_norm={per_channel_norm}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, foreach=False)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_step, gamma=lr_gamma)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xn(xtr), yn(ytr)),
        batch_size=batch_size, shuffle=True)
    xn_va, yn_va = xn(xva), yn(yva)
    best = {"val": float("inf"), "ep": -1, "state": None}

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot, tot_mse, tot_fq, nb = 0.0, 0.0, 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            mse = F.mse_loss(pred, yb)                  # MSE in normalized space
            loss = mse
            fq = torch.zeros((), device=pred.device)
            if freq_w > 0.0:
                fq = freq_l1(pred, yb, freq_dim)        # L1 in the frequency domain
                loss = loss + freq_w * fq
            loss.backward()
            opt.step()
            tot += loss.item()
            tot_mse += mse.item()
            tot_fq += float(fq)
            nb += 1
        sched.step()

        if val_select:
            vl = _val_loss(model, xn_va, yn_va, batch_size, device)
            if vl < best["val"]:
                best = {"val": vl, "ep": ep,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            if ep == 1 or ep % 10 == 0:
                print(f"[FSU-FNO] ep {ep}/{epochs} loss={tot / nb:.4f} mse={tot_mse / nb:.6f} "
                      f"fq={tot_fq / nb:.4f} val={vl:.6f} "
                      f"(best ep{best['ep']} {best['val']:.6f})", flush=True)
        elif ep == 1 or ep % 10 == 0:
            print(f"[FSU-FNO] ep {ep}/{epochs} loss={tot / nb:.4f} mse={tot_mse / nb:.6f} "
                  f"fq={tot_fq / nb:.4f}", flush=True)
    train_time_s = time.time() - t0
    print(f"[FSU-FNO] train time {train_time_s:.1f}s", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    m = _evaluate(model, xn(xte), yte, yn, batch_size, device)
    m["train_time_s"] = round(train_time_s, 1)
    m["train_time_per_epoch_s"] = round(train_time_s / max(epochs, 1), 2)
    m["epochs"] = epochs
    m["params"] = model.count_params()
    m["n_train"] = int(xtr.shape[0]); m["n_test"] = int(xte.shape[0])
    meta = {"epochs": epochs, "seed": seed, "freq_w": freq_w, "freq_dim": freq_dim,
            "per_channel_norm": per_channel_norm, "lr": lr, "lr_step": lr_step,
            "lr_gamma": lr_gamma, "batch_size": batch_size,
            "data": os.path.basename(os.path.normpath(data_dir))}
    save_checkpoint(os.path.join(out_dir, "model.pt"), model, xn, yn, meta)
    json.dump(m, open(os.path.join(out_dir, "test_metrics.json"), "w"), indent=2)
    print("[FSU-FNO] test metrics (last epoch):", flush=True)
    for k, v in m.items():
        print(f"   {k}: {v:.4f}", flush=True)

    if val_select and best["state"] is not None:
        model.load_state_dict(best["state"])
        mb = _evaluate(model, xn(xte), yte, yn, batch_size, device)
        save_checkpoint(os.path.join(out_dir, "model_best.pt"), model, xn, yn,
                        dict(meta, best_epoch=best["ep"]))
        mb["_best_epoch"] = best["ep"]
        mb["train_time_s"] = round(train_time_s, 1)
        mb["train_time_per_epoch_s"] = round(train_time_s / max(epochs, 1), 2)
        mb["epochs"] = epochs
        mb["params"] = model.count_params()
        json.dump(mb, open(os.path.join(out_dir, "test_metrics_best.json"), "w"), indent=2)
        print(f"[FSU-FNO] test metrics (best-val, ep{best['ep']}):", flush=True)
        for k, v in mb.items():
            print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}", flush=True)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory holding input.mat and output.mat")
    ap.add_argument("--out", default="./checkpoints/fsu_fno")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--width", type=int, default=36)
    ap.add_argument("--modes", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--val_select", action="store_true",
                    help="also keep the best-validation checkpoint (off = last-epoch only)")
    ap.add_argument("--lr_step", type=int, default=2)
    ap.add_argument("--lr_gamma", type=float, default=0.9)
    ap.add_argument("--freq_w", type=float, default=0.0,
                    help="weight of the frequency-domain L1 term; 0 disables it. "
                         "0.1 reproduces the released checkpoints; 7.3e-4 is the "
                         "equal-magnitude calibration")
    ap.add_argument("--freq_dim", default="xy", choices=["x", "xy"])
    ap.add_argument("--per_channel_norm", action="store_true",
                    help="standardize inputs per channel (labels stay global). Mandatory "
                         "for datasets whose channels differ by orders of magnitude (S4/S5)")
    args = ap.parse_args()
    train(args.data, args.out, epochs=args.epochs, width=args.width, modes=args.modes,
          batch_size=args.batch_size, device=args.device, seed=args.seed,
          val_select=args.val_select, lr_step=args.lr_step, lr_gamma=args.lr_gamma,
          freq_w=args.freq_w, freq_dim=args.freq_dim, per_channel_norm=args.per_channel_norm)
