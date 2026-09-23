"""Few-shot adaptation on level5 (the five structurally unseen packages).

  python finetune.py --shots K [--epochs 50] [--lr 1e-4] [--freq_w 0.1]

Protocol -- identical to the benchmark's few-shot table:
  - Each of the five cases holds 1000 consecutive samples: the first 500 form
    the adaptation pool, the last 500 the holdout. The holdout (2500 samples in
    total) is shared by every K and never overlaps the adaptation samples.
    K-shot takes the first K samples of each case's pool, so the budgets are
    nested (the K=10 set is a subset of the K=50 set, and so on).
  - The starting point is the level4 checkpoint, i.e. the zero-shot weights.
    The normalizers stored in that checkpoint are reused unchanged -- only the
    weights adapt, never the preprocessing.
  - Adam(lr=1e-4, weight_decay=1e-4), batch 20, 50 epochs by default; the
    holdout is scored every --eval_every epochs.
  - K=0 evaluates without training. It differs slightly from the 5000-sample
    zero-shot number because the holdout is only 2500 samples.

Adaptation loss -- the spectral curriculum:

    L_e = MSE(pred, y) + lambda_e * freq_L1(pred, y),
    lambda_e = freq_w * min(1, e / freq_warmup)          (e is 1-based)

  --freq_w 0                       plain MSE, the released setting (default)
  --freq_w 0.1 --freq_warmup 0     the pre-training weight, held constant
  --freq_w 0.1 --freq_warmup 10    linear warm-up over the first 10 epochs

  The two terms fix different errors. A model moved to an unseen package starts
  with a large global temperature offset -- a low-frequency error that the MSE
  removes fastest; the frequency term sharpens boundaries and hot spots, which
  only pays off once that offset is gone. Ramping lambda lets each term act
  when it is the useful one, at no cost in parameters or labels.

  freq_w = 0.1 is the weight the checkpoints were pre-trained with, where the
  frequency term is ~137x the MSE and therefore dominates the objective;
  applying it from epoch 1 on a K-sample budget is what --freq_warmup 0 tests.

Run-to-run spread is not negligible here, so compare arms over several seeds
(see fewshot_sweep.py), never on single runs.

Writes results/fewshot_k{K}.json (plain-MSE seed 0, the legacy name) or
results/fewshot_k{K}_{tag}_s{seed}.json, holding the per-evaluation curve, the
final point and a late-phase stability summary.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from checkpoint import load_checkpoint
from data import load_mat_pair
from losses import freq_l1
from metrics import compute_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("FSU_DATA", os.path.join(HERE, "data"))
N_CASES, PER_CASE, POOL = 5, 1000, 500


def lambda_at(epoch, freq_w, freq_warmup):
    """Weight of the frequency term at epoch e (1-based); warm-up 0 = constant."""
    if freq_w <= 0.0:
        return 0.0
    if freq_warmup and freq_warmup > 0:
        return freq_w * min(1.0, epoch / float(freq_warmup))
    return freq_w


def schedule_tag(freq_w, freq_warmup):
    """Short label for the loss schedule, used in filenames and reports."""
    if freq_w <= 0.0:
        return "mse"
    if freq_warmup and freq_warmup > 0:
        return f"warm{freq_w:g}e{int(freq_warmup)}"
    return f"fixed{freq_w:g}"


def split_indices(shots):
    """-> (adaptation indices, holdout indices) under the nested first-K protocol."""
    ft_idx, ho_idx = [], []
    for c in range(N_CASES):
        base = c * PER_CASE
        ft_idx += list(range(base, base + shots))
        ho_idx += list(range(base + POOL, base + PER_CASE))
    return ft_idx, ho_idx


def summarize_curve(curve, epochs, late_frac=0.6):
    """Final point plus how much holdout RMSE still moves over the late epochs.

    A schedule that only reorders when each loss term acts should show up as a
    flatter tail, so the tail spread is reported next to the final number
    instead of being left for the reader to eyeball.
    """
    late = [c["rmse"] for c in curve if c["epoch"] >= late_frac * epochs]
    trained = [c["rmse"] for c in curve if c["epoch"] > 0] or [curve[-1]["rmse"]]
    return {
        "rmse_final": curve[-1]["rmse"],
        "rmse_best": min(trained),
        "rmse_late_std": float(np.std(late)) if len(late) > 1 else 0.0,
        "rmse_late_range": float(max(late) - min(late)) if len(late) > 1 else 0.0,
        "n_late_evals": len(late),
    }


def evaluate(model, xn, yn, x_ho, y_ho, batch, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, x_ho.shape[0], batch):
            preds.append(yn.inverse(model(xn(x_ho[i:i + batch].to(device)))).cpu())
    out = torch.cat(preds, 0)
    return compute_metrics(out.permute(0, 3, 1, 2).numpy(),
                           y_ho.permute(0, 3, 1, 2).numpy(), topk=50)


def few_shot_run(x_all, y_all, shots, ckpt, epochs=50, lr=1e-4, weight_decay=1e-4,
                 batch=20, eval_every=5, freq_w=0.0, freq_dim="xy", freq_warmup=10,
                 seed=0, device="cuda", log=print, ft_idx=None):
    """One adaptation run. Returns the result dict that gets written to results/.

    ft_idx overrides which samples are adapted on; the default is the protocol's
    first K per case. Anything passed here must come from the adaptation pool
    (the first POOL of each case) or it contaminates the holdout.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    default_ft, ho_idx = split_indices(shots)
    if ft_idx is None:
        ft_idx = default_ft
    else:
        ft_idx = list(ft_idx)
        bad = [i for i in ft_idx if i % PER_CASE >= POOL]
        if bad:
            raise ValueError(f"{len(bad)} adaptation indices fall in the holdout "
                             f"half of their case, e.g. {bad[:5]}")
    x_ho, y_ho = x_all[ho_idx], y_all[ho_idx]
    xn, model, yn = load_checkpoint(ckpt, device)
    tag = schedule_tag(freq_w, freq_warmup)
    log(f"[fewshot] K={shots} seed={seed} loss={tag} finetune={len(ft_idx)} "
        f"holdout={len(ho_idx)} lr={lr} epochs={epochs} ckpt={ckpt}")

    curve = [dict(epoch=0, **evaluate(model, xn, yn, x_ho, y_ho, batch, device))]
    log(f"  ep 0 (zero-shot) rmse={curve[0]['rmse']:.2f}")

    t0 = time.time()
    if shots > 0:
        x_ft, y_ft = x_all[ft_idx], y_all[ft_idx]
        # the normalizers live on `device`, so normalize there and keep the
        # adaptation set on the host; batches move back to the device in the loop
        with torch.no_grad():
            x_ft_n = xn(x_ft.to(device)).cpu()
            y_ft_n = yn(y_ft.to(device)).cpu()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_ft_n, y_ft_n),
            batch_size=batch, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay, foreach=False)
        for ep in range(1, epochs + 1):
            lam = lambda_at(ep, freq_w, freq_warmup)
            model.train()
            tot, tot_mse, tot_fq, nb = 0.0, 0.0, 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                pred = model(xb)
                mse = F.mse_loss(pred, yb)              # MSE in normalized space
                loss = mse
                fq = torch.zeros((), device=pred.device)
                if lam > 0.0:
                    fq = freq_l1(pred, yb, freq_dim)    # same term as pre-training
                    loss = loss + lam * fq
                loss.backward()
                opt.step()
                tot += loss.item()
                tot_mse += mse.item()
                tot_fq += float(fq)
                nb += 1
            if ep % eval_every == 0 or ep == epochs:
                m = evaluate(model, xn, yn, x_ho, y_ho, batch, device)
                curve.append(dict(epoch=ep, train_loss=tot / max(nb, 1),
                                  train_mse=tot_mse / max(nb, 1),
                                  train_freq=tot_fq / max(nb, 1), lam=lam, **m))
                log(f"  ep {ep} lam={lam:.3f} loss={tot / max(nb, 1):.4f} "
                    f"mse={tot_mse / max(nb, 1):.6f} rmse={m['rmse']:.3f} "
                    f"maxae={m['max_absolute_error']:.3f} top50={m['top_mae']:.3f}")

    return {"model": "FSU-FNO", "shots": shots, "seed": seed, "loss_schedule": tag,
            "ft_idx": list(ft_idx),
            "freq_w": freq_w, "freq_dim": freq_dim, "freq_warmup": freq_warmup,
            "n_finetune": len(ft_idx), "n_holdout": len(ho_idx), "epochs": epochs,
            "lr": lr, "finetune_time_s": round(time.time() - t0, 1),
            "curve": curve, "final": curve[-1],
            "stability": summarize_curve(curve, epochs)}


def result_path(shots, freq_w, freq_warmup, seed, out_dir=None):
    """Legacy name for the plain-MSE seed-0 run, tagged names for everything else."""
    out_dir = out_dir or os.path.join(HERE, "results")
    tag = schedule_tag(freq_w, freq_warmup)
    name = (f"fewshot_k{shots}.json" if tag == "mse" and seed == 0
            else f"fewshot_k{shots}_{tag}_s{seed}.json")
    return os.path.join(out_dir, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, required=True,
                    help="labeled samples per case (K); 0 evaluates the zero-shot baseline")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "level4", "model.pt"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--freq_w", type=float, default=0.0,
                    help="target weight of the frequency-domain L1 term; 0 (default) "
                         "keeps the released plain-MSE adaptation, 0.1 is the "
                         "pre-training weight")
    ap.add_argument("--freq_dim", default="xy", choices=["x", "xy"])
    ap.add_argument("--freq_warmup", type=int, default=10,
                    help="epochs over which lambda ramps linearly from 0 to --freq_w; "
                         "0 applies --freq_w from the first epoch. Inert when --freq_w 0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_json", default=None, help="override the output path")
    args = ap.parse_args()

    x_all, y_all = load_mat_pair(os.path.join(DATA, "level5_steady"))
    assert x_all.shape[0] == N_CASES * PER_CASE, \
        f"level5 should hold 5000 samples, found {x_all.shape[0]}"

    out = few_shot_run(x_all, y_all, args.shots, args.ckpt, epochs=args.epochs,
                       lr=args.lr, weight_decay=args.weight_decay, batch=args.batch,
                       eval_every=args.eval_every, freq_w=args.freq_w,
                       freq_dim=args.freq_dim, freq_warmup=args.freq_warmup,
                       seed=args.seed, device=args.device,
                       log=lambda s: print(s, flush=True))

    path = args.out_json or result_path(args.shots, args.freq_w, args.freq_warmup, args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    print(f"[fewshot] final rmse={out['final']['rmse']:.3f} "
          f"late_std={out['stability']['rmse_late_std']:.3f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
