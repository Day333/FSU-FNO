"""FSU-FNO evaluation entry point.

  python test.py --level level2|level3|level4    in-support: that level's test
                                                 split (the trailing 20%, 3000 samples)
  python test.py --level level5                  structural zero-shot: the level4
                                                 checkpoint applied to all 5000
                                                 held-out samples, with per-case RMSE

Checkpoints are the tensor-only files written by train.py (see checkpoint.py).
Override the dataset root with the FSU_DATA environment variable.
"""
import argparse
import json
import os

import torch

from checkpoint import load_checkpoint
from data import load_mat_pair, split_train_val_test
from metrics import compute_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("FSU_DATA", os.path.join(HERE, "data"))
# level5 holds five structurally unseen packages, 1000 consecutive samples each.
N_CASES, FIRST_CASE = 5, 16


def _infer(ckpt_path, x, y, batch=20, device="cuda"):
    xn, model, yn = load_checkpoint(ckpt_path, device)
    preds = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            xb = xn(x[i:i + batch].to(device))
            preds.append(yn.inverse(model(xb)).cpu())
    out = torch.cat(preds, 0)
    return out, compute_metrics(out.permute(0, 3, 1, 2).numpy(),
                                y.permute(0, 3, 1, 2).numpy(), topk=50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True,
                    choices=["level2", "level3", "level4", "level5"])
    ap.add_argument("--ckpt", default=None,
                    help="defaults to checkpoints/<level>/model.pt; level5 uses level4's")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    zero_shot = args.level == "level5"
    ckpt = args.ckpt or os.path.join(
        HERE, "checkpoints", "level4" if zero_shot else args.level, "model.pt")

    x_all, y_all = load_mat_pair(os.path.join(DATA, f"{args.level}_steady"))
    if zero_shot:
        x_eval, y_eval = x_all, y_all                # evaluation-only set: use all 5000
    else:
        *_, x_eval, y_eval = split_train_val_test(x_all, y_all, 0.8)

    out, m = _infer(ckpt, x_eval, y_eval, args.batch, args.device)
    print(f"[FSU-FNO] {args.level} n={x_eval.shape[0]} ckpt={ckpt}")
    for k, v in m.items():
        print(f"   {k}: {v:.4f}")

    if zero_shot:
        n_case = x_eval.shape[0] // N_CASES
        for c in range(N_CASES):
            seg = slice(c * n_case, (c + 1) * n_case)
            rc = compute_metrics(out[seg].permute(0, 3, 1, 2).numpy(),
                                 y_eval[seg].permute(0, 3, 1, 2).numpy(), topk=50)
            print(f"   Case{FIRST_CASE + c}: RMSE={rc['rmse']:.2f}")
    else:
        ref = os.path.join(os.path.dirname(ckpt), "test_metrics.json")
        if os.path.exists(ref):
            r = json.load(open(ref))
            print(f"   (recorded rmse: {r.get('rmse'):.4f} -> "
                  f"{'MATCH' if abs(r.get('rmse', 0) - m['rmse']) < 1e-3 else 'DIFFERS'})")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(m, open(os.path.join(HERE, "results", f"{args.level}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
