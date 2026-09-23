"""Ablation of the few-shot loss schedule on level5.

  python fewshot_sweep.py [--shots 5 10 50] [--seeds 0 1 2] [--arms mse fixed warm]

Three arms, all starting from the same level4 checkpoint and the same K samples:

  mse    plain MSE                                    (the released setting)
  fixed  MSE + 0.1 * freq_L1 from epoch 1             (pre-training weight, held)
  warm   MSE + 0.1 * min(1, e/10) * freq_L1           (linear warm-up, proposed)

The question is whether letting the MSE remove the global temperature offset
first and only then ramping in the frequency term buys accuracy on a K-sample
budget -- and whether it calms the late-epoch oscillation the plain-MSE runs
show. RMSE, MaxAE and Top-50 MAE are reported as mean +/- std over seeds, with
the late-phase RMSE spread alongside.

Seed caveat: the protocol fixes both the initialization (the level4 checkpoint)
and the adaptation samples (the first K of each case's pool), so a seed only
changes the minibatch order and the nondeterministic parts of the backward
pass. The spread reported here is therefore the floor of the run-to-run noise,
not the spread over which samples you happen to get.

The data is loaded once for the whole sweep. Runs whose result file already
exists are skipped unless --overwrite is given, so an interrupted sweep can be
restarted with the same command.

Writes results/curriculum/fewshot_k{K}_{tag}_s{seed}.json per run plus
results/curriculum/summary.json and summary.md.
"""
import argparse
import json
import os
import time

import numpy as np

from data import load_mat_pair
from finetune import DATA, HERE, N_CASES, PER_CASE, few_shot_run, schedule_tag

# arm -> (freq_w, freq_warmup)
ARMS = {
    "mse": (0.0, 0),
    "fixed": (0.1, 0),
    "warm": (0.1, 10),
}
REPORT = [("rmse", "RMSE"), ("max_absolute_error", "MaxAE"), ("top_mae", "Top-50 MAE"),
          ("max_temperature_error", "PeakErr")]


def _agg(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0)),
            "n": len(values)}


def summarize(runs):
    """runs: {(shots, arm): [result dict per seed]} -> per-cell aggregate."""
    out = {}
    for (shots, arm), rs in sorted(runs.items()):
        cell = {"shots": shots, "arm": arm, "tag": rs[0]["loss_schedule"],
                "seeds": [r["seed"] for r in rs]}
        for key, _ in REPORT:
            cell[key] = _agg([r["final"][key] for r in rs])
        cell["rmse_best"] = _agg([r["stability"]["rmse_best"] for r in rs])
        cell["rmse_late_std"] = _agg([r["stability"]["rmse_late_std"] for r in rs])
        out[f"k{shots}_{arm}"] = cell
    return out


def paired_deltas(runs, shots_list, pairs):
    """Per-seed RMSE differences between two arms.

    Seeds share the initialization and the K samples, so arm A and arm B at the
    same seed differ only in the loss schedule. Pairing on the seed removes the
    batch-order noise that the unpaired std is dominated by -- which is the only
    way to separate two arms whose means sit inside each other's spread.
    """
    out = {}
    for shots in shots_list:
        for a, b in pairs:
            ra = {r["seed"]: r for r in runs.get((shots, a), [])}
            rb = {r["seed"]: r for r in runs.get((shots, b), [])}
            seeds = sorted(set(ra) & set(rb))
            if not seeds:
                continue
            d = [ra[s]["final"]["rmse"] - rb[s]["final"]["rmse"] for s in seeds]
            out[f"k{shots}_{a}_vs_{b}"] = {
                "shots": shots, "arm": a, "baseline": b, "seeds": seeds,
                "per_seed": d, "mean": float(np.mean(d)), "std": float(np.std(d, ddof=0)),
                "wins": sum(x < 0 for x in d),
                "rel_pct": 100.0 * float(np.mean(d)) / float(np.mean(
                    [rb[s]["final"]["rmse"] for s in seeds])),
            }
    return out


def markdown_paired(deltas):
    lines = ["| K | comparison | ΔRMSE (paired) | rel. | wins |", "|---|---|---|---|---|"]
    for v in deltas.values():
        lines.append(f"| {v['shots']} | {v['arm']} − {v['baseline']} | "
                     f"{v['mean']:+.3f} ± {v['std']:.3f} | {v['rel_pct']:+.1f}% | "
                     f"{v['wins']}/{len(v['seeds'])} |")
    return "\n".join(lines)


def markdown_table(summary, shots_list, arms):
    lines = ["| K | schedule | " + " | ".join(n for _, n in REPORT)
             + " | late RMSE std | vs MSE |",
             "|---|---|" + "---|" * (len(REPORT) + 2)]
    for shots in shots_list:
        base = summary.get(f"k{shots}_mse")
        for arm in arms:
            cell = summary.get(f"k{shots}_{arm}")
            if cell is None:
                continue
            cols = [f"{cell[k]['mean']:.3f} ± {cell[k]['std']:.3f}" for k, _ in REPORT]
            delta = "--"
            if base is not None and arm != "mse" and base["rmse"]["mean"] > 0:
                rel = 100.0 * (cell["rmse"]["mean"] - base["rmse"]["mean"]) / base["rmse"]["mean"]
                delta = f"{rel:+.1f}%"
            lines.append(f"| {shots} | {arm} ({cell['tag']}) | " + " | ".join(cols)
                         + f" | {cell['rmse_late_std']['mean']:.3f} | {delta} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, nargs="+", default=[5, 10, 50])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "level4", "model.pt"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--freq_dim", default="xy", choices=["x", "xy"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default=os.path.join(HERE, "results", "curriculum"))
    ap.add_argument("--overwrite", action="store_true",
                    help="rerun cells whose result file already exists")
    ap.add_argument("--dry_run", action="store_true", help="list the runs and exit")
    args = ap.parse_args()

    plan = [(k, arm, s) for k in args.shots for arm in args.arms for s in args.seeds]
    os.makedirs(args.out_dir, exist_ok=True)

    def path_of(shots, arm, seed):
        w, warm = ARMS[arm]
        return os.path.join(args.out_dir,
                            f"fewshot_k{shots}_{schedule_tag(w, warm)}_s{seed}.json")

    if args.dry_run:
        for k, arm, s in plan:
            state = "exists" if os.path.exists(path_of(k, arm, s)) else "todo"
            print(f"K={k:<3} arm={arm:<5} seed={s}  {state}  {path_of(k, arm, s)}")
        print(f"{len(plan)} runs, {args.epochs} epochs each")
        return

    x_all, y_all = load_mat_pair(os.path.join(DATA, "level5_steady"))
    assert x_all.shape[0] == N_CASES * PER_CASE, \
        f"level5 should hold 5000 samples, found {x_all.shape[0]}"

    runs, t0 = {}, time.time()
    for i, (shots, arm, seed) in enumerate(plan, 1):
        freq_w, freq_warmup = ARMS[arm]
        path = path_of(shots, arm, seed)
        if os.path.exists(path) and not args.overwrite:
            res = json.load(open(path))
            print(f"[{i}/{len(plan)}] cached {os.path.basename(path)} "
                  f"rmse={res['final']['rmse']:.3f}", flush=True)
        else:
            print(f"[{i}/{len(plan)}] K={shots} arm={arm} seed={seed} "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)
            res = few_shot_run(x_all, y_all, shots, args.ckpt, epochs=args.epochs,
                               lr=args.lr, batch=args.batch, eval_every=args.eval_every,
                               freq_w=freq_w, freq_dim=args.freq_dim,
                               freq_warmup=freq_warmup, seed=seed, device=args.device,
                               log=lambda s: print(s, flush=True))
            res["arm"] = arm
            json.dump(res, open(path, "w"), indent=2)
        runs.setdefault((shots, arm), []).append(res)

    summary = summarize(runs)
    pairs = [(a, b) for a, b in (("fixed", "mse"), ("warm", "mse"), ("warm", "fixed"))
             if a in args.arms and b in args.arms]
    deltas = paired_deltas(runs, args.shots, pairs)
    meta = {"epochs": args.epochs, "lr": args.lr, "batch": args.batch,
            "ckpt": args.ckpt, "shots": args.shots, "seeds": args.seeds,
            "arms": {a: {"freq_w": ARMS[a][0], "freq_warmup": ARMS[a][1]} for a in args.arms},
            "sweep_time_s": round(time.time() - t0, 1)}
    json.dump({"meta": meta, "cells": summary, "paired": deltas},
              open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)

    table = markdown_table(summary, args.shots, args.arms)
    body = (f"# level5 few-shot loss schedule\n\n"
            f"{len(args.seeds)} seeds, {args.epochs} epochs, lr={args.lr}, "
            f"holdout 2500 samples. Mean +/- std over seeds; all values in kelvin.\n\n"
            f"{table}\n")
    if deltas:
        body += ("\n## Paired per-seed RMSE differences\n\n"
                 "Same seed = same initialization, same K samples, same batch order;\n"
                 "only the loss schedule differs. Negative favours the first arm.\n\n"
                 + markdown_paired(deltas) + "\n")
    open(os.path.join(args.out_dir, "summary.md"), "w").write(body)
    print("\n" + body, flush=True)
    print(f"[sweep] {len(plan)} runs in {time.time() - t0:.0f}s -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
