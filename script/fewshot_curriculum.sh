#!/bin/bash
# level5 few-shot loss-schedule ablation: plain MSE vs constant lambda=0.1 vs
# linear warm-up over 10 epochs, for K = 5 / 10 / 50 and 3 seeds (27 runs).
# Resumable: cells whose result json exists are skipped, so rerun the same
# command after an interruption.
# Usage: [PY=/path/to/python] [CUDA_VISIBLE_DEVICES=N] bash script/fewshot_curriculum.sh [extra args]
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
mkdir -p logs
"$PY" fewshot_sweep.py "$@" 2>&1 | tee -a logs/fewshot_curriculum.log
