#!/bin/bash
# Paper S5 SCFT curve: K = 0 (zero-shot) / 10 / 50 / 100 / 250 / 500.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
mkdir -p logs
for k in 0 10 50 100 250 500; do
  echo "== finetune K=$k =="
  "$PY" finetune.py --shots "$k" --freq_w 0.1 --freq_warmup 10 \
      2>&1 | tee "logs/fewshot_k$k.log"
done
