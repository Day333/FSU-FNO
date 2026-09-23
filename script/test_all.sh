#!/bin/bash
# Evaluate everything: level2/3/4 in-support, then level5 zero-shot from the
# level4 checkpoint (with per-case RMSE).
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
mkdir -p logs
for lv in level2 level3 level4 level5; do
  echo "== test $lv =="
  "$PY" test.py --level "$lv" 2>&1 | tee "logs/test_$lv.log"
done
