#!/bin/bash
# Train FSU-FNO (freq_w = 0.1) on level2/3/4. The level4 inputs (P=7) mix channels
# whose magnitudes differ by orders of magnitude, so per-channel normalization is
# mandatory there.
# Usage: [PY=/path/to/python] [CUDA_VISIBLE_DEVICES=N] bash script/train_all.sh
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
mkdir -p logs
for lv in level2 level3 level4; do
  extra=""
  [ "$lv" = "level4" ] && extra="--per_channel_norm"
  echo "== train $lv =="
  "$PY" train.py --data "${FSU_DATA:-$PWD/data}/${lv}_steady" \
      --out "checkpoints/$lv" --freq_w 0.1 --seed 0 $extra 2>&1 | tee "logs/train_$lv.log"
done
