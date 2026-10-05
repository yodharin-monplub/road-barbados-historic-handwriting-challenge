#!/usr/bin/env bash
set -euo pipefail
cd /workspace/road-htr
mkdir -p Other/outputs/model107/train/fold0
test ! -e Other/outputs/model107/gpu_train.log
export PYTHONPATH=src
.venv-lighton/bin/python Model/model107/train.py > Other/outputs/model107/gpu_train.log 2>&1
.venv-lighton/bin/python Model/model107/infer.py \
  --adapter Other/outputs/model107/train/fold0/best_adapter \
  --input Data/processed/train_folds.csv --fold 0 \
  --output Other/outputs/model107/fold0_oof.csv > Other/outputs/model107/gpu_infer.log 2>&1
