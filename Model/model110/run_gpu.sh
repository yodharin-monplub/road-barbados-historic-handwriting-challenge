#!/usr/bin/env bash
set -euo pipefail
cd /workspace/road-htr
export PYTHONPATH=src
test ! -e Other/outputs/model110/fold0_train.log
.venv-lighton/bin/python Model/model110/train.py > Other/outputs/model110/fold0_train.log 2>&1
.venv-lighton/bin/python Model/model110/infer.py \
    --config Other/outputs/model110/config_fold0.yaml \
    --adapter Other/outputs/model110/train/fold0/best_adapter \
    --input Data/processed/train_folds.csv --fold 0 \
    --output Other/outputs/model110/fold0_oof.csv > Other/outputs/model110/fold0_infer.log 2>&1
