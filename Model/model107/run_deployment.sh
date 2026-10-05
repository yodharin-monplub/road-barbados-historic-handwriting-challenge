#!/usr/bin/env bash
set -euo pipefail
cd /workspace/road-htr
export PYTHONPATH=src
test ! -e Other/outputs/model107/deployment_train.log
.venv-lighton/bin/python Model/model107/train.py --all-data \
    > Other/outputs/model107/deployment_train.log 2>&1
.venv-lighton/bin/python Model/model107/infer.py \
    --config Other/outputs/model107/config_foldall.yaml \
    --adapter Other/outputs/model107/train/foldall/best_adapter \
    --input Data/raw/Test.csv \
    --output Other/outputs/model107/test_raw.csv \
    > Other/outputs/model107/deployment_infer.log 2>&1
