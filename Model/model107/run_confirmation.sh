#!/usr/bin/env bash
set -euo pipefail
cd /workspace/road-htr
export PYTHONPATH=src
for fold in 1 2 3 4; do
    if test -f "Other/outputs/model107/fold${fold}_oof.manifest.json"; then
        continue
    fi
    # Never overwrite an interrupted training experiment silently.
    test ! -e "Other/outputs/model107/fold${fold}_train.log"
    .venv-lighton/bin/python Model/model107/train.py --fold "$fold" \
        > "Other/outputs/model107/fold${fold}_train.log" 2>&1
    .venv-lighton/bin/python Model/model107/infer.py \
        --config "Other/outputs/model107/config_fold${fold}.yaml" \
        --adapter "Other/outputs/model107/train/fold${fold}/best_adapter" \
        --input Data/processed/train_folds.csv --fold "$fold" \
        --output "Other/outputs/model107/fold${fold}_oof.csv" \
        > "Other/outputs/model107/fold${fold}_infer.log" 2>&1
done
