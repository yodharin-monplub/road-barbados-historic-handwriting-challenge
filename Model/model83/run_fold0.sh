#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

mkdir -p Other/outputs/model83/fold0

for variant in enhanced center center_enhanced; do
  output="Other/outputs/model83/fold0/${variant}.csv"
  test ! -e "$output"
  test ! -e "${output%.csv}.manifest.json"
  .venv-lighton/bin/python Model/model73/infer.py \
    --adapter Other/outputs/model73/fold0/checkpoint-400 \
    --input Data/processed/train_folds.csv --fold 0 \
    --image-variant "$variant" --output "$output"
done

.venv-lighton/bin/python Model/model83/evaluate_fold.py \
  --truth Data/processed/train_folds.csv --fold 0 \
  --source full=Other/outputs/model73/fold0_oof_step400.csv \
  --source enhanced=Other/outputs/model83/fold0/enhanced.csv \
  --source center=Other/outputs/model83/fold0/center.csv \
  --source center_enhanced=Other/outputs/model83/fold0/center_enhanced.csv \
  --source kraken=Other/outputs/model35/fold0_epoch07_verified/predictions/03_confusions_fold0.csv \
  --output Other/outputs/model83/fold0/medoid.csv \
  --report Other/outputs/model83/fold0/report.json
