#!/usr/bin/env bash
set -euo pipefail

project_dir="${1:-$PWD}"
cd "$project_dir"

.venv-Model/model95/bin/python Model/model95/infer.py \
  --config Model/model95/config.json \
  --records Other/outputs/model95/fold0/images.jsonl \
  --audit Other/outputs/model95/qwen3_vl_snapshot_audit.json \
  --output-dir Other/outputs/model95/fold0/direct

.venv-Model/model95/bin/python Model/model95/evaluate.py \
  --direct Other/outputs/model95/fold0/direct/predictions.csv \
  --details Other/outputs/model95/fold0/direct/details.jsonl \
  --fold 0 \
  --output-dir Other/outputs/model95/fold0/evaluation
