#!/usr/bin/env bash
set -euo pipefail

project_dir="${1:-$PWD}"
cd "$project_dir"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash Model/model93/setup_runpod.sh "$project_dir"

resume_args=()
if [[ -f Other/outputs/model93/fold0/generations.jsonl ]]; then
  resume_args+=(--resume)
fi

PYTHONPATH=src .venv-Model/model93/bin/python Model/model93/infer.py \
  --config Model/model93/config.json \
  --prompts Other/outputs/model93/prompts/fold0.jsonl \
  --model-audit Other/outputs/model93/qwen_snapshot_audit.json \
  --output Other/outputs/model93/fold0/aggressive.csv \
  --details Other/outputs/model93/fold0/generations.jsonl \
  --batch-size 32 \
  "${resume_args[@]}"

sha256sum \
  Other/outputs/model93/fold0/aggressive.csv \
  Other/outputs/model93/fold0/generations.jsonl \
  Other/outputs/model93/fold0/generations.jsonl.manifest.json
