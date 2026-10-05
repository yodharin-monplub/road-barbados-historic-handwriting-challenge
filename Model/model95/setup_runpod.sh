#!/usr/bin/env bash
set -euo pipefail

project_dir="${1:-$PWD}"
cd "$project_dir"

python3 -m venv .venv-model95
.venv-Model/model95/bin/python -m pip install --upgrade pip
.venv-Model/model95/bin/python -m pip install -r Model/model95/requirements.txt

cache_dir="${HF_HOME:-/workspace/.cache/huggingface}"
mkdir -p "$cache_dir"
.venv-Model/model95/bin/python Model/model95/audit_model.py \
  --config Model/model95/config.json \
  --cache-dir "$cache_dir" \
  --output Other/outputs/model95/qwen3_vl_snapshot_audit.json
