#!/usr/bin/env bash
set -euo pipefail

project_dir="${1:-$PWD}"
cd "$project_dir"

python3 -m venv .venv-model93
.venv-Model/model93/bin/python -m pip install --upgrade pip
.venv-Model/model93/bin/python -m pip install -r Model/model93/requirements.txt

cache_dir="${HF_HOME:-/workspace/.cache/huggingface}"
mkdir -p "$cache_dir"
.venv-Model/model93/bin/python Model/model93/audit_model.py \
  --config Model/model93/config.json \
  --cache-dir "$cache_dir" \
  --output Other/outputs/model93/qwen_snapshot_audit.json
