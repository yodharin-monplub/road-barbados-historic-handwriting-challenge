#!/usr/bin/env bash
# Generic host setup (Model145/146): bash setup_host.sh <hf_repo> <local_checkpoint_dir>
set -euo pipefail
REPO=$1; DEST=$2
cd /workspace
apt-get update -qq && apt-get install -y -qq build-essential > /tmp/apt.log 2>&1
pip install -q uv
[ -x /workspace/venv/bin/python ] || uv venv /workspace/venv --python 3.11 -q
uv pip install -q --python /workspace/venv/bin/python "torch==2.9.*" torchvision \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install -q --python /workspace/venv/bin/python transformers==5.16.1 accelerate==1.14.0 \
  peft==0.20.0 safetensors==0.8.0 sentencepiece Pillow==12.3.0 PyYAML==6.0.3 numpy huggingface_hub

mkdir -p /workspace/project
cd /workspace/project
tar -xzf /workspace/code.tgz
mkdir -p Other/outputs Data/raw
for p in /workspace/pseudo*.tgz; do [ -f "$p" ] && tar -xzf "$p"; done
# Original caps only (height <= 176, longest edge <= 2800, q92) - see Model140's preprocessing note.
if [ ! -d Data/raw/images ]; then
  tar -xf /workspace/images.tar -C Data/raw && mv Data/raw/images Data/raw/images_full
  /workspace/venv/bin/python Other/scripts/shrink_images.py \
    --source Data/raw/images_full --output Data/raw/images --workers 8
fi
/workspace/venv/bin/python - "$REPO" "$DEST" <<'EOF'
import sys
from huggingface_hub import HfApi, snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
info = HfApi().model_info(repo)
print(repo, "revision:", info.sha, "license:", (info.card_data or {}).get("license"))
snapshot_download(repo, revision=info.sha, local_dir=dest)
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
EOF
echo "images: $(ls Data/raw/images | wc -l)  pseudo dirs: $(ls -d Other/outputs/pseudo* 2>/dev/null | tr '\n' ' ')"
echo SETUP_OK
