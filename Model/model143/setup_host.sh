#!/usr/bin/env bash
# Host setup for the Model143 Qwen3-VL-4B vision-adapted member (Ampere, CUDA 12.8).
set -euo pipefail
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
# Original caps only (height <= 176, longest edge <= 2800, q92) - see Model140's preprocessing note.
# No local Python on the uploading machine, so the full images are shrunk here with the same script.
if [ ! -d Data/raw/images ]; then
  if [ -f /workspace/images_small.tar ]; then
    tar -xf /workspace/images_small.tar -C Data/raw && mv Data/raw/images_small Data/raw/images
  else
    tar -xf /workspace/images.tar -C Data/raw && mv Data/raw/images Data/raw/images_full
    /workspace/venv/bin/python Other/scripts/shrink_images.py \
      --source Data/raw/images_full --output Data/raw/images --workers 8
  fi
fi
/workspace/venv/bin/python - <<'EOF'
from huggingface_hub import HfApi, snapshot_download
sha = HfApi().model_info("Qwen/Qwen3-VL-4B-Instruct").sha
print("Qwen3-VL-4B revision:", sha)
snapshot_download("Qwen/Qwen3-VL-4B-Instruct", revision=sha,
                  local_dir="Other/checkpoints/model123/qwen3-vl-4b-instruct")
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
EOF
echo "images: $(ls Data/raw/images | wc -l)"
echo SETUP_OK
