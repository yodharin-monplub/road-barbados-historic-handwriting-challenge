#!/usr/bin/env bash
# Host setup for Model144 (Qwen2.5-VL-7B, vision attention adapted), Ampere, CUDA 12.8.
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
tar -xzf /workspace/pseudo4.tgz
# Original caps only (height <= 176, longest edge <= 2800, q92) - see Model140's preprocessing note.
if [ ! -d Data/raw/images ]; then
  tar -xf /workspace/images.tar -C Data/raw && mv Data/raw/images Data/raw/images_full
  /workspace/venv/bin/python Other/scripts/shrink_images.py \
    --source Data/raw/images_full --output Data/raw/images --workers 8
fi
/workspace/venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
# Same revision Model144 fold 0 used.
snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct", revision="cc594898137f460bfe9f0759e9844b3ce807cfb5",
                  local_dir="Other/checkpoints/model136/qwen2.5-vl-7b-instruct")
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
EOF
echo "images: $(ls Data/raw/images | wc -l)  pseudo4: $(ls Other/outputs/pseudo4/*.csv | wc -l)"
echo SETUP_OK
