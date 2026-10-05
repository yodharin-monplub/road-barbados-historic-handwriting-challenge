#!/usr/bin/env bash
# Host setup for the Model141 GOT-OCR2 member (Ampere, CUDA 12.8).
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
tar -xzf /workspace/got_code.tgz
mkdir -p Other/outputs Data/raw
tar -xzf /workspace/pseudo4.tgz -C Other/outputs
[ -d Data/raw/images ] || { tar -xf /workspace/images_small.tar -C Data/raw && mv Data/raw/images_small Data/raw/images; }
/workspace/venv/bin/python - <<'EOF'
from huggingface_hub import HfApi, snapshot_download
sha = HfApi().model_info("stepfun-ai/GOT-OCR-2.0-hf").sha
print("GOT-OCR2 revision:", sha)
snapshot_download("stepfun-ai/GOT-OCR-2.0-hf", revision=sha,
                  local_dir="Other/checkpoints/model141/got-ocr2-hf")
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
EOF
echo "images: $(ls Data/raw/images | wc -l)  pseudo4: $(ls Other/outputs/pseudo4/*.csv | wc -l)"
echo SETUP_OK
