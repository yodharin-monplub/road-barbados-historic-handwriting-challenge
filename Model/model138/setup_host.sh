#!/usr/bin/env bash
# One-shot host setup for the Model138 GRPO trial on an Ampere GPU (CUDA 12.8).
set -euo pipefail
cd /workspace
apt-get update -qq && apt-get install -y -qq build-essential > /tmp/apt.log 2>&1
pip install -q uv
[ -x /workspace/venv/bin/python ] || uv venv /workspace/venv --python 3.11 -q
uv pip install -q --python /workspace/venv/bin/python "torch==2.9.*" torchvision \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install -q --python /workspace/venv/bin/python transformers==5.16.1 accelerate==1.14.0 \
  peft==0.20.0 safetensors==0.8.0 sentencepiece Pillow==12.3.0 PyYAML==6.0.3 numpy huggingface_hub

mkdir -p /workspace/project/Data/raw
cd /workspace/project
tar -xzf /workspace/grpo_code.tgz
mkdir -p outputs
tar -xzf /workspace/pseudo3.tgz -C Other/outputs
[ -d Data/raw/images ] || { tar -xf /workspace/images_small.tar -C Data/raw && mv Data/raw/images_small Data/raw/images; }
/workspace/venv/bin/hf download lightonai/LightOnOCR-2-1B-base \
  --revision c1bc8eb6625be7b34178ff53481b242f7b8ffcb8 \
  --local-dir Other/checkpoints/model73/lightonocr-2-1b-base > /dev/null
/workspace/venv/bin/python - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
EOF
echo "images: $(ls Data/raw/images | wc -l)  pseudo3: $(ls Other/outputs/pseudo3/*.csv | wc -l)"
echo SETUP_OK

