#!/usr/bin/env bash
# Host setup for the Model136 Qwen2.5-VL-7B fold-0 gate (Ampere, CUDA 12.8).
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
tar -xzf /workspace/q7b_code.tgz
mkdir -p Other/outputs && tar -xzf /workspace/pseudo3.tgz -C Other/outputs
[ -d Data/raw/images ] || { tar -xf /workspace/images_small.tar -C Data/raw && mv Data/raw/images_small Data/raw/images; }
/workspace/venv/bin/hf download Qwen/Qwen2.5-VL-7B-Instruct \
  --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --local-dir Other/checkpoints/model136/qwen2.5-vl-7b-instruct > /dev/null
/workspace/venv/bin/python - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("has Qwen2_5_VL:", hasattr(transformers, "Qwen2_5_VLForConditionalGeneration"))
EOF
du -sh Other/checkpoints/model136
echo "images: $(ls Data/raw/images | wc -l)"
echo SETUP_OK

