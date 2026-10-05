#!/usr/bin/env bash
# One-time setup of a rented vast.ai GPU host for Model122.
set -euo pipefail
cd /workspace
pip install -q uv
[ -x /workspace/venv/bin/python ] || uv venv /workspace/venv --python 3.11 -q
uv pip install -q --python /workspace/venv/bin/python torch==2.14.0 torchvision \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install -q --python /workspace/venv/bin/python transformers==5.16.1 accelerate==1.14.0 \
  peft==0.20.0 safetensors==0.8.0 sentencepiece Pillow==12.3.0 PyYAML==6.0.3 numpy huggingface_hub
/workspace/venv/bin/python - <<'EOF'
import torch, transformers
print(torch.__version__, transformers.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())
from huggingface_hub import snapshot_download
print(snapshot_download("lightonai/LightOnOCR-2-1B-base",
                        revision="c1bc8eb6625be7b34178ff53481b242f7b8ffcb8",
                        local_dir="/workspace/project/checkpoints/model73/lightonocr-2-1b-base"))
EOF
echo SETUP_OK
