#!/usr/bin/env bash
# Verify the Kraken environment and the pinned PP-OCRv6 artifact.
set -uo pipefail
cd /workspace/project
mkdir -p Other/checkpoints/model8
src=$(find /root/.local/share/htrmopo -name 'medium.safetensors' | head -1)
[ -n "$src" ] && cp -n "$src" Other/checkpoints/model8/medium.safetensors
sha256sum Other/checkpoints/model8/medium.safetensors
echo "expected  15313b51ace64cbfa81f8f6ef25ad64f04e5a6fb7f7823e67b107527bc081ac9"
/workspace/kvenv/bin/python - <<'EOF'
import importlib.metadata as meta
import torch
print("kraken", meta.version("kraken"))
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
EOF
