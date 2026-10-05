#!/usr/bin/env bash
# Set up Kraken 7.1 (Python 3.12) plus the organizer-approved PP-OCRv6 medium
# historical checkpoint, on a rented Ampere GPU host.
set -euo pipefail
cd /workspace
pip install -q uv 2>/dev/null || true
[ -x /workspace/kvenv/bin/python ] || uv venv /workspace/kvenv --python 3.12 -q
uv pip install -q --python /workspace/kvenv/bin/python "kraken==7.1.*" || \
  uv pip install -q --python /workspace/kvenv/bin/python kraken
/workspace/kvenv/bin/python -c "import kraken, torch; print('kraken', kraken.__version__, 'torch', torch.__version__, torch.cuda.is_available())"
mkdir -p /workspace/project/checkpoints/model8
cd /workspace/project
if [ ! -f Other/checkpoints/model8/medium.safetensors ]; then
  XDG_DATA_HOME="$PWD/checkpoints/model8/registry" /workspace/kvenv/bin/kraken get 10.5281/zenodo.21788410
  found=$(find Other/checkpoints/model8/registry -name 'medium.safetensors' | head -1)
  [ -n "$found" ] && cp "$found" Other/checkpoints/model8/medium.safetensors
fi
sha256sum Other/checkpoints/model8/medium.safetensors
echo "expected 15313b51ace64cbfa81f8f6ef25ad64f04e5a6fb7f7823e67b107527bc081ac9"
echo KRAKEN_SETUP_OK
