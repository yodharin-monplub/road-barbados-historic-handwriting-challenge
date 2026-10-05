#!/usr/bin/env bash
# Model144 fold 0 on a host already set up by Model141/Model143 (venv, images, pseudo4).
set -euo pipefail
cd /workspace/project
sed -i 's/\r$//' Model/model122/train.py Model/model144/config_vision.json
/workspace/venv/bin/python - <<'EOF'
from huggingface_hub import HfApi, snapshot_download
sha = HfApi().model_info("Qwen/Qwen2.5-VL-7B-Instruct").sha
print("Qwen2.5-VL-7B revision:", sha)
snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct", revision=sha,
                  local_dir="Other/checkpoints/model136/qwen2.5-vl-7b-instruct")
EOF
ls Other/outputs/pseudo4/fold0.csv
printf 'Model/model144/config_vision.json 0 --predict-test\n' > /tmp/queue144.txt
bash Model/model122/vast_queue.sh /tmp/queue144.txt
