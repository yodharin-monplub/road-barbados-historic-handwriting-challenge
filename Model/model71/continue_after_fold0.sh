#!/usr/bin/env bash
set -euo pipefail

cd /workspace/road

training_pid_file="Other/outputs/model71/fold0_train.pid"
continuation_log="Other/outputs/model71/fold0_continuation.log"

if [[ ! -s "${training_pid_file}" ]]; then
  echo "missing training PID file: ${training_pid_file}" >&2
  exit 2
fi

training_pid="$(tr -d '[:space:]' < "${training_pid_file}")"
if [[ ! "${training_pid}" =~ ^[0-9]+$ ]]; then
  echo "invalid training PID: ${training_pid}" >&2
  exit 2
fi

echo "waiting for Model71 fold-0 training PID ${training_pid}"
while kill -0 "${training_pid}" 2>/dev/null; do
  sleep 30
done

checkpoint_count="$(find Other/outputs/model71/fold0 -maxdepth 1 -type f \
  -name 'checkpoint_*.ckpt' | wc -l)"
best_count="$(find Other/outputs/model71/fold0 -maxdepth 1 -type f \
  -name 'best_*.safetensors' | wc -l)"

if [[ "${checkpoint_count}" -ne 10 || "${best_count}" -lt 1 ]]; then
  echo "training did not finish cleanly: checkpoints=${checkpoint_count}, best=${best_count}" >&2
  exit 3
fi

echo "training complete; evaluating all ${checkpoint_count} checkpoints"
PYTHONPATH=src .venv-kraken/bin/python Model/model22/evaluate_checkpoints.py \
  --checkpoint-dir Other/outputs/model71/fold0 \
  --input Data/processed/train_folds.csv \
  --output-dir Other/outputs/model71/checkpoint_evaluation/fold0 \
  --fold 0 \
  --config Model/model71/config.yaml \
  --batch-size 4

echo "Model71 fold-0 checkpoint evaluation complete"
touch Other/outputs/model71/fold0_evaluation_complete
