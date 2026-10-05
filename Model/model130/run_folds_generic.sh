#!/usr/bin/env bash
# Generic Kraken fold runner: run_folds_generic.sh <model_dir> <train_out_base> <export_base> <fold...>
#   e.g. run_folds_generic.sh model132 Other/outputs/model132 Other/outputs/model132_kraken128 0 1 2 3 4
# Epoch policy is FIXED at zero-based epoch 7 for every fold (Model36's policy).
set -uo pipefail
cd /workspace/project
KPY=/workspace/kvenv/bin/python
MODEL_DIR=$1; TRAIN_BASE=$2; EXPORT_BASE=$3; shift 3
mkdir -p logs "${EXPORT_BASE}"

if pgrep -af "bin/ketos" > /dev/null; then
  echo "ABORT: a ketos process is already running"; pgrep -af "bin/ketos" | cut -c1-100; exit 1
fi

for fold in "$@"; do
  out=${TRAIN_BASE}/fold${fold}
  if ! ls "${out}/checkpoint_09"*.ckpt >/dev/null 2>&1; then
    echo "=== train ${MODEL_DIR} fold ${fold} $(date -u)"
    rm -rf "${out}"
    $KPY "${MODEL_DIR}/train.py" --fold "${fold}" > "logs/${MODEL_DIR}_train_fold${fold}.log" 2>&1
  fi
  ckpt=$(ls -t "${out}/checkpoint_07"*.ckpt 2>/dev/null | head -1)
  if [ -z "${ckpt}" ]; then echo "fold ${fold}: no epoch-7 checkpoint"; continue; fi
  export=${EXPORT_BASE}/fold${fold}
  mkdir -p "${export}"
  /workspace/kvenv/bin/ketos convert -o "${export}/epoch07.safetensors" "${ckpt}" \
      > "logs/${MODEL_DIR}_export_fold${fold}.log" 2>&1
  weights=${export}/epoch07.safetensors
  [ -f "${weights}" ] || { echo "fold ${fold}: convert failed"; continue; }
  echo "=== infer ${MODEL_DIR} fold ${fold} $(date -u)"
  $KPY "${MODEL_DIR}/infer.py" --input Data/processed/train_folds.csv --fold "${fold}" \
      --checkpoint "${weights}" --output "${export}/oof_ep8_greedy.csv" \
      > "logs/${MODEL_DIR}_infer_fold${fold}.log" 2>&1
  $KPY "${MODEL_DIR}/infer.py" --input Data/processed/test_ids_derived.csv \
      --checkpoint "${weights}" --output "${export}/test_beam1.csv" \
      > "logs/${MODEL_DIR}_test_fold${fold}.log" 2>&1
  echo "=== done fold ${fold} $(date -u)"
done
echo KRAKEN_FOLDS_DONE
