#!/usr/bin/env bash
# Model130: PP-OCRv6/Kraken member on every fold, then OOF + test inference.
# Epoch policy is FIXED at zero-based epoch 7 (Model36's documented policy),
# not chosen per fold, so no held-out fold selects its own checkpoint.
set -uo pipefail
cd /workspace/project
KPY=/workspace/kvenv/bin/python
mkdir -p logs Other/outputs/model130_kraken

# Two ketos runs writing one fold directory interleave their checkpoints and
# make the fixed epoch ambiguous. Refuse to start if one is already live.
if pgrep -af "bin/ketos" > /dev/null; then
  echo "ABORT: a ketos process is already running:"; pgrep -af "bin/ketos" | cut -c1-100
  exit 1
fi

for fold in "$@"; do
  out=Other/outputs/model22/fold${fold}
  # Train only if this fold has no finished run. Always start from an empty
  # directory: a killed run leaves "-v1" duplicates that make the fixed
  # epoch-7 checkpoint ambiguous.
  if ! ls "${out}/checkpoint_09"*.ckpt >/dev/null 2>&1; then
    echo "=== train fold ${fold} $(date -u)"
    rm -rf "${out}"
    $KPY Model/model22/train.py --fold "${fold}" > "logs/kraken_train_fold${fold}.log" 2>&1
  fi
  ckpt=$(ls -t "${out}/checkpoint_07"*.ckpt 2>/dev/null | head -1)
  if [ -z "${ckpt}" ]; then echo "fold ${fold}: no epoch-7 checkpoint"; continue; fi
  export=Other/outputs/model130_kraken/fold${fold}
  mkdir -p "${export}"
  # model36_export only accepts Model36's all-data runs, so convert the fixed
  # epoch-7 checkpoint to weights with ketos itself.
  /workspace/kvenv/bin/ketos convert -o "${export}/epoch07.safetensors" "${ckpt}" \
      > "logs/kraken_export_fold${fold}.log" 2>&1
  weights=$(ls "${export}"/epoch07.safetensors 2>/dev/null | head -1)
  if [ -z "${weights}" ]; then echo "fold ${fold}: export failed"; continue; fi
  echo "=== infer fold ${fold} with $(basename "${weights}") $(date -u)"
  $KPY Model/model22/infer.py --input Data/processed/train_folds.csv --fold "${fold}" \
      --checkpoint "${weights}" --output "${export}/oof_ep8_greedy.csv" \
      > "logs/kraken_infer_fold${fold}.log" 2>&1
  $KPY Model/model22/infer.py --input Data/processed/test_ids_derived.csv \
      --checkpoint "${weights}" --output "${export}/test_beam1.csv" \
      > "logs/kraken_test_fold${fold}.log" 2>&1
  echo "=== done fold ${fold} $(date -u)"
done
echo KRAKEN_FOLDS_DONE
