#!/usr/bin/env bash
# usage: bash go154.sh <fold> [<fold> ...]   - TrOCR member, folds run serially on ONE host
# Expects /workspace/{code.tgz,images.tar,setup_host.sh} (setup_host.sh = Model/model146's).
set -uo pipefail
cd /workspace
sed -i 's/\r$//' setup_host.sh
# A truncated images.tar leaves a partial image folder that setup then trusts (Model153).
want=$(stat -c %s /workspace/images.tar)
echo "images.tar bytes: $want"
CKPT=/workspace/project/Other/checkpoints/model154/trocr-large-handwritten
bash setup_host.sh "microsoft/trocr-large-handwritten" "$CKPT" || { echo SETUP_FAILED; exit 1; }
cd /workspace/project
n=$(ls Data/raw/images | wc -l)
[ "$n" -eq 5472 ] || { echo "IMAGES_INCOMPLETE: $n of 5472"; exit 1; }
# road_htr.trocr_cli resolves config paths against Other/ with the old names.
ln -sfn ../Data Other/data
sed -i 's/\r$//' Model/model154/train.py Model/model154/infer.py Model/model154/config.yaml Model/model154/make_tokenizer.py
/workspace/venv/bin/python Model/model154/make_tokenizer.py "$CKPT" || { echo TOKENIZER_FAILED; exit 1; }
PY=/workspace/venv/bin/python
P=/workspace/project
mkdir -p logs
for F in "$@"; do
  out=Other/outputs/model154_trocr/fold$F
  if [ -f "$out/DONE" ]; then echo "skip fold $F"; continue; fi
  echo "=== train fold $F $(date -u)"
  PYTHONUNBUFFERED=1 $PY Model/model154/train.py --fold "$F" > "logs/m154_train_fold$F.log" 2>&1 \
    || { echo "TRAIN_FAILED fold $F"; tail -n 20 "logs/m154_train_fold$F.log"; exit 1; }
  model=$P/Other/outputs/model154/fold$F/best_model
  mkdir -p "$out"
  echo "=== infer fold $F $(date -u)"
  PYTHONUNBUFFERED=1 $PY Model/model154/infer.py --model "$model" \
    --input "$P/Data/processed/train_folds.csv" --fold "$F" \
    --output "$P/$out/oof_ep10_greedy.csv" > "logs/m154_oof_fold$F.log" 2>&1 \
    || { echo "OOF_FAILED fold $F"; tail -n 20 "logs/m154_oof_fold$F.log"; exit 1; }
  PYTHONUNBUFFERED=1 $PY Model/model154/infer.py --model "$model" \
    --input "$P/Data/processed/test_ids_derived.csv" \
    --output "$P/$out/test_beam1.csv" > "logs/m154_test_fold$F.log" 2>&1 \
    || { echo "TEST_FAILED fold $F"; tail -n 20 "logs/m154_test_fold$F.log"; exit 1; }
  wc -l "$out"/*.csv
  rm -rf "Other/outputs/model154/fold$F"/checkpoint-*   # keep disk small; best_model stays
  touch "$out/DONE"
  echo "=== done fold $F $(date -u)"
done
echo M154_ALL_DONE
