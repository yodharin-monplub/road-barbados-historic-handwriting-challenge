#!/usr/bin/env bash
set -euo pipefail

cd /workspace/road
current_pid="${1:?current fold-2 trainer PID required}"

validate_trainer() {
  local fold="$1" pid="$2" command
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  command="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
  [[ "$command" == *"Model/model73/train.py --fold ${fold}"* ]]
}

checkpoint_ready() {
  local fold="$1" dir="Other/outputs/model73/fold${1}/checkpoint-400"
  [[ -s "${dir}/adapter_model.safetensors" ]]
  [[ -s "${dir}/adapter_config.json" ]]
  [[ -s "${dir}/processor_config.json" ]]
  [[ -s "${dir}/tokenizer.json" ]]
  [[ -s "${dir}/trainer_state.json" ]]
  jq -e '.global_step == 400' "${dir}/trainer_state.json" >/dev/null
}

for fold in 2 3 4; do
  while ! checkpoint_ready "$fold"; do
    if ! validate_trainer "$fold" "$current_pid"; then
      echo "fold ${fold} trainer exited before checkpoint 400" >&2
      exit 1
    fi
    sleep 10
  done

  validate_trainer "$fold" "$current_pid"
  kill -TERM "$current_pid"
  for _ in $(seq 1 30); do
    kill -0 "$current_pid" 2>/dev/null || break
    sleep 2
  done
  if kill -0 "$current_pid" 2>/dev/null; then
    echo "fold ${fold} trainer did not stop after SIGTERM" >&2
    exit 1
  fi
  cp "Other/outputs/model73/fold${fold}_train.log" \
    "Other/outputs/model73/fold${fold}_train_stopped_after_step400.log"

  prediction="Other/outputs/model73/fold${fold}_oof_step400.csv"
  [[ ! -e "$prediction" ]]
  env PYTHONUNBUFFERED=1 PYTHONPATH=src .venv-lighton/bin/python Model/model73/infer.py \
    --adapter "Other/outputs/model73/fold${fold}/checkpoint-400" \
    --input Data/processed/train_folds.csv --fold "$fold" --output "$prediction" \
    > "Other/outputs/model73/fold${fold}_oof_step400.log" 2>&1
  env PYTHONPATH=src .venv-lighton/bin/python Model/model74/evaluate.py \
    --truth Data/processed/train_folds.csv --predictions "$prediction" \
    --manifest "Other/outputs/model73/fold${fold}_oof_step400.manifest.json" \
    --receipt "Other/outputs/model74/fold${fold}_step400.evaluation.json" --fold "$fold"

  if [[ "$fold" != 4 ]]; then
    next_fold="$((fold + 1))"
    [[ ! -e "Other/outputs/model73/fold${next_fold}" ]]
    env PYTHONUNBUFFERED=1 PYTHONPATH=src .venv-lighton/bin/python \
      Model/model73/train.py --fold "$next_fold" \
      > "Other/outputs/model73/fold${next_fold}_train.log" 2>&1 &
    current_pid="$!"
  fi
done

touch Other/outputs/model75_folds_2_4_complete
