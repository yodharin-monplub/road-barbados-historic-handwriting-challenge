#!/usr/bin/env bash
# Run a queue of "config fold [extra args]" jobs sequentially; skips finished ones.
# usage: nohup bash Model/model122/vast_queue.sh queue.txt > logs/queue.log 2>&1 &
set -uo pipefail
cd /workspace/project
mkdir -p logs
while read -r config fold extra; do
  [ -z "${config:-}" ] && continue
  case "$config" in \#*) continue ;; esac
  name=$(basename "$config" .json)
  out=$(/workspace/venv/bin/python -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8-sig'))['output_dir'])" "$config")
  if [ -f "$out/fold$fold/DONE" ]; then echo "skip $name fold$fold"; continue; fi
  echo "=== start $name fold$fold $(date -u)"
  PYTHONUNBUFFERED=1 /workspace/venv/bin/python Model/model122/train.py --config "$config" --fold "$fold" $extra \
    > "logs/${name}_fold${fold}.log" 2>&1
  echo "=== end $name fold$fold rc=$? $(date -u)"
done < "$1"
echo QUEUE_DONE
