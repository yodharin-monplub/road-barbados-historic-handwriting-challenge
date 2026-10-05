#!/usr/bin/env bash
# Report Kraken host state: live processes, GPU, per-fold artifacts.
cd /workspace/project
echo "live_ketos=$(pgrep -cf 'bin/ketos' 2>/dev/null; true)"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
for fold in 0 1 2 3 4; do
  ck=$(ls Other/outputs/model22/fold${fold}/checkpoint_*.ckpt 2>/dev/null | wc -l)
  oof=$([ -f Other/outputs/model130_kraken/fold${fold}/oof_ep8_greedy.csv ] && echo yes || echo no)
  test=$([ -f Other/outputs/model130_kraken/fold${fold}/test_beam1.csv ] && echo yes || echo no)
  echo "fold${fold}: checkpoints=${ck} oof=${oof} test=${test}"
done
tail -3 logs_kraken_folds3.log 2>/dev/null
