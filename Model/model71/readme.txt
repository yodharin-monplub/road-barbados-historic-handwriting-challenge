MODEL 71 — NATIVE-HEIGHT PP-OCRV6 A6000 CHALLENGER
==================================================

Purpose
-------
Model71 tests the clearest GPU-capacity limitation in the promoted Model35
pipeline. Model35's recognizer component (Model36) trained at 96-pixel line
height because the local RTX 4050 had only 6 GiB VRAM. The approved historical
PP-OCRv6 medium checkpoint expects 128-pixel-high lines. Model71 restores that
native height on a 48 GiB RTX A6000 while retaining the compact competition
codec, exact labels, full aspect-preserving line images, and fold-safe scoring.

This is an OCR recognizer experiment, not an LLM. The earlier Qwen2-VL
Model50 path scored only 0.394881794 raw on fold 0 and is not being repeated.

Controlled protocol
-------------------
The first run uses fold 0 only. It trains ten complete epochs and evaluates
every checkpoint with the exact challenge metric. Relative to Model36 it
changes input height 96 -> 128, normalized width cap 3584 -> 6144, and uses a
physical batch of four with two-step accumulation, preserving effective batch
eight. It keeps the same approved base checkpoint, seed, compact codec,
AdamW learning rate 5e-5, warmup 100, ten-epoch cosine horizon, augmentation,
NFD normalization, and whitespace policy.

At height 128, two of the three valid rows that Model22/36 had to exclude at
height 96 return to training. EcxuqKeZl7OQexfB remains one CTC timestep too
short for its repeated-character target and is the sole Model71 training
exclusion. External validation still predicts and scores every canonical row;
the folds remain 819 rows each.

RunPod hardware
---------------
The attached Pod was checked before transfer:

  GPU: NVIDIA RTX A6000, 46,068 MiB VRAM
  CPU: 96 logical cores
  RAM: 503 GiB
  driver-reported CUDA capability: 13.0

Commands
--------
Create the environment on the Pod:

  cd /workspace/road
  python3 -m venv .venv-kraken
  .venv-kraken/bin/pip install --upgrade pip
  .venv-kraken/bin/pip install -r model71/requirements.txt -e .

Audit preparation without allocating the GPU:

  PYTHONPATH=src .venv-kraken/bin/python model71/train.py --fold 0 --dry-run

Train fold 0:

  PYTHONPATH=src .venv-kraken/bin/python model71/train.py --fold 0

Evaluate all ten checkpoints with the established resumable evaluator:

  PYTHONPATH=src .venv-kraken/bin/python model22/evaluate_checkpoints.py \
    --checkpoint-dir outputs/model71/fold0 \
    --input data/processed/train_folds.csv \
    --output-dir outputs/model71/checkpoint_evaluation/fold0 \
    --fold 0 --config model71/config.yaml --batch-size 4

Promotion gate
--------------
Model71 has no score until the held-fold predictions exist. The best raw epoch
may be selected only on fold 0, after which that epoch and correction terminal
stage must be frozen before folds 1--4. The primary gate is corrected fold-0
score above Model35's 0.846941606 on the same 819 rows, followed by improvement
on the combined confirmation folds and complete five-fold OOF. Model35's public
score is 0.840671603 and its five-fold OOF is 0.841721630; neither value may be
used to tune Model71 from test predictions.

Status
------
Configuration and wrappers were created on 2026-09-05. The data, approved
checkpoint, and code were checksum-verified on the RunPod. A CTC dry run and
13 focused Kraken CLI tests passed there. Fold-0 CUDA training was launched on
the RTX A6000 at 2026-09-05 00:45 UTC; PID 6994 spawned the expected Kraken
trainer, bf16 initialized successfully, and live sampling reached 97% GPU
utilization at about 11.8 GiB VRAM. No Model71 validation score, test
inference, or submission exists yet.

continue_after_fold0.sh is the guarded RunPod continuation. It waits for the
exact training PID, refuses to evaluate unless all ten epoch checkpoints and
Kraken's final best-weights marker exist, then evaluates every checkpoint.
It deliberately does not train folds 1--4 or terminate the Pod: those actions
depend on the fold-0 promotion evidence and verified artifact recovery.
