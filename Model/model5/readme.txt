MODEL 5 — Qwen3-VL-4B-Instruct QLoRA

Purpose
-------
This is the main general-purpose VLM candidate. Qwen3-VL preserves arbitrary
aspect ratios, can follow the central-line instruction, and provides a language
prior for archaic legal phrases. The exact 4B checkpoint is Apache-2.0 and
pinned to ebb281ec70b05090aa6165b016eac8ec08e71b17.

Do not substitute Qwen2.5-VL-3B: its research-only licence is unsuitable for
this challenge. This configuration is 4-bit NF4 QLoRA with deterministic SFT;
it does not call an API and never uploads challenge data.

Hardware
--------
Start with one 24 GB CUDA GPU. Very wide images create many vision tokens; use
32–48 GB if 24 GB still OOMs. Lower the processor pixel budget only after a
measured validation comparison because fine handwriting needs resolution.

Commands
--------
python model5/train.py --fold 0

python model5/infer.py \
  --adapter outputs/model5/fold0/best_adapter \
  --input data/processed/train_folds.csv \
  --fold 0 \
  --output outputs/model5/oof_fold0.csv

For final test inference, pass data/raw/Test.csv plus --sample and --submission
as documented in model1/readme.txt. Keep the five beam candidates in JSONL for
model7 rather than assuming beam rank 1 is always optimal for edit distance.

Validation priorities
---------------------
1. Full image versus center crop.
2. LoRA rank 32 versus 64.
3. Beam count and length penalty.
4. Freeze configuration, then five-fold OOF and all-data final training.

Status
------
The audited CSVs and folds are ready. GPU execution is pending; this candidate
has not yet been trained or scored.
