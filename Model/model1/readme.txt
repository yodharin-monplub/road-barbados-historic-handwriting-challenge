MODEL 1 — LightOnOCR-2-1B zero-shot and preprocessing baseline

Purpose
-------
This is the quickest serious baseline and a useful ensemble teacher. The
checkpoint is OCR-specific, about 1B parameters, Apache-2.0, and pinned to
c97bd377f04481830395218fa8951df9deaba756. It does not train on any external
data. The default prompt is empty because this specialized checkpoint is
designed to transcribe the supplied image directly.

Hardware
--------
The local RTX 4050 and CUDA 13 PyTorch runtime are working. The exact 2.01 GB
checkpoint passed a real one-row BF16 inference smoke. CPU inference is
possible but too slow for experiment iteration.

Verified local setup
--------------------
The isolated Python 3.12 environment is `.venv-lighton`. It shares only the
project's existing torch 2.13.0+cu130 runtime and has Transformers 5.16.1 plus
the versions in `requirements-local.txt` installed directly. This is required
because LightOnOCR-2 support starts in Transformers v5 and its native classes
must be used rather than the generic Mistral3 auto mapping.

The complete upstream snapshot is local at
`checkpoints/model1/lightonocr-2-1b`. `config_local.yaml` pins upstream commit
`c97bd377f04481830395218fa8951df9deaba756`, the Apache-2.0 model-card license,
an exact file allowlist, and every SHA-256. Inference refuses any file drift
and is network-independent.

Run the mandatory one-row smoke after other GPU work releases the device:

  PYTHONPATH=src .venv-lighton/bin/python model1/infer.py \
    --config model1/config_local.yaml \
    --input data/processed/train_folds.csv --fold 0 --limit 1 \
    --output outputs/model1/smoke_fold0.csv

Validation experiment
---------------------
Run fold 0 using the full image:

PYTHONPATH=src .venv-lighton/bin/python model1/infer.py \
  --config model1/config_local.yaml \
  --input data/processed/train_folds.csv \
  --fold 0 \
  --output outputs/model1/fold0_full.csv

Repeat with --image-variant center, enhanced, and center_enhanced. Score every
CSV with scripts/score_csv.py and keep the variant chosen by aggregate
out-of-fold score, never by looking at test images.

Test and submission
-------------------
PYTHONPATH=src .venv-lighton/bin/python model1/infer.py \
  --config model1/config_local.yaml \
  --input data/raw/Test.csv \
  --output outputs/model1/test.csv \
  --sample data/raw/SampleSubmission.csv \
  --submission submissions/model1.csv

The adjacent partial lines visible in many crops are the main risk. Center
cropping may reduce hallucinated text but can clip tall ascenders/descenders;
that tradeoff must be selected on validation. Generation is capped at 40 tokens
because the pinned tokenizer encodes every supplied target in at most 34; this
prevents rare non-terminating repetition. JSONL beside each prediction CSV
retains all beam candidates for model7.

Status
------
The pinned checkpoint, isolated runtime, local loader, drift verification, and
one-row GPU smoke all passed. The original full-view run was stopped after a
277-row diagnostic scored below zero because of repetition loops. With the
40-token cap, mild repetition penalty, and center crop, the first 200 held-out
fold-0 rows scored 0.564566638—far below the Kraken candidates. This is a
diagnostic subset, not a complete OOF claim. Zero-shot LightOnOCR is retired as
a submission candidate; its artifacts remain for reproducibility and possible
diversity analysis.
