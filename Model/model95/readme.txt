Model95: direct Qwen3-VL 30B-A3B image transcription
=====================================================

Goal
----
Test a large open-weight multimodal model as an independent OCR witness.  Unlike
Model93, this model sees the manuscript crop itself and no prior OCR text.  Its
exact user-requested prompt is:

  read this image and convert to text, it's old english. Do not change to normal english

Model
-----
Qwen/Qwen3-VL-30B-A3B-Instruct, pinned at commit
9c4b90e1e4ba969fd3b5378b57d966d725f1b86c.  The model card declares Apache-2.0.
All metadata and all 13 weight shards are pinned by SHA-256 in config.json and
are fully verified before inference.  BF16 weights occupy about 57.9 GiB; the
runner requires a single GPU with at least 75 GiB, so use an A100 80GB.

Leakage control
---------------
prepare.py writes only ID, split/fold, image filename, and image SHA-256.  It
does not write targets and the prompt contains no retrieved labels.  Fold 0 is
the development gate.  Do not run test inference until fold 0 gains at least
0.001 over Model88 and untouched-fold confirmation also passes.

Local preparation
-----------------

  python3 model95/prepare.py --split oof --fold 0 \
    --output outputs/model95/fold0/images.jsonl

RunPod A100 80GB
----------------

  bash model95/setup_runpod.sh "$PWD"
  bash model95/run_fold0.sh "$PWD"

Inference is deterministic, offline after the audited snapshot download, and
resumable through details.jsonl.  It verifies every input image hash immediately
before use.  Raw model text and minimally unwrapped predictions are both saved.

Artifacts
---------

  outputs/model95/qwen3_vl_snapshot_audit.json
  outputs/model95/fold0/images.jsonl
  outputs/model95/fold0/images.jsonl.manifest.json
  outputs/model95/fold0/direct/details.jsonl
  outputs/model95/fold0/direct/predictions.csv
  outputs/model95/fold0/evaluation/comparison.csv
  outputs/model95/fold0/evaluation/report.json

No Zindi submission is created automatically.
