Model93: local Qwen historical-English OCR verifier
====================================================

Status
------
The A100 fold-0 screen completed on 2026-09-06. Raw Model93 did not promote:

  Model88 fold-0 baseline: 0.8989699979153494
  Model93 aggressive:      0.8979901609957579  (-0.0009798369195915)
  Model93 learned gate:    0.8970314483069086  (-0.0019385496084408)

All 819 rows completed with one validity fallback. The full 4,095-row OOF
artifact was generated only to confirm the separately numbered Model94
absolute-edit safety experiment; Model93 itself remains rejected.

Model and eligibility
---------------------
Model93 uses the public, ungated Qwen/Qwen3-30B-A3B checkpoint at immutable
revision ad44e777bcd18fa416d9da3bd8f70d33ebb85d39. The Hugging Face model card
declares Apache-2.0, 30.532B BF16 parameters in total, and 3.3B active per
token. BF16 weights need about 61 GB, so the required target is an A100 80 GB.
The model is used locally; no hosted inference API is called.

Data contract
-------------
The current Model88 prediction is the primary witness. Full, enhanced, center,
center-enhanced LightOnOCR predictions and corrected Kraken are additional
noisy witnesses. Four examples are retrieved from supplied competition data:
three near OCR corrections and one near identity example. For every OOF row,
all retrieved labels come from other canonical folds. Test retrieval may use
all supplied training folds. No external dictionary or corpus is used.

The instruction identifies the register as seventeenth-century Early Modern
legal English and explicitly prohibits modernization, translation, expansion
of scribal abbreviations, invented content, or explanations. Thinking is off
and decoding is deterministic. Malformed, blank, non-ASCII, overlength, large
edit, or word-count-outlier outputs fall back to Model88.

Fold-0 workflow
---------------
Prepare prompts locally before starting the paid pod:

  PYTHONPATH=src python3 model93/prepare.py --config model93/config.json \
    --split oof --fold 0 --output outputs/model93/prompts/fold0.jsonl

On the A100 80 GB pod, install the isolated requirements and run:

  bash model93/setup_runpod.sh "$PWD"

  PYTHONPATH=src .venv-model93/bin/python model93/infer.py --config model93/config.json \
    --prompts outputs/model93/prompts/fold0.jsonl \
    --model-audit outputs/model93/qwen_snapshot_audit.json \
    --output outputs/model93/fold0/aggressive.csv \
    --details outputs/model93/fold0/generations.jsonl --batch-size 32

Interrupted inference is resumable by repeating the command with --resume.
The convenience command `bash model93/run_fold0_runpod.sh "$PWD"` performs the
GPU check, install/audit, inference, automatic resume selection, and final
artifact checksums.
After copying the signed artifacts back, evaluate raw aggressive correction
and an inner-cross-fitted Model88-versus-Qwen gate:

  PYTHONPATH=src python3 model93/evaluate.py \
    --truth data/processed/train_folds.csv \
    --primary outputs/model88/word_rover_vs_kraken_selector/crossfit_predictions.csv \
    --llm outputs/model93/fold0/aggressive.csv \
    --selector-config model79/config.json --fold 0 --minimum-gain 0.001 \
    --output-dir outputs/model93/fold0/evaluation

Only a gated improvement of at least +0.001 promotes the remaining folds.
After generating and combining all five aggressive OOF files, fit the frozen
gate on fold 0 and evaluate only untouched folds 1--4:

  PYTHONPATH=src python3 model93/confirm.py \
    --truth data/processed/train_folds.csv \
    --primary outputs/model88/word_rover_vs_kraken_selector/crossfit_predictions.csv \
    --llm outputs/model93/allfold_aggressive_oof.csv \
    --selector-config model79/config.json \
    --output-dir outputs/model93/confirmation

Confirmation requires +0.0005, at least three nonnegative folds, and a paired
bootstrap positive fraction of at least 0.95. Only then is a deployment gate
fitted on all OOF rows and applied to test predictions.

Combine the five fold CSVs in canonical truth order:

  PYTHONPATH=src python3 scripts/combine_fold_outputs.py \
    --truth data/processed/train_folds.csv \
    --input 0=outputs/model93/fold0/aggressive.csv \
    --input 1=outputs/model93/fold1/aggressive.csv \
    --input 2=outputs/model93/fold2/aggressive.csv \
    --input 3=outputs/model93/fold3/aggressive.csv \
    --input 4=outputs/model93/fold4/aggressive.csv \
    --output outputs/model93/allfold_aggressive_oof.csv

If confirmation passes, prepare and infer the test prompts with the same
audited checkpoint, then create both candidate submissions:

  PYTHONPATH=src python3 model93/prepare.py --config model93/config.json \
    --split test --output outputs/model93/prompts/test.jsonl

  PYTHONPATH=src .venv-model93/bin/python model93/infer.py --config model93/config.json \
    --prompts outputs/model93/prompts/test.jsonl \
    --model-audit outputs/model93/qwen_snapshot_audit.json \
    --output outputs/model93/test/aggressive.csv \
    --details outputs/model93/test/generations.jsonl --batch-size 32

  PYTHONPATH=src python3 model93/deploy.py \
    --confirmation-report outputs/model93/confirmation/report.json \
    --truth data/processed/train_folds.csv --test data/raw/Test.csv \
    --sample data/raw/SampleSubmission.csv \
    --primary-oof outputs/model88/word_rover_vs_kraken_selector/crossfit_predictions.csv \
    --llm-oof outputs/model93/allfold_aggressive_oof.csv \
    --primary-test outputs/model88/test.csv \
    --llm-test outputs/model93/test/aggressive.csv \
    --selector-config model79/config.json \
    --output-dir outputs/model93/deployment

The deploy step refuses to run unless confirmation/report.json says promoted.
It writes submission_gated.csv and submission_aggressive.csv, preserving the
official sample schema. The gated file is the default submission candidate;
the aggressive file exists so the requested raw-versus-gated comparison is
fully auditable.

Nothing in Model93 submits to Zindi automatically.
