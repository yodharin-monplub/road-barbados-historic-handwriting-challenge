MODEL 9 — confidence-gated multi-view PP-OCR

Purpose
-------
Run the same recognizer on two fixed transforms: the complete crop and a 74%
center band. Select between their complete transcriptions using only mean OCR
character confidence and output length. This deliberately does not use image
size, aspect ratio, or pixel count, because Zindi's general computer-vision
rules prohibit image metadata as a model/submission feature.

Why this is separate from model7
--------------------------------
Model7 is a heterogeneous text-consensus ensemble. Model9 calibrates two views
of one CTC model before that stage. The fixed search grid contains nine
confidence margins and four minimum output-length ratios; parameters are tuned
with the exact local metric and never from leaderboard feedback.

Tune on OOF JSONL
-----------------
The JSONL files beside model8 prediction CSVs contain the needed confidence:

  python model9/tune_selector.py \
    --truth data/processed/train_folds.csv \
    --fold 0 \
    --primary outputs/model8/base_fold0_full.jsonl \
    --alternate outputs/model8/base_fold0_center.jsonl \
    --output-config outputs/model9/fold0_confidence_selector.json \
    --output-predictions outputs/model9/fold0_confidence_oof.csv

Repeat both fixed views for folds 1–4, combine each view's non-overlapping OOF
JSONL, then refit without `--fold`. Require stable per-fold gains before
freezing the test rule.

Apply to test
-------------
Generate both fixed test views with model8, then run:

  python model9/apply_selector.py \
    --test data/raw/Test.csv \
    --primary outputs/model8/test_full.jsonl \
    --alternate outputs/model8/test_center.jsonl \
    --config outputs/model9/confidence_selector.json \
    --output outputs/model9/test.csv \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model9.csv

Status
------
The deterministic confidence tuner, per-fold reporting, test application, and
submission writer are implemented. On fold 0, the initial unrestrained views
score 0.730780 (full), 0.726881 (center), and 0.741768 after confidence gating
with a 0.90 length guard. Constraining both sources to Train.csv's character
inventory raises the ensemble to 0.742542. Multi-fold confirmation remains
required. Both fixed test views are complete; the frozen fold-0 rule selected
the center view for 739 of 1,374 test rows and produced `outputs/model9/test.csv`.
