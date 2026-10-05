MODEL 19 — three-view confidence + character-LM selector

Purpose
-------
Select among full, center-band, and autocontrast-enhanced PP-OCR outputs. A
fold-excluded character language model is combined with mean OCR confidence,
while two view offsets calibrate systematic confidence differences. The full
view is always eligible; cropped/enhanced candidates must pass a length guard.

Eligibility
-----------
The selector uses only OCR text/confidence and supplied fold-excluded labels.
It never uses image dimensions, aspect ratio, pixel count, file metadata,
external corpora, manual test edits, or leaderboard labels.

Tune
----
  python model19/tune_three_view_lm.py \
    --truth data/processed/train_folds.csv \
    --primary outputs/model8/allfold_full_constrained.jsonl \
    --secondary outputs/model8/allfold_center_constrained.jsonl \
    --tertiary outputs/model8/allfold_enhanced_constrained.jsonl \
    --output-config outputs/model19/allfold_config.json \
    --output-predictions outputs/model19/allfold_oof.csv

Apply
-----
  python model19/apply_three_view_lm.py \
    --train data/processed/train_folds.csv \
    --test data/raw/Test.csv \
    --primary outputs/model8/test_full.jsonl \
    --secondary outputs/model8/test_center.jsonl \
    --tertiary outputs/model8/test_enhanced.jsonl \
    --config model19/allfold_config.json \
    --output outputs/model19/test.csv

Status
------
Promoted. Five-fold raw OOF is 0.753781973, improving the two-view Model 18
selector on every fold. After the fold-safe Models 11–15 chain, OOF is
0.810909662 (weighted WER 3.272472; weighted CER 5.801107).
