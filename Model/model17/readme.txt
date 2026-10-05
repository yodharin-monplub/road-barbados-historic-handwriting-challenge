MODEL 17 — token-confidence fusion of full and center OCR views

Purpose
-------
Align the two fixed-view transcriptions at word level, choose a base line with
Model9's legal confidence rule, and replace only aligned donor words whose
character-confidence quality is higher by a frozen margin. Normalized edit
distance gates reduce alignment-driven replacements.

Eligibility
-----------
The model uses recognized text and the recognizer's character confidence only.
No image dimensions, aspect ratio, pixel count, file metadata, hidden labels,
or manually authored test corrections are used.

Tune
----
  python model17/tune_token_fusion.py \
    --truth data/processed/train_folds.csv \
    --primary outputs/model8/allfold_full_constrained.jsonl \
    --alternate outputs/model8/allfold_center_constrained.jsonl \
    --output-config outputs/model17/allfold_config.json \
    --output-predictions outputs/model17/allfold_oof.csv

Apply
-----
  python model17/apply_token_fusion.py \
    --test data/raw/Test.csv \
    --primary outputs/model8/test_full.jsonl \
    --alternate outputs/model8/test_center.jsonl \
    --config outputs/model17/allfold_config.json \
    --output outputs/model17/test.csv

Status
------
Implementation and unit tests are complete. On fold 0, token fusion improved
the raw two-view multi-score from 0.742542 to 0.743958 while changing 213
tokens. The model is promoted only after an aggregate and per-fold five-fold
OOF comparison.
