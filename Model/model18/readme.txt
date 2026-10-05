MODEL 18 — fold-safe character-language-model view selector

Purpose
-------
Combine each view's mean OCR confidence with the average character n-gram
log-probability of its recognized text. The legal documents repeat distinctive
character sequences, so a fold-excluded label language model may resolve cases
where raw recognizer confidence is misleading.

Leakage and eligibility
-----------------------
For OOF rows, each character LM is fit only on the other four folds. For test,
one LM is fit on all supplied training labels. The gate consumes recognized
text and OCR confidence only; it never uses image dimensions, pixels, file
metadata, external text, manual rules, or leaderboard labels.

Tune
----
  python model18/tune_lm_selector.py \
    --truth data/processed/train_folds.csv \
    --primary outputs/model8/allfold_full_constrained.jsonl \
    --alternate outputs/model8/allfold_center_constrained.jsonl \
    --output-config outputs/model18/allfold_config.json \
    --output-predictions outputs/model18/allfold_oof.csv

Apply
-----
  python model18/apply_lm_selector.py \
    --train data/processed/train_folds.csv \
    --test data/raw/Test.csv \
    --primary outputs/model8/test_full.jsonl \
    --alternate outputs/model8/test_center.jsonl \
    --config outputs/model18/allfold_config.json \
    --output outputs/model18/test.csv

Status
------
Promoted. The raw selector scores 0.749226572 across five folds versus
0.737254681 for Model 9, with a gain on every fold. After Models 11–15, the
five-fold score is 0.808761727. The validated frozen test candidate is
submissions/model18_lm_allfold.csv.
