MODEL 16 — character-confidence distribution view selector

Purpose
-------
Extend Model9's mean-confidence gate by testing robust lower-tail, geometric,
harmonic, and blended summaries of the recognizer's character confidences.
A few very uncertain characters can matter even when a line's mean confidence
is high. The output remains one complete fixed-view transcription.

Eligibility
-----------
Selection uses only recognized output text and model confidence values. It
does not use image width, height, aspect ratio, pixel count, file size, or any
other prohibited image metadata. The rule grid is fixed in source code and is
tuned on OOF predictions only.

Tune
----
  python model16/tune_distribution_selector.py \
    --truth data/processed/train_folds.csv \
    --primary outputs/model8/allfold_full_constrained.jsonl \
    --alternate outputs/model8/allfold_center_constrained.jsonl \
    --output-config outputs/model16/allfold_config.json \
    --output-predictions outputs/model16/allfold_oof.csv

Apply
-----
  python model16/apply_distribution_selector.py \
    --test data/raw/Test.csv \
    --primary outputs/model8/test_full.jsonl \
    --alternate outputs/model8/test_center.jsonl \
    --config outputs/model16/allfold_config.json \
    --output outputs/model16/test.csv

Status
------
Implementation and tests are complete. The fold-0 sweep selected the exact
same mean-confidence/0.90-length rule as Model9 and therefore added no gain.
It remains a controlled all-fold comparison, not a promoted stage.
