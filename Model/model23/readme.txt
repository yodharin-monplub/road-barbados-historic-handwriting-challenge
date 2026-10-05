MODEL 23 — deterministic ampersand normalization

DEPLOYMENT NOTE (2026-09-04): the original `outputs/model23/test.csv` and
`submissions/model23_ampersand.csv` were accidentally built from raw
`outputs/model19/test.csv`, not the postprocessed Model19 stage-15 vector. They
must not be submitted. The corrected stage is
`outputs/model23/test_from_stage15.csv`, built from
`outputs/pipeline_allfold/three-view_model15.csv`.

Purpose
-------
Repair a systematic PP-OCR transcription error: the historical ampersand glyph
is frequently decoded as a plus sign. This stage replaces every literal "+"
with "&" after Model 19's fold-safe text-repair chain.

Evidence
--------
The clean training references contain 756 ampersands and only 6 plus signs;
Model 19 predictions contain 452 ampersands and 181 plus signs. On five-fold
OOF, 162 of the most frequent remaining word substitutions are "&" -> "+".

The deterministic rule raises OOF from 0.810909662 to 0.813091389
(+0.002181728). Every fold improves:

  fold 0  0.816515621
  fold 1  0.812353184
  fold 2  0.812134574
  fold 3  0.810901108
  fold 4  0.813548492

A seeded 5,000-sample paired row bootstrap gives a 95% percentile interval of
[+0.001790, +0.002613], with a positive gain in every resample. Of the 143 OOF
rows changed, 139 improve, 2 regress, and 2 have equal metric cost.

Reproduce OOF
-------------
  python model23/normalize_ampersands.py \
    --predictions outputs/model19/allfold_stage15.csv \
    --output outputs/model23/allfold_oof.csv \
    --truth data/processed/train_folds.csv

Build the test candidate
------------------------
  python model23/normalize_ampersands.py \
    --predictions outputs/pipeline_allfold/three-view_model15.csv \
    --output outputs/model23/test_from_stage15.csv

Limitations
-----------
Five training lines contain six true plus signs, so the rule is not logically
perfect. In practice the systematic ampersand recovery dominates this rare
case on every fold. The transformation uses only OCR text and training-label
statistics; it does not use image metadata or test labels.

The corrected descendant submission is produced by Model65.
