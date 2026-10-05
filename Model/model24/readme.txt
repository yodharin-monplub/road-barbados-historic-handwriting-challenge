MODEL 24 — fold-excluded historical-abbreviation caret restoration

DEPLOYMENT NOTE (2026-09-04): the original test/submission artifact inherited
Model23's raw-source lineage error and must not be submitted. The corrected
stage is `outputs/model24/test_from_stage15.csv` and is rooted at the verified
postprocessed Model19 vector.

Purpose
-------
Restore carets in recurring historical abbreviations after Model 23. Examples
in the supplied labels include W^m, M^r, Adm^rs, Ex^rs, and w^tsoever.

Method and leakage control
--------------------------
Tokens are grouped after removing only the caret character. A caretless OCR
token is replaced when the most frequent exact-case training variant contains
a caret, has at least one example, and accounts for at least half of variants.
Existing carets are never removed or moved. During OOF evaluation, each row's
mapping is learned from the other four folds only. Test mappings use all train
labels. No image metadata, test label, external corpus, or manual row edit is
used.

Measured result
---------------
Layered after Model 23, this raises five-fold OOF from 0.813091389 to
0.813705301 (+0.000613912). The fold scores are:

  fold 0  0.817038894
  fold 1  0.813641700
  fold 2  0.812692382
  fold 3  0.811553613
  fold 4  0.813596549

A seeded 5,000-sample paired row bootstrap against Model 23 gives a 95%
percentile interval of [+0.000292, +0.000935] and a positive gain in 99.98% of
resamples.

Reproduce OOF
-------------
  python model24/restore_carets.py \
    --train data/processed/train_folds.csv \
    --predictions outputs/model23/allfold_oof.csv \
    --config model24/config.json \
    --output outputs/model24/allfold_oof.csv \
    --oof

Build the test candidate
------------------------
  python model24/restore_carets.py \
    --train data/processed/train_folds.csv \
    --predictions outputs/model23/test_from_stage15.csv \
    --config model24/config.json \
    --output outputs/model24/test_from_stage15.csv

The corrected descendant submission is produced by Model65.
