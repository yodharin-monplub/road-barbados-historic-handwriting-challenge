MODEL 25 — high-support unambiguous casing restoration

DEPLOYMENT NOTE (2026-09-04): `submissions/model25_case.csv` is obsolete and
must not be submitted because it inherited Model23's raw-source lineage error.
The corrected stage is `outputs/model25/test_from_stage15.csv`; the current
fully receipted descendant is Model66.

Purpose
-------
Restore systematic casing errors for tokens whose spelling is unambiguous in
the supplied labels. Typical full-training mappings include john -> John,
lord -> Lord, richard -> Richard, july -> July, and capt -> Capt.

Method and leakage control
--------------------------
For each lowercase token key, the exact-case form must be unanimous and occur
at least 20 times. During OOF evaluation, the map for a row is learned from the
other four folds only. The test map uses all training labels. The rule changes
case only: spelling, punctuation, carets, and whitespace are otherwise
preserved. It uses no image metadata, test labels, external corpus, or manual
row edits.

Measured result
---------------
Layered after Model 24, five-fold OOF increases from 0.813705301 to
0.814147514 (+0.000442213). Every fold improves:

  fold 0  0.817450092
  fold 1  0.813936918
  fold 2  0.812984515
  fold 3  0.812266320
  fold 4  0.814095845

A seeded 5,000-sample paired row bootstrap against Model 24 gives a 95%
percentile interval of [+0.000219, +0.000677], with a positive gain in every
resample.

Reproduce OOF
-------------
  python model25/restore_unambiguous_case.py \
    --train data/processed/train_folds.csv \
    --predictions outputs/model24/allfold_oof.csv \
    --config model25/config.json \
    --output outputs/model25/allfold_oof.csv \
    --oof

Build the test candidate
------------------------
  python model25/restore_unambiguous_case.py \
    --train data/processed/train_folds.csv \
    --predictions outputs/model24/test_from_stage15.csv \
    --config model25/config.json \
    --output outputs/model25/test_from_stage15.csv

The corrected descendant submission is produced by Model65.
