MODEL 59 — conservative residual outer-fold error repair
========================================================

Deployment correction (2026-09-04): the original named submission inherited
Model23's raw-source lineage error and must not be submitted. The corrected
stage is `outputs/model59/rebased_test/test.csv`, where this rule changes two
rows. Model66 is the current fully receipted descendant.

Purpose
-------
Model59 applies one deliberately narrower residual pass after Model58. It
learns only systematic errors that remain in Model58's genuine OOF outputs and
builds every held-fold map from the other four folds.

Frozen rule and leakage control
-------------------------------
Only alphabetic source cores of at least six characters are eligible. The
target must be within two edits, appear at least twice, and account for 100% of
the source core's aligned outcomes. For fold f, no prediction or label from f
enters its mapping. The deployment map is fitted from all 4,095 honest Model58
OOF prediction/label pairs, never from in-sample OCR or test labels.

Measured result
---------------
The rule changes 11/4,095 rows and raises complete OOF from 0.816080816 to
0.816206285 (+0.000125469). Folds 0, 1, and 3 improve; folds 2 and 4 are
unchanged. A seeded 5,000-sample paired bootstrap gives an incremental 95%
interval of [+0.000035865, +0.000229815], positive in 99.74% of resamples.
The cumulative Model25-to-Model59 interval is [+0.001623104, +0.002508408],
positive in every resample.

The all-data residual map contains 32 substitutions fitted from 4,095 honest
Model58 OOF pairs.

Reproduce OOF
-------------
  PYTHONPATH=src python model59/residual_crossfold_error_repair.py oof \
    --predictions outputs/model58/oof/allfold_oof.csv \
    --output-dir outputs/model59/oof

Build local test candidate
--------------------------
  PYTHONPATH=src python model59/residual_crossfold_error_repair.py apply \
    --training-predictions outputs/model58/oof/allfold_oof.csv \
    --predictions outputs/model58/rebased_test/test.csv \
    --output-dir outputs/model59/rebased_test

This is a CPU-only deterministic postprocessor. No leaderboard score is
claimed.
