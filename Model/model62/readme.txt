MODEL 62 — exact-context recurring insertion repair
===================================================

Purpose
-------
Model62 restores short tokens that OCR repeatedly omits between the same exact
left and right tokens. It complements Model61, which handles replacement spans
but cannot represent a missing source span.

Frozen rule and leakage control
-------------------------------
Only one- or two-token difflib insertion blocks are eligible. The exact
left/right boundary must support the same insertion at least three times and
the insertion must explain at least half of all occurrences of that boundary.
Each held-fold map is built from Model61 OOF prediction/label pairs in the other
four folds. The deployment map uses all 4,095 honest OOF pairs and no test
labels.

Measured result
---------------
Model62 changes nine OOF rows and raises Model61 from 0.816661718 to
0.816829814 (+0.000168097); every fold improves. Its 5,000-sample incremental
bootstrap interval is [+0.000071633, +0.000283833], positive in every sample.
The validated test candidate changes two rows and has SHA-256
5839f84ee27ef3d75c32a77a2576224b29fd5fada2f38e9af27b0be3c20d1c27.

Reproduce OOF
-------------
  PYTHONPATH=src python model62/context_insertion_repair.py oof \
    --predictions outputs/model61/oof/allfold_oof.csv \
    --output-dir outputs/model62/oof

Build local test candidate
--------------------------
  PYTHONPATH=src python model62/context_insertion_repair.py apply \
    --training-predictions outputs/model61/oof/allfold_oof.csv \
    --predictions outputs/model61/test_candidate/test.csv \
    --output-dir outputs/model62/test_candidate \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model62_context_insertions.csv

This branch is superseded by Models63–65. This is a CPU-only deterministic
postprocessor; no leaderboard score is claimed.
