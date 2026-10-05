MODEL 58 — outer-fold systematic error repair
==============================================

Deployment correction (2026-09-04): the original named submission inherited
Model23's raw-source lineage error and must not be submitted. The corrected
stage is `outputs/model58/rebased_test/test.csv`, where this rule changes 10
rows. Model66 is the current fully receipted descendant.

Purpose
-------
Model58 layers after Model57 and relearns recurring OCR-to-truth word errors
from genuine OOF prediction/label pairs. Unlike the earlier within-fold inner
cross-fit, each held fold is repaired by a map fitted on the other four complete
outer folds, matching how a final test map can be fitted from all honest OOF
pairs.

Frozen rule and leakage control
-------------------------------
Aligned alphabetic source/target cores may differ by at most two edits, must be
at least two characters, occur at least twice, and have at least 2/3 precision
among all outcomes for the source core. For fold f, every mapping pair has
truth.fold != f. The test mapping uses all 4,095 Model57 OOF prediction/label
pairs. No in-sample OCR predictions, test labels, image metadata, manual IDs,
or leaderboard feedback are used.

Measured result
---------------
The fixed rule changes 195/4,095 rows and raises complete OOF from Model57's
0.814617639 to 0.816080816 (+0.001463178). All five folds improve, with gains
from +0.001205 to +0.001860. A seeded 5,000-sample paired bootstrap gives an
incremental 95% interval of [+0.001113787, +0.001821606], positive in every
resample. The cumulative Model25-to-Model58 interval is
[+0.001509568, +0.002366016], also positive in every resample.

The all-data map contains 98 substitutions fitted from 4,095 honest OOF pairs.

Reproduce OOF
-------------
  PYTHONPATH=src python model58/crossfold_error_repair.py oof \
    --predictions outputs/model57/oof/allfold_oof.csv \
    --output-dir outputs/model58/oof

Build local test candidate
--------------------------
  PYTHONPATH=src python model58/crossfold_error_repair.py apply \
    --training-predictions outputs/model57/oof/allfold_oof.csv \
    --predictions outputs/model57/rebased_test/test.csv \
    --output-dir outputs/model58/rebased_test

This is a CPU-only deterministic postprocessor. No leaderboard score is
claimed.
