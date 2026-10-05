MODEL 57 — fold-safe contextual medial-case restoration
=======================================================

Deployment correction (2026-09-04): the original named submission inherited
Model23's raw-source lineage error and must not be submitted. The corrected
stage is `outputs/model57/rebased_test/test.csv`, where this rule changes seven
rows. Model66 is the current fully receipted descendant.

Purpose
-------
Model57 layers after Model56. It learns the exact spelling of a medial token
conditioned on the casefolded alphabetic cores of its immediate left and right
neighbors. It accepts only case-changing proposals, optionally accompanied by
edge punctuation; punctuation-only proposals are deliberately excluded.

Frozen rule and leakage control
-------------------------------
The exact token spelling must be unanimous in at least 10 fold-excluded label
examples for the left/current/right core triple. Every OOF map excludes the
row's outer fold; deployment uses all clean labels. IDs, images, confidence,
test labels, and held-out targets never reach the transformation. Only matched
medial token spans change, preserving all other whitespace byte-for-byte.

Measured result
---------------
The frozen rule changes 39/4,095 rows and raises complete OOF from Model56's
0.814483608 to 0.814617639 (+0.000134030). Every fold improves. A seeded
5,000-sample paired bootstrap gives an incremental 95% interval of
[-0.000011023, +0.000284408], positive in 96.38% of resamples. The cumulative
Model25-to-Model57 interval is [+0.000238092, +0.000704935], positive in every
resample. The isolated stage is therefore less certain than the full chain.

Reproduce OOF
-------------
  PYTHONPATH=src python model57/restore_contextual_medial_case.py oof \
    --predictions outputs/model56/oof/allfold_oof.csv \
    --output-dir outputs/model57/oof

Build local test candidate
--------------------------
  PYTHONPATH=src python model57/restore_contextual_medial_case.py apply \
    --predictions outputs/model56/rebased_test/test.csv \
    --output-dir outputs/model57/rebased_test

This is a CPU-only deterministic postprocessor. No leaderboard score is
claimed.
