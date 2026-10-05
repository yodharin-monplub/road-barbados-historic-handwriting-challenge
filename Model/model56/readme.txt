MODEL 56 — fold-safe contextual initial-token canonicalization
==============================================================

Deployment correction (2026-09-04): the original named submission inherited
Model23's raw-source lineage error and must not be submitted. The corrected
stage is `outputs/model56/rebased_test/test.csv`, where this rule changes four
rows. Model66 is the current fully receipted descendant.

Purpose
-------
Model56 layers after Model55 and conditions the exact spelling of the first
token on the casefolded alphabetic cores of the first two tokens. It restores
stable initial capitalization or edge punctuation without imposing a global
case rule on mid-line occurrences.

Frozen rule and leakage control
-------------------------------
The exact initial spelling must have at least 20 fold-excluded label examples
and at least 90% dominance for its initial-core/next-core pair. Every OOF map
excludes the row's outer fold; deployment uses all clean labels. Only the first
non-whitespace span changes. The rule uses no image metadata, test labels,
confidence, manual IDs, or held-out target at decision time.

Measured result
---------------
The rule changes 14/4,095 rows and raises complete OOF from Model55's
0.814343584 to 0.814483608 (+0.000140025). Folds 0--3 improve and fold 4 is
unchanged. A seeded 5,000-sample paired bootstrap gives an incremental 95%
interval of [+0.000048345, +0.000245647], positive in 99.96% of resamples.
The cumulative Model25-to-Model56 interval is [+0.000158341, +0.000519908],
positive in every resample.

Reproduce OOF
-------------
  PYTHONPATH=src python model56/restore_contextual_initial_tokens.py oof \
    --predictions outputs/model55/oof/allfold_oof.csv \
    --output-dir outputs/model56/oof

Build local test candidate
--------------------------
  PYTHONPATH=src python model56/restore_contextual_initial_tokens.py apply \
    --predictions outputs/model55/rebased_test/test.csv \
    --output-dir outputs/model56/rebased_test

This is a CPU-only deterministic postprocessor. No leaderboard score is
claimed.
