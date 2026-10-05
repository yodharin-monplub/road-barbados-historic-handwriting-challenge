MODEL 55 — fold-safe contextual final-token canonicalization
============================================================

Deployment correction (2026-09-04): the original named submission inherited
Model23's raw-source lineage error and must not be submitted. The corrected
stage is `outputs/model55/rebased_test/test.csv`, where this rule changes 10
rows. Model66 is the current fully receipted descendant.

Purpose
-------
Model55 layers after Model54 and conditions the exact final-token spelling on
the casefolded alphabetic cores of both the preceding and final tokens. This
captures recurring line endings whose punctuation/case is stable only in local
context.

Frozen rule and leakage control
-------------------------------
One exact final spelling must have at least six fold-excluded label occurrences
and at least 80% of all occurrences for that previous-core/final-core pair.
Every OOF row's map excludes its outer fold; the test map uses all clean labels.
Only the final non-whitespace span changes, and no image metadata, test label,
confidence, or manual row list is used.

Measured result
---------------
The rule changes 30/4,095 rows and raises complete OOF from Model54's
0.814251941 to 0.814343584 (+0.000091642). Every fold improves. This brings the
cumulative gain over Model25 to +0.000196070. A seeded 5,000-sample paired
bootstrap for the cumulative Model25-to-Model55 change gives a 95% interval of
[+0.000047328, +0.000349005], positive in 99.58% of resamples. The incremental
Model54-to-Model55 interval is [-0.000027105, +0.000213016], so Model55's extra
stage is supported more by all-fold consistency and cumulative evidence than by
standalone bootstrap certainty.

Reproduce OOF
-------------
  PYTHONPATH=src python model55/restore_contextual_final_tokens.py oof \
    --predictions outputs/model54/oof/allfold_oof.csv \
    --output-dir outputs/model55/oof

Build local test candidate
--------------------------
  PYTHONPATH=src python model55/restore_contextual_final_tokens.py apply \
    --predictions outputs/model54/rebased_test/test.csv \
    --output-dir outputs/model55/rebased_test

This is a CPU-only deterministic postprocessor. No leaderboard score is
claimed.
