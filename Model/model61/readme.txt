MODEL 61 — exact recurring one/two-token span repair
====================================================

Purpose
-------
Model61 targets exact recurring replacement spans after Model60, including
split/merge errors that a one-token-to-one-token spelling map cannot express.
Examples include "with in" to "within" and "here of" to "hereof".

Frozen rule and leakage control
-------------------------------
Each mapping covers a one- or two-token difflib replacement block, occurs at
least three times, and explains at least half of all exact source-span
occurrences. Longest matches are applied first. A held fold's map is fitted
only from genuine Model60 OOF prediction/label pairs in the other four folds.
The test map uses all 4,095 honest OOF pairs and no test labels.

Measured result
---------------
Model61 changes 66 OOF rows and raises Model60 from 0.816334380 to
0.816661718 (+0.000327338); every fold improves. Its 5,000-sample incremental
bootstrap interval is [+0.000088885, +0.000559290], positive in 99.66% of
samples. The validated test candidate changes 26 rows and has SHA-256
599e1349088baf6e5ab283418c3290a61fc1f8a64adca3453991d3403b77d66a.

Reproduce OOF
-------------
  PYTHONPATH=src python model61/exact_span_repair.py oof \
    --predictions outputs/model60/oof/allfold_oof.csv \
    --output-dir outputs/model61/oof

Build local test candidate
--------------------------
  PYTHONPATH=src python model61/exact_span_repair.py apply \
    --training-predictions outputs/model60/oof/allfold_oof.csv \
    --predictions outputs/model60/rebased_test/test.csv \
    --output-dir outputs/model61/test_candidate \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model61_exact_spans.csv

This branch is superseded by Models63–65. This is a CPU-only deterministic
postprocessor; no leaderboard score is claimed.
