MODEL 60 — outer-fold contextual spelling repair
================================================

Purpose
-------
Model60 applies only the context stage of the frozen Model35 text recipe after
Model59. The boundary and confusion stages were separately measured and
excluded because they reduced complete OOF from the context-stage peak.

Frozen rule and leakage control
-------------------------------
The language model uses train-label unigram and bigram counts plus edit distance
at most two. Its settings are fixed in config.json. For held fold f, the
language model is built only from labels in the other four folds. Deployment
uses all 4,095 clean training labels and no test labels.

Measured result
---------------
Model60 changes 81 OOF rows and raises Model59 from 0.816206285 to
0.816334380 (+0.000128095), with all five folds improving. Its incremental
5,000-sample bootstrap interval is [-0.000059868, +0.000320027], positive in
91.26% of samples. Model63 supersedes it with a safer frequency gate.

Reproduce OOF
-------------
  PYTHONPATH=src python model60/crossfold_context_repair.py oof \
    --predictions outputs/model59/oof/allfold_oof.csv \
    --output-dir outputs/model60/oof

Build local test candidate
--------------------------
  PYTHONPATH=src python model60/crossfold_context_repair.py apply \
    --predictions outputs/model59/rebased_test/test.csv \
    --output-dir outputs/model60/rebased_test \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model60_rebased_context.csv

The original `model60_crossfold_context.csv` is invalid because it inherited
the Model23 raw-source lineage error and must not be submitted. The corrected
candidate changes 23 rows and has SHA-256
73bbaf6884ba5dca8ad8d09a751a431e0e3b02e9e5ff0586358ad767a5d6bd44.
This is a CPU-only deterministic postprocessor; no leaderboard score is claimed.
