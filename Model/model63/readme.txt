MODEL 63 — frequency-gated contextual spelling repair
=====================================================

Purpose
-------
Model63 replaces Model60 with a safer version of the same outer-fold context
repair. A row is accepted only when every proposed replacement candidate occurs
at least twice in the label language model.

Leakage control
---------------
Every held-fold language model excludes that fold's labels. The test language
model uses all 4,095 clean training labels and no test labels. The fixed row
gate is applied identically to OOF and test predictions.

Measured result
---------------
The gate accepts 67 of 81 proposed OOF rows and raises Model59 from
0.816206285 to 0.816401608 (+0.000195322), with every fold improving. Its
5,000-sample interval is [+0.000030266, +0.000364395], positive in 98.92% of
samples. On the corrected test lineage it accepts 17 of 23 proposed rows.

Reproduce OOF
-------------
  PYTHONPATH=src python model63/gated_context_repair.py oof \
    --predictions outputs/model59/oof/allfold_oof.csv \
    --output-dir outputs/model63/oof

Build local test stage
----------------------
  PYTHONPATH=src python model63/gated_context_repair.py apply \
    --predictions outputs/model59/rebased_test/test.csv \
    --output-dir outputs/model63/test_candidate

This is a CPU-only deterministic postprocessor. No leaderboard score is
claimed.
