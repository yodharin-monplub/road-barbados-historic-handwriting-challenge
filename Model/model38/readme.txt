MODEL 38 — bounded robustness audit of the Model33 fold-0 selector

Question
--------
Is Model33's 0.842145 fold-0 score stable under reasonable deterministic
repartitioning, or is it a lucky partition / 54-feature overfit?

Answer
------
Conditional on the already fixed Model25 and Model22 epoch-7 candidates, the
selector result is robust to this bounded audit. It is not explained by the
original partition seed, and most of the gain remains with a predeclared
30-feature compact representation.

The fixed panel contains the original Model33 seed plus three alternative
label-independent salts. Every salt evaluates the untouched 54-feature,
L2=0.08 Model33 specification and one 30-feature compact ablation. All eight
runs are reported; none is selected because of its held-out score.

Results
-------
  Model25 baseline                         0.817450092
  Model22 epoch-7 candidate                0.813019600
  unattainable row-oracle ceiling          0.854166125

  full 54-feature selector, 4 salts
    minimum / mean / maximum score         0.841943822 / 0.842291861 / 0.842819931
    population standard deviation          0.000325048
    minimum / mean gain over Model25       +0.024493730 / +0.024841769
    Model25-beating runs                    4 / 4
    runs with all 5 partitions improving   4 / 4
    worst of all 20 partition gains        +0.012974566

  compact 30-feature selector, 4 salts
    minimum / mean / maximum score         0.841187311 / 0.841788248 / 0.842571072
    population standard deviation          0.000572064
    minimum / mean gain over Model25       +0.023737219 / +0.024338157
    Model25-beating runs                    4 / 4
    runs with all 5 partitions improving   4 / 4
    worst of all 20 partition gains        +0.012746019

The original 0.842145141 run is close to the alternative-salt mean, not an
outlier. Removing 24 features costs only about 0.00050 on mean score, while the
compact model still gains at least 0.02374 over Model25 in every complete run.
This is strong evidence against both the lucky-split and all-54-features
explanations on fold 0.

Selection and calibration stability
-----------------------------------
For the full selector, pairwise candidate-choice agreement across salts is
91.55% on average (minimum 90.48%). A total of 84.62% of rows receive the same
choice under all four salts; 15.38% fall in the deliberately broad unstable
choice-rate band [0.25, 0.75]. The selected Model22 count is tightly bounded at
362-370 rows.

The compact/full choices agree on 90.84%-91.82% of rows under matched salts.
For the full model, importance-weighted choice accuracy ranges from 84.30% to
85.43%. Weighted Brier score ranges 0.11992-0.12376 and ten-bin weighted ECE
ranges 0.07097-0.07997. About 12.82%-13.92% of probabilities are within
[0.45, 0.55]. Thus ranking/selection is stable, while the probability values
are only moderately calibrated and should not be interpreted as literal OCR
correctness probabilities.

Leakage control
---------------
For each salt, five balanced outer partitions are assigned from ID hashes only.
For each reported partition:

  * its IDs are absent from selector labels and metric-importance weights;
  * its targets are absent from the language model used to score it;
  * selector-training language features are themselves five-way nested OOF;
  * its score is computed only after its probabilities and choices are frozen.

The report records fit/evaluation ID hashes, nested group sizes, and explicit
exclusion assertions for all 20 outer fits. Changing a held partition's labels
cannot change its probabilities; this invariant has a unit test.

The audit is conditional on Model22 checkpoint_07. That checkpoint was chosen
in the earlier fold-0 epoch screen, so Model38 does not remove checkpoint-screen
optimism and does not turn this into a five-fold OOF estimate. Fixed-epoch
folds 1-4 remain the decisive next validation.

Reproduce
---------
  PYTHONPATH=src .venv-kraken/bin/python model38/run.py \
    --config model38/config.json \
    --output-dir outputs/model38_rerun

The runner caps BLAS/OMP libraries at one CPU thread so it can coexist with GPU
training. On the audited machine the bounded eight-run panel took 130 seconds
and at most 61.5 MB RSS.

Tests
-----
  PYTHONPATH=src .venv-kraken/bin/python -m unittest \
    tests.test_model38_selector_robustness -v

The tests verify exact feature masks, optimizer parity with Model33, held-group
label-mutation invariance, calibration math, and source-hash failure.

Artifacts
---------
  outputs/model38/report.json
    Complete method, provenance, all seed split audits, and summaries.

  outputs/model38/runs.csv
    All 8 run-level scores, partition minima, calibration, choice counts, and
    coefficient norms. SHA-256:
    d3685d91e417ad327acb392b3ce8b74648047bbbf6fd99c026f168af8d833360

  outputs/model38/selections.csv
    All 6,552 held-out probabilities/choices with partition and oracle audit
    fields. SHA-256:
    c807c8509e4ed942a64752061b40ba8cec7fb021b66c02dd6261b88ada1f0784

  outputs/model38/summary.csv
    Compact per-variant distribution/stability table. SHA-256:
    18899d647bc6641cdd9fd9f4b99e47599f2ca039ee42c13260ee5c77ae9d018c

Development-run note
--------------------
An initially proposed 8-salt x 8-variant sweep was stopped after roughly 2.5
minutes at the coordinator's request to avoid competing with active Qwen
training. The implementation writes only after the complete declared panel, so
that aborted attempt produced no score or artifact and did not inform a
best-run choice. The final config was reduced before results to the bounded
4-salt x 2-variant panel reported above.
