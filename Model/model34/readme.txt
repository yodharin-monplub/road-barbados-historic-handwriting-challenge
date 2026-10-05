MODEL 34 — leakage-safe Model22/Model25 token-confidence fusion

Purpose
-------
Use the complementary recognition errors of the fine-tuned Model22 Kraken
model and the established Model25 postprocessed predictions. Model25 is always
the base transcript. Model22 may donate a closely aligned token, or individual
high-confidence characters inside one, only through observable text and CTC
confidence rules.

Current result (fold 0 only)
----------------------------
The completed checkpoint_07 Model22 probability export covers all 819 fold-0
rows and is independently authenticated by its inference manifest and exact
evaluation receipt.

  Model22 checkpoint_07 raw       0.813019600
  Model25 baseline                0.817450092
  Model34 inner-cross-fitted      0.819147344
  gain over Model25              +0.001697252

Model34 improves both exact metric components on this rehearsal:

  weighted WER  3.160280710 -> 3.129696760
  weighted CER  5.595869995 -> 5.549348685

It changes 180 of 819 rows. A deterministic 5,000-sample paired row bootstrap
has a 95% percentile interval of [+0.000031417, +0.003350088], with a positive
gain in 97.68% of resamples. This is promising, but it is still only one outer
fold: it is not yet a five-fold OOF score and is not a submission candidate.

The Model34 fusion rule is cross-fitted, but the supplied Model22 epoch-7
artifact came from an earlier ten-checkpoint screen on fold 0. Consequently,
this is not a fully nested end-to-end model-selection estimate. Freeze epoch 7
before producing folds 1-4 and use those new folds for the unbiased confirmation.

Leakage control
---------------
Fold 0 is partitioned into five deterministic hash groups. For each held-out
group, its rule is selected using only the other four groups' references. The
held-out group's targets are used only once, for evaluation after selection.
The reported 0.819147344 concatenates these five held-out predictions.

Model34's inner groups are not the Model33 groups. Model34 uses direct SHA-256
modulo assignment with salt model34-token-fusion-v1; Model33 uses balanced
hash-rank assignment with salt model33-selector-v1:single-fold:0. Do not feed
this crossfit CSV into a fresh Model33 single-fold crossfit and describe the
composition as fully nested. Although each row's own label is excluded from
its Model34 prediction, Model34 can use labels of its peers in a different
Model33 held-out group. A strict stack needs pair-excluded nested Model34
artifacts per Model33 split, or a Model34 rule frozen on independent outer-fold
data.

The finite grid starts with a byte-for-byte no-op rule. Selection maximizes the
worst gain across the available training groups, then the aggregate competition
metric, then fewer changed rows. A learned rule must improve every available
training group by at least 0.00025 or the system fails closed to Model25.

Inference behavior
------------------
Words are aligned with a deterministic minimum-cost dynamic program. Candidate
rules gate Model22 donations using:

  * minimum, q10, or mean per-character confidence;
  * token edit-distance ratio;
  * whole-row disagreement ratio;
  * exact neighboring-token context;
  * token-level or character-level donation.

Model25 retains case whenever spellings agree case-insensitively. Carets and
ampersands already restored by Models24/23 cannot be removed. A Model22 plus
sign is interpreted as the already-validated Model23 ampersand normalization.
Word insertions/deletions are not donated, which keeps the first version
conservative.

Provenance and failure behavior
-------------------------------
model34/config.json pins SHA-256 hashes for the truth table, Model25 OOF,
Model22 JSONL/CSV, inference manifest, and evaluation receipt. The receipt must
authenticate the JSONL, CSV, manifest, truth hash, outer fold, and checkpoint.
All IDs and companion prediction strings must agree exactly. Any missing,
extra, duplicate, changed, or unauthenticated artifact raises an error; the
tool never silently joins partial rows.

Reproduce the fold-0 rehearsal
------------------------------
  PYTHONPATH=src .venv-kraken/bin/python model34/crossfit.py \
    --config model34/config.json \
    --output outputs/model34/fold0_crossfit.csv \
    --report outputs/model34/fold0_report.json \
    --deployment-config outputs/model34/deployment_rule.json

Independent exact-score check
-----------------------------
  PYTHONPATH=src .venv-kraken/bin/python scripts/score_csv.py \
    --truth data/processed/train_folds.csv \
    --predictions outputs/model34/fold0_crossfit.csv \
    --fold 0

Tests
-----
  PYTHONPATH=src .venv-kraken/bin/python -m unittest \
    tests.test_model34_token_fusion -v

Deployment (not ready yet)
--------------------------
outputs/model34/deployment_rule.json is a provisional rule fitted on all
fold-0 rehearsal labels. It is deliberately not used in the reported fold-0
score. Before considering one final Zindi submission:

  1. Produce held-out Model22 checkpoint-fixed JSONL predictions for folds 1-4.
  2. Run the same outer-fold/inner-cross-fit audit on all five folds.
  3. Freeze one rule from the complete five-fold OOF evidence.
  4. Generate matching Model22 test confidences and apply that frozen rule to
     Model25 test predictions with both input hashes supplied explicitly.

The apply command refuses unpinned inputs:

  PYTHONPATH=src .venv-kraken/bin/python model34/apply.py \
    --baseline PATH_TO_MODEL25_TEST.csv \
    --fine PATH_TO_MODEL22_TEST.jsonl \
    --deployment-config PATH_TO_FINAL_RULE.json \
    --expected-baseline-sha256 SHA256 \
    --expected-fine-sha256 SHA256 \
    --output outputs/model34/test.csv

Audited artifacts
-----------------
  outputs/model34/fold0_crossfit.csv
    SHA-256 954006dd329ea775b4c972dd040dbfd9da2598e962b6438eced6c78879fb60e6

  outputs/model34/fold0_report.json
    Contains every split's selected rule, train/held-out scores, exact lineage,
    bootstrap audit, and the provisional all-fold-0 deployment rule.
