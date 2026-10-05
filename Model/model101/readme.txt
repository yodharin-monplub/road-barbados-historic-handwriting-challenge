Model101: guarded Model88/Model100 selector
===========================================

Model101 tests the already-frozen leakage-safe selector between the previous
public candidate Model88 and the new character-consensus Model100. Every OOF
row is excluded from selector fitting and target-language feature fitting.
No configuration is tuned for this pair, and no GPU is required.

Run:

  PYTHONPATH=src python3 model101/evaluate.py \
    --truth data/processed/train_folds.csv \
    --candidate-a outputs/model88/word_rover_vs_kraken_selector/crossfit_predictions.csv \
    --candidate-b outputs/model100/character_rover_vs_kraken/crossfit_predictions.csv \
    --config model101/config.json \
    --output-dir outputs/model101/model88_vs_model100

OOF result
----------

  Model88:  0.8990278333490157
  Model100: 0.8996529328792820
  Model101: 0.8998601038879951

Model101 gains +0.0002071710 over Model100 and +0.0008322705 over Model88.
It improves all five canonical folds. A seeded 5,000-sample paired row
bootstrap gives an incremental 95% interval over Model100 of
[+0.0000433843, +0.0003818356], with 99.42% positive resamples. This interval
is conditional on saved predictions and excludes training randomness,
correlation, model-selection effects, and test distribution shift.

Candidate
---------

  submissions/model101_guarded_character_rover.csv
  SHA-256 7ae8d9ad4f895f7db2fee72826a14acb18578fcd9f1661d7d0fae35998dd9cbf

The deployed selector keeps Model100 on 1,281 test rows and Model88 on 93; the
result differs from Model100 on 25 rows. It is schema-validated against the
official sample submission. Nothing is submitted automatically.
