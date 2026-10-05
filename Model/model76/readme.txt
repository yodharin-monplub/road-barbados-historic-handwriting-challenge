Model 76: signed five-fold Model73 OOF audit
================================================

This model combines the five independently generated Model73 heldout CSVs in
canonical training-row order and computes the exact competition metric. It
fails closed unless every CSV matches its inference manifest, signed Model74
receipt, truth checksum, fold number, and exact heldout ID set.

Run on the training host after folds 0-4 finish:

  PYTHONPATH=src .venv-lighton/bin/python model76/evaluate_oof.py \
    --truth data/processed/train_folds.csv --artifact-root outputs \
    --output outputs/model76/model73_step400_oof.csv \
    --receipt outputs/model76/model73_step400_oof.evaluation.json

This is evaluation only. It never submits to Zindi and never uses test labels.
