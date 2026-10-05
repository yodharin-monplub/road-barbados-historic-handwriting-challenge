Model 80: confidence-aware Model73 + Model35 selector
=====================================================

Model80 extends Model79 by supplying the genuine per-character Kraken
confidence sequence from the underlying Model35 recognizer. Candidate A is
Model73; candidate B is corrected Model35. The fixed Model33 selector config is
reused without retuning. For every outer fold, both the selector and target
language model exclude that fold, and selector-training language features are
nested-cross-fitted.

Exact five-fold result
----------------------

  Model73:            0.8894664157724265
  Model79:            0.8928218510472717
  Model80:            0.8936202068699509
  delta vs Model73:  +0.0041537910975245
  delta vs Model79:  +0.0007983558226792
  selected Model35:   212 / 4095 rows

Model80 improves Model73 on every fold. A paired 5,000-sample bootstrap gives
a conditional 95% interval of [+0.0002634402, +0.0015442620] for Model80 minus
Model79, positive in 99.92% of resamples. This excludes training randomness,
model-selection effects, correlated records, and test distribution shift.

The five OOF details files are combined in canonical order with
`scripts/combine_fold_outputs.py`; its fold-ID verifier rejects missing,
duplicate, or wrong-fold rows. The combined file is
`outputs/model80/model35_epoch07_details.jsonl`.

Reproduce OOF:

  PYTHONPATH=src python3 model80/evaluate.py \
    --truth data/processed/train_folds.csv \
    --candidate-a outputs/model76/model73_step400_oof.csv \
    --candidate-b outputs/model35/allfold_epoch07_verified/predictions/03_confusions_oof.csv \
    --details-b outputs/model80/model35_epoch07_details.jsonl \
    --config model80/config.json \
    --output-dir outputs/model80/model73_vs_model35_with_confidence_verified

After Model77 test inference, apply the fitted deployment selector using the
matching Model35 all-data confidence file. That file's source strings were
verified to exactly equal all 1,374 Model35 raw test predictions:

  PYTHONPATH=src python3 model80/apply.py --input data/raw/Test.csv \
    --candidate-a outputs/model77/test.csv \
    --candidate-b outputs/model35/deployment_epoch07/confusions_test.csv \
    --details-b outputs/model36/deployment_epoch07/test_predictions.jsonl \
    --model outputs/model80/deployment.json --output outputs/model80/test.csv \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model80_confidence_selector.csv

Model80 never uploads or submits automatically.
