Model 79: Model73 + corrected Model35 cross-fitted selector
==========================================================

Model79 reuses the already-frozen, dependency-free Model33 logistic selector
without hyperparameter retuning. Candidate A is the signed five-fold Model73
checkpoint-400 OOF; candidate B is Model35's five-fold corrected confusions
OOF. Prediction-time features contain only the two strings, their disagreement
and shape, and target-language scores learned outside the evaluated fold.

Exact five-fold result
----------------------

  Model73 primary:             0.8894664157724265
  corrected Model35 alternate: 0.8417216298681414
  Model79 selector:            0.8928218510472717
  delta versus Model73:       +0.0033554352748453
  selected Model35 rows:       146 / 4095

The selector improves the primary on every canonical fold. Fold scores are
0.8919363193, 0.8956454153, 0.8931315064, 0.8913564456, and 0.8920420596.
Each evaluated row is excluded from its selector and target-language-model fit;
training-row language features are nested-cross-fitted.

A 5,000-sample paired row bootstrap gives a conditional 95% interval of
[+0.0012241571, +0.0061954283] for Model79 minus Model73, with a positive gain
in every resample. This interval does not cover training randomness or test
distribution shift.

Reproduce OOF:

  PYTHONPATH=src python3 model79/evaluate.py \
    --truth data/processed/train_folds.csv \
    --candidate-a outputs/model76/model73_step400_oof.csv \
    --candidate-b outputs/model35/allfold_epoch07_verified/predictions/03_confusions_oof.csv \
    --config model79/config.json --output-dir outputs/model79/model73_vs_model35

After Model77 all-data test inference, fit and apply the deployment selector:

  PYTHONPATH=src python3 model79/fit.py \
    --truth data/processed/train_folds.csv \
    --candidate-a outputs/model76/model73_step400_oof.csv \
    --candidate-b outputs/model35/allfold_epoch07_verified/predictions/03_confusions_oof.csv \
    --config model79/config.json --output-model outputs/model79/deployment.json

  PYTHONPATH=src python3 model79/apply.py --input data/raw/Test.csv \
    --candidate-a outputs/model77/test.csv \
    --candidate-b outputs/model35/deployment_epoch07/confusions_test.csv \
    --model outputs/model79/deployment.json --output outputs/model79/test.csv \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model79_selector.csv

Model79 never uploads or submits automatically.
