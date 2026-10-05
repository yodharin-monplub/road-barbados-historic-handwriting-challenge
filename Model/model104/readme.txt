Model104: Model102 versus augmented LightOnOCR Model92
======================================================

Model104 tests whether Model92's new augmentation-trained OCR errors are
complementary to the current Model102 ensemble.  Candidate A is Model102 and
candidate B is the fixed Model92 checkpoint-400 prediction.  It uses the
unchanged Model101 text-only metric-weighted logistic selector.

The development screen uses five-way inner cross-validation within canonical
fold 0.  Every scored row is excluded from both its selector fit and its target
language-model fit.  Step 400 was already frozen by Model92 before folds 1--4.
This fold-0 result is development evidence only; Model104 cannot promote until
the same fixed recipe is confirmed using complete fold-excluded predictions.

Run the development screen:

  PYTHONPATH=src python3 model104/evaluate.py \
    --truth data/processed/train_folds.csv \
    --candidate-a outputs/model102/model101_vs_model89/crossfit_predictions.csv \
    --candidate-b outputs/model92/fold0_oof_step400.csv \
    --config model104/config.json \
    --output-dir outputs/model104/fold0_screen \
    --single-fold 0

No test prediction or submission is created automatically.

After all five held-fold decodes are copied into `outputs/model92` using the
names `fold0_oof_step400.csv` through `fold4_oof_step400.csv`, assemble and
score them with:

  PYTHONPATH=src python3 model104/assemble_model92_oof.py \
    --truth data/processed/train_folds.csv \
    --input-dir outputs/model92 \
    --output outputs/model104/model92_step400_oof.csv \
    --report outputs/model104/model92_step400_oof.report.json

Development result
------------------

The five-way inner-cross-fitted fold-0 screen selected Model92 on 416/819 rows
and scored 0.9011748664.  Model102 scored 0.9000633025 on the same rows, so the
selector gained +0.0011115639.  Raw Model92 scored 0.8924811370.  The positive
screen confirms that Model92 contributes complementary signal, but fold 0 was
already Model92's checkpoint-selection fold.  Promotion therefore remains
conditional on the untouched folds 1--4.

Full OOF result
---------------

All five Model92 held-fold checkpoint-400 artifacts passed SHA-256, row-count,
heldout-ID, and blank-output checks.  Raw Model92 scores 0.8914273974 across
4,095 rows, versus 0.8894664158 for the matching Model73 recipe.

The unchanged cross-fitted selector scores 0.9014774424 versus 0.9000848760
for Model102, a gain of +0.0013925664.  Fold gains are +0.0021071190,
+0.0015481335, +0.0010309374, -0.0002892676, and +0.0025638059.  A seeded
5,000-sample paired row bootstrap gives a 95% interval of
[+0.0002956659, +0.0024747605], positive in 99.24% of resamples.  This remains
conditional on the saved predictions and does not include training randomness,
record correlation, model-selection effects, or test distribution shift.

Deployment completed (2026-09-07)
--------------------------------
The frozen selector was applied to Model102 and the Model105 all-data refit.
It chose Model105 on 705 of 1,374 test rows. One blank Model105 transcript used
Model102 as a fallback. The final file passed schema, ID, and nonblank checks:
submissions/model104_model105_selector.csv
SHA-256: 2fc2dea27a0b9741200d845fc929fb9af3670105424f346bda4239d0dabab735
No new public score is available until this file is submitted. The OOF score
above uses held-fold Model92 candidates, not the all-data Model105 predictions.

Public result supplied by the user: 0.909699454 (submission RUHb7acF).
This improves on Model102's 0.909128148 by +0.000571306.
