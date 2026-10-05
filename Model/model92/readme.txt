Model92: mild training-time document augmentation
=================================================

Model92 tests whether small, label-preserving visual perturbations improve the
LightOnOCR Model73 recipe. It changes only the training images; validation and
inference remain the original full image. Augmentations use only the supplied
competition images: brightness +/-8%, contrast +/-12%, rotation +/-0.5 degree,
vertical translation +/-1.5%, and occasional Gaussian blur up to radius 0.4.

The first gate is canonical fold 0, trained from the same pinned base, LoRA,
effective batch size, seed, optimizer, and learning rate as Model73. Training
is fixed to 400 optimizer steps because Model73's best validated checkpoint was
step 400. Promote only if exact fold-0 OCR scoring improves meaningfully.

GPU commands:

  PYTHONPATH=src .venv-lighton/bin/python model92/train.py --fold 0

  PYTHONPATH=src .venv-lighton/bin/python model92/infer.py \
    --adapter outputs/model92/fold0/best_adapter \
    --input data/processed/train_folds.csv --fold 0 \
    --output outputs/model92/fold0_oof.csv

No external dataset is used and nothing is submitted automatically.

Fold-0 result
-------------

The authorized Vast.ai RTX 4090 run completed all 400 optimizer steps and
decoded every one of the 819 canonical fold-0 rows.  Exact competition scores
for the preserved checkpoints were:

  step 100: 0.8606481022
  step 200: 0.8780014993
  step 300: 0.8908166962
  step 400: 0.8924811370

The fixed Model73 checkpoint-400 baseline on the same rows is 0.8906962580,
so Model92 step 400 gains +0.0017848790 and passes the promotion gate.  Step
400 is frozen before inspecting folds 1--4.  Those confirmation folds must use
their own fold-excluded training runs and checkpoint 400 regardless of their
individual validation losses.
