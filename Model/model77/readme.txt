Model 77: all-data LightOnOCR deployment
=========================================

Model77 refits the promoted Model73 recipe on all 4,095 labeled lines. The
five-fold selection policy froze Model73 checkpoint 400 before this refit.
Each fold trained on 3,276 lines, so 400 optimizer steps at effective batch 16
represent 6,400 example exposures (1.9536 passes). The all-data run uses 500
optimizer steps, also 8,000 / 4,095 = 1.9536 passes. Hyperparameters, seed,
resolution, base revision and LoRA topology are otherwise unchanged.

Training:

  PYTHONUNBUFFERED=1 PYTHONPATH=src .venv-lighton/bin/python model77/train.py

Test inference and candidate creation:

  PYTHONUNBUFFERED=1 PYTHONPATH=src .venv-lighton/bin/python model77/infer.py \
    --adapter outputs/model77/all/best_adapter --input data/raw/Test.csv \
    --output outputs/model77/test.csv --sample data/raw/SampleSubmission.csv \
    --submission submissions/model77_lightonocr.csv

The CSV is a candidate only. Nothing in Model77 uploads or submits it.
