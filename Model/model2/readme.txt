MODEL 2 — LightOnOCR-2-1B-base LoRA domain adaptation

Purpose
-------
Fine-tune the Apache-2.0 LightOnOCR base checkpoint on the provided Barbados
labels. This should adapt spelling, period vocabulary, ink degradation, and the
central-line annotation convention while retaining a strong OCR prior. The
checkpoint revision is immutable.

The configuration uses BF16 LoRA rather than 4-bit quantization because a 1B
model normally fits on a 16–24 GB GPU and the small dataset benefits from clean
gradients. The initial crop is the center 74% of image height. Compare it with
full images via cross-validation before training all data.

Setup
-----
Create a Python 3.11/3.12 environment, install model2/requirements.txt, then
install this repository with "pip install -e .". Prepare the five folds first.

Train and validate fold 0
-------------------------
python model2/train.py --fold 0

python model2/infer.py \
  --adapter outputs/model2/fold0/best_adapter \
  --input data/processed/train_folds.csv \
  --fold 0 \
  --output outputs/model2/oof_fold0.csv

python scripts/score_csv.py \
  --truth data/processed/train_folds.csv \
  --predictions outputs/model2/oof_fold0.csv \
  --fold 0

Run folds 0–4 for reliable OOF estimates. Do not select the epoch solely from
training loss—generate and score validation text.

Final fit
---------
After hyperparameters are frozen, set validation_fold to null and output_dir to
outputs/model2/all in a copied final config, then train on all eligible rows.
Generate Test.csv predictions using best_adapter and the model1 command shape.

Status
------
Training/inference code and pinned configuration are ready. Three visually
confirmed unusable rows are excluded programmatically and logged; the other 18
community-reported suspects were inspected and retained as valid signal.
