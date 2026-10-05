MODEL 22 — GPU-safe PP-OCRv6 domain fine-tuning

Purpose
-------
Fine-tune the approved historical PP-OCRv6 medium checkpoint on the Barbados
training transcriptions. This attacks the recognizer ceiling exposed by the
0.814499013 public score of Model 19 rather than adding more post-processing.

Why this differs from Model 8
-----------------------------
The original training settings exceeded the 6 GB RTX 4050 memory budget before
the first batch. Model 22 uses the checkpoint's native 96-pixel line height,
batch size 1, gradient accumulation 8, and a 3584-pixel training width cap.
All Model22-eligible lines fit both the width cap and the PP-OCRv6 CTC time
axis after height normalization. A deterministic preparation preflight checks
height 96, horizontal padding 16, and width subsampling 8 and aborts with IDs
and reasons instead of allowing Kraken to silently replace an invalid sample.
Codec resize mode "new" replaces the pretrained model's very large multilingual
decoder with the competition alphabet, saving substantial memory and compute.
Inference leaves the older external charset mask disabled: a fold's compact
codec can legitimately omit a character present only in validation.

Model22 data policy
-------------------
The canonical ``data/processed/train_folds.csv`` remains unchanged at 4,095
rows (819 per fold). ``model22/exclude_ids.txt`` is deliberately local to this
model and contains six IDs:

* Three confirmed unusable image/transcription pairs from the original CSV.
  They are already absent from the 4,095-row clean folds.
* Three otherwise valid rows whose targets cannot CTC-align at Model22's input
  geometry: ``EcxuqKeZl7OQexfB`` (fold 2), ``gP7kGWfDFYMbZ8nO`` (fold 3), and
  ``PO7QQLWIFOT65BTz`` (fold 4).

Consequently the all-data Kraken training manifest has 4,092 rows. Fold 0 and
fold 1 have 3,273 train / 819 validation rows; folds 2, 3, and 4 have 3,274
train / 818 validation rows. Generated ``.gt.txt`` labels strip leading and
trailing whitespace only; internal spacing remains unchanged. Model22
inference uses the matching opt-in cleanup: it strips prediction edges and
normalizes CR/LF line breaks but preserves internal spaces. Legacy Kraken
models keep their existing whitespace-collapsing default.

Training
--------
Fold 0 is the first honest development fold:

  .venv-kraken/bin/python model22/train.py --fold 0

The run is fixed at 10 epochs with validation after every epoch. Checkpoints
and the exact command/config manifest are written under outputs/model22/fold0.
The supplied CSV and raw images are never edited.

Validation inference
--------------------
Kraken chooses its automatic best checkpoint by ordinary micro CER, while the
challenge combines length-weighted WER and CER. After a clean training finish,
convert, infer, and score every saved epoch with the exact objective:

  .venv-kraken/bin/python model22/evaluate_checkpoints.py \
    --checkpoint-dir outputs/model22/fold0 \
    --input data/processed/train_folds.csv \
    --output-dir outputs/model22/checkpoint_evaluation/fold0 \
    --fold 0 --batch-size 1

The helper processes one numerically ordered checkpoint per conversion,
resumes only SHA-validated weights/predictions/sidecars, and writes
epoch-ordered leaderboard.json and leaderboard.csv files. It pins the exact
4,095-row canonical CSV (819 rows per fold), checks the current inference
settings against the training run_manifest, and permits only training-only or
explicitly CLI-overridden config differences. It refuses a still-changing or
mixed-run checkpoint directory by default and requires the evaluation output
to be a disjoint sibling tree, as in the command above. Do not use
``--allow-incomplete-training`` for ordinary checkpoint selection. Treat the
supplied raw image directory as immutable, and run only one evaluator process
per output directory.

For one already converted checkpoint, the equivalent manual command is:

  .venv-kraken/bin/python model22/infer.py \
    --checkpoint outputs/model22/fold0/<checkpoint>.safetensors \
    --input data/processed/train_folds.csv \
    --fold 0 \
    --output outputs/model22/fold0_predictions.csv

  PYTHONPATH=src python scripts/score_csv.py \
    --truth data/processed/train_folds.csv \
    --predictions outputs/model22/fold0_predictions.csv \
    --fold 0

External checkpoint evaluation always reads the canonical fold CSV, not the
Kraken training/validation manifests. It therefore predicts and scores all
819 OOF rows for every fold, including the one CTC-incompatible training row
omitted from each of folds 2, 3, and 4. The exact scorer rejects any missing or
extra validation ID.

Decision rule
-------------
Do not create a submission from this model until domain-fine-tuned OOF beats
the corresponding frozen-model and Model 19 fold scores. If it does, train all
five folds for OOF validation, then train on all clean rows with the epoch count
selected without leaderboard feedback.

The final ``--all-data`` run keeps all 4,092 Model22-eligible rows in its
training manifest. A deterministic 128-row subset is also listed in an overlapping
evaluation manifest so Kraken does not make an internal 90/10 holdout and
continues to apply training augmentation. That evaluation metric is in-sample
plumbing only and must not be used to select hyperparameters or epochs; use OOF
results for selection.

Status
------
Fold-0 training completed cleanly and every saved epoch was independently
converted, inferred on all 819 canonical held-out rows, and scored. The exact
challenge metric—not Kraken's filename proxy—selects zero-based epoch 7:

  multi-score   0.813019600
  weighted WER  3.410939
  weighted CER  4.934375

The frozen Model8 checkpoint scored 0.731393742 on the same fold. Model25's
post-processed prior pipeline remains slightly higher at 0.817450092, although
the two systems make complementary errors. Epoch 9, despite the best Kraken
proxy, scored only 0.801538687. `outputs/model22/checkpoint_evaluation/fold0/
leaderboard.json` records all ten exact results. Model36 freezes eight completed
epochs for the remaining folds; Model22 itself does not yet claim five-fold OOF.
