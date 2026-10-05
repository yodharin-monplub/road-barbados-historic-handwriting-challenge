MODEL 8 — approved PP-OCRv6 medium historical HTR (Kraken 7.1)

Purpose
-------
This is the strongest immediately runnable historical-handwriting baseline in
the repository. It uses the PP-OCRv6 medium model published at Zenodo record
21788410. Zindi explicitly approved this exact model for this challenge. The
weights are Apache-2.0, trained on historical handwritten and printed material,
and expect 128-pixel-high line images without a narrow fixed-width crop.

Pinned artifact
---------------
DOI: https://doi.org/10.5281/zenodo.21788410
File: medium.safetensors
MD5: e3411a453ce3b9e9efae3f8631d85762
SHA256: 15313b51ace64cbfa81f8f6ef25ad64f04e5a6fb7f7823e67b107527bc081ac9

The checked artifact is available at checkpoints/model8/medium.safetensors.
To reproduce the download in an activated Kraken environment:

  XDG_DATA_HOME="$PWD/checkpoints/model8/registry" kraken get 10.5281/zenodo.21788410

Environment
-----------
Kraken 7.1 requires Python 3.10–3.13. This workstation's system Python is 3.14,
so a separate Python 3.12 environment is already installed at .venv-kraken.

  source .venv-kraken/bin/activate

Zero-shot OOF benchmark
-----------------------
Run the untouched approved model on fold 0 before training:

  python model8/infer.py \
    --input data/processed/train_folds.csv \
    --fold 0 \
    --no-constrain-charset \
    --output outputs/model8/base_fold0_full.csv

  PYTHONPATH=src python scripts/score_csv.py \
    --truth data/processed/train_folds.csv \
    --predictions outputs/model8/base_fold0_full.csv \
    --fold 0

Measured on the 819 clean rows in fold 0 (CPU, full-image variant):

  weighted WER: 4.8273724577
  weighted CER: 7.4887361959
  normalized multi-score: 0.7307800610

This is a useful baseline, not a claim about the hidden test score. The aspect
breakdown is saved in reports/model8_base_fold0_full.json. That first benchmark
used the pretrained decoder unchanged; the configured final decoder also masks
characters absent from Train.csv's known alphabet and is validated separately.

With the alphabet constraint enabled, the fixed full view scores 0.731393742;
the fixed center view scores 0.727728669. The center output remains useful as a
confidence-selected candidate in model9 even though it is weaker by itself.

Fine-tuning
-----------
Use one CUDA GPU. The loader preserves line aspect ratio at height 128 and caps
only at width 6144, above the observed normalized maximum. It writes generated
Kraken manifests and .gt.txt files under data/processed; raw inputs are never
modified. The union codec keeps every supplied target character encodable in
each fold. Fold 0 is the default:

  python model8/train.py --fold 0

Repeat folds 1–4 for honest OOF estimates. After choosing the epoch count from
OOF logs, set training.all_data_epochs and train one final model:

  python model8/train.py --all-data

The all-data command keeps every eligible row in the training manifest and
also supplies a deterministic overlapping evaluation subset. The overlap is
required to prevent Kraken from making its own 90/10 holdout and to keep
augmentation enabled. Its in-sample evaluation metric is runtime plumbing,
not a valid basis for checkpoint or hyperparameter selection; freeze the final
epoch count from OOF results first.

Kraken stores the best safetensors checkpoints inside the configured output
directory. Pass the selected file explicitly to inference:

  python model8/infer.py \
    --checkpoint outputs/model8/foldall/<selected>.safetensors \
    --input data/raw/Test.csv \
    --output outputs/model8/test.csv \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model8.csv

Validation strategy
-------------------
Compare full, center, enhanced, and center_enhanced variants only on OOF data.
Do not choose a transform from leaderboard feedback. Keep this CTC-family model
in the model7 consensus because its errors differ from the autoregressive VLMs.

Status
------
The environment, approved weights, exact checksums, deterministic manifest
builder, fine-tuning wrapper, confidence logging, inference, and submission
validation are implemented. Full-image fold-0 inference and scoring are
complete, as are center-band and constrained decoding validation. Full-view
and center-view test inference are complete, and both raw outputs include
confidence logs for model9. The full-view and two-view Model14 initial
submissions pass schema and empty-prediction validation.
