MODEL 3 — rectangular TrOCR-large handwritten fine-tuning

Purpose
-------
TrOCR gives an OCR-specific encoder-decoder signal that is architecturally
different from the causal VLMs. The original checkpoint expects 384x384 input,
but these challenge images have median aspect ratio 16.78 (maximum 40.54).
Squashing them to a square destroys letter shape. This implementation instead
letterboxes each crop to 256x1024 and explicitly interpolates ViT positional
embeddings at training and generation time.

Model revision: e68501f437cd2587ae5d68ee457964cac824ddee.
Before a prize-code handoff, archive and verify the exact Microsoft/UniLM model
terms; MODEL_LICENSES.md records this open item.

Hardware
--------
TrOCR-large is roughly 2.2 GB of weights and its rectangular patch sequence is
long. Start on a 32 GB GPU. Gradient checkpointing and batch size 1 are enabled.
If memory is tight, try 192x768 as a separate validated experiment rather than
silently changing the final configuration.

Commands
--------
python model3/train.py --fold 0

python model3/infer.py \
  --model outputs/model3/fold0/best_model \
  --input data/processed/train_folds.csv \
  --fold 0 \
  --output outputs/model3/oof_fold0.csv

Generate test predictions by replacing --input with data/raw/Test.csv and adding
--sample data/raw/SampleSubmission.csv --submission submissions/model3.csv.

Validation grid
---------------
Compare center_fraction 1.0 and 0.84, then 256x1024 versus 192x1024. Validate
the maximum target token length after the CSV audit; no target may be truncated.
Keep N-best beam output for consensus rescoring in model7.

Status
------
Implemented and syntax-tested against the audited CSV schema. Training and OOF
inference await a CUDA device; this workstation has no functional NVIDIA driver.
