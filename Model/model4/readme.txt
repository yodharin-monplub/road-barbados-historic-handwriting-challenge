MODEL 4 — custom CRNN-CTC + train-only character language model

Purpose
-------
This model is deliberately independent of autoregressive VLMs. It keeps only
4x horizontal downsampling, so a 2,048-pixel line has up to 512 CTC timesteps.
That matters for the unusually wide crops. A three-layer BiLSTM models writing
sequence context, and a character 5-gram model trained only on the permitted
training labels rescores prefix-beam candidates.

The optional LineMix augmentation concatenates two provided training images and
their labels. Every pixel and character comes from Train.csv; Zindi explicitly
confirmed fully automated recombination of train crops is permitted. Validation
rows are never mixed into a fold's training data or language model.

Hardware
--------
One 8–12 GB CUDA GPU should suffice. This is the cheapest candidate to run over
all five folds. CPU training works but will be slow.

Commands
--------
python model4/train.py --fold 0

python model4/infer.py \
  --checkpoint outputs/model4/fold0 \
  --input data/processed/train_folds.csv \
  --fold 0 \
  --output outputs/model4/oof_fold0.csv

Change decoder.mode to greedy for a quick baseline. Tune lm_alpha and
space_bonus on combined OOF predictions, not one lucky fold. Then retrain all
data with validation_fold: null and output_dir: outputs/model4/all.

Why it may help the ensemble
----------------------------
CTC tends to omit uncertain strokes instead of inventing fluent words. VLMs do
the opposite. Validation-tuned consensus can exploit that disagreement, and
model7 retains eight beam candidates per image in the JSONL output.

Status
------
Architecture, deterministic training, greedy validation, prefix beam search,
character LM, manifests, and submission inference are implemented. The data and
folds are ready; GPU training has not been run on this CPU-only workstation.
