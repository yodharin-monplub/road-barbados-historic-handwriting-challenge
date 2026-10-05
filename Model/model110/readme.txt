Model110: new training-derived passages rendered into synthetic handwriting
==========================================================================

Purpose: carry the user's paragraph-synthesis idea through to OCR fine-tuning
and real heldout evaluation. Models108/109 were generator pilots only.

Use pretrained DiffusionPen (the pinned Model108 assets); do not use Model109's
degraded generator adapter. Construct new passages using a trigram text model
with bigram fallback,
fit only to competition training folds 1-4. No external paragraphs/dictionaries.
Every historical token and transition retains its training-source provenance.
Render the 384 most frequent supported words in four training-derived styles,
then lay out generated word images as 12-word lines in three-line paragraphs.
Every line uses one consistent style-reference set. No real word pixels enter
the synthetic lines; real images are used only as generator style references.

Filter generated words when at least one frozen competition-fine-tuned OCR
checker reproduces the requested word exactly. This is a noisy automatic filter,
NOT verified ground truth. The prior paired control found joint agreement on
only 56/200 real automatic crops, versus 20/200 Model108 generated words; both-
checker exact agreement was therefore too insensitive to serve as a verdict.
The union rule is fixed before this experiment's OCR CV is seen.

Train a separate LightOnOCR LoRA adapter from the same base as Model92 using
3276 real lines plus 819 synthesized lines (80/20 rows), same seed/effective
batch, mild augmentation and 400 optimizer steps. Validation has only the 819
real fold-0 images. No generator, text model, source/style crop or quality
checker is fit using fold 0. New paragraph/line labels are never modernized.

Fold-0 gain >=0.001 versus Model92 is a development screen before expanding to
the other four folds. A submission requires full-OOF confirmation, not synthetic
word acceptance or training loss. No score or submission exists at creation.

The recombined passages are not guaranteed to be coherent prose. This tests
training-only linguistic recombination plus generated handwriting, not whether
an external language model can invent historically accurate paragraphs.

Pipeline artifacts:
outputs/model110/fold0/words: generator requests/images and OCR checks.
outputs/model110/fold0/training: line images, labels.csv, passages.txt, manifest.
outputs/model110/config_fold0.yaml: fixed OCR configuration.
outputs/model110/fold0_oof.csv and fold0_report.json: pending real validation.

Execution on 2026-09-07:
- Local RTX4050 CUDA generator: model110/generate_words.py, resumable word bank.
- road-model110-preparation.service runs scripts/prepare_model110.py; waits for
  the generator, checks words, composes passages and validates the training set.
- road-model110-ocr-ablation.service runs scripts/run_model110.py. It checks
  every 600 seconds and waits until preparation is ready AND Model107 finishes
  its confirmation/conditional deployment. It never takes an active instance.
- Only then use the existing authorized Vast instance 50076619 for fold-0
  training and real-image validation. Download predictions and adapter, verify
  hashes/provenance, STOP (not destroy) the instance, and score locally.
- No automatic submission or further fold training is enabled for Model110.
Status files: outputs/model110/preparation_status.json and run_status.json.
Targeted verification: 24 augmentation/provenance tests plus 3 VLM CLI tests
passed before cloud launch. No measured Model110 competition score yet.
