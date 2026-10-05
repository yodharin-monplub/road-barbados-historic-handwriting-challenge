Model27: guarded Qwen2-VL-2B QLoRA challenger
================================================

Purpose
-------
This is a narrow VLM experiment that fits the local 6 GiB GPU. It is
derived from the organizer starter's VLM choice, Qwen/Qwen2-VL-2B-Instruct,
but replaces unsafe starter defaults with a narrow, auditable five-fold QLoRA
experiment.

Measured status (2026-09-03)
----------------------------
The real fold-0 smoke passed, and epoch 1 trained on folds 1-4 without a CUDA
or numerical failure. The complete 819-row greedy validation scored only
0.106889805 (weighted WER 11.856336, weighted CER 43.900581). Outputs were
fluent period-style legal text but were poorly grounded in the image. The run
was intentionally stopped just after that complete validation (global step
207/612) instead of spending two more epochs on a decisively noncompetitive
architecture. The best epoch-1 adapter and exact predictions remain under
``outputs/model27/fold0`` for diagnosis. This model is rejected for submission;
Model39 expands both linguistic and visual adaptation.

Provenance and licence
----------------------
Base model: Qwen/Qwen2-VL-2B-Instruct
Pinned revision: 895c3a49bc3fa70a340399125c650a463535e71c
Licence: Apache-2.0
Model page: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
Licence file: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/blob/895c3a49bc3fa70a340399125c650a463535e71c/LICENSE

Organizer source: data/raw/Starters.zip::Starters/VLM/config.yaml and trainer.py.
The archive and both members have pinned SHA256 values in config.yaml. Training
refuses to start if any of those bytes differ. The implementation does not use
trust_remote_code.

The complete base snapshot is already present at
checkpoints/model27/qwen2-vl-2b-instruct. Its exact 13-file allowlist and every
SHA256 are pinned in config.yaml. Before importing the model, Model27 verifies
the file set, all hashes, and config metadata (class, 28 text layers, 1,536
hidden size, 32 vision blocks, merge size 2). Both HF_HUB_OFFLINE and
TRANSFORMERS_OFFLINE are forced, and every from_pretrained call uses the local
path plus local_files_only=true. The manifest records the model ID, upstream
revision, licence, local path, architecture, and all verified file hashes.

What is deliberately different from the starter
------------------------------------------------
The starter's clean_label collapses all whitespace and its broad suffix-only
LoRA targets are not tied to a verified architecture. Model27 strips label
edges only and preserves all interior characters/spaces. It discovers full
module paths and requires the pinned 28-layer layout before injecting rank-16
LoRA into exactly:

  * 28 language self-attention q_proj modules;
  * 28 language self-attention v_proj modules;
  * visual.merger.mlp.0 and visual.merger.mlp.2.

The vision encoder is frozen. The two merger base matrices remain frozen; only
their verified LoRA adapters train. Any architecture drift, missing/extra
target, unexpected trainable tensor, or trainable vision-backbone tensor is a
hard error. There are exactly 58 adapted modules and at most 4 million trainable
parameters. The paths are persisted as one anchored regex because PEFT 0.14 can
otherwise shorten long target lists to suffixes; the 58 modules actually
wrapped are independently checked during both training and adapter reload.

Memory contract
---------------
The base is loaded as bitsandbytes NF4 4-bit with double quantization, explicit
single-CUDA placement (never device_map="auto"), float16 compute, SDPA, k-bit
training preparation, non-reentrant gradient checkpointing, batch size 1, and
gradient accumulation 16.

Images are opened whole: there is no crop, center-band extraction, thumbnail,
or aspect-ratio distortion before Qwen's native smart resize. The processor is
locked to 3,136..250,880 pixels, equivalent to at most 320 post-merger visual
tokens, with a hard 512-token total sequence limit. Over-cap examples abort;
labels are never silently truncated.

Before training, the training example with the largest conservative estimate
of visual tokens plus label characters executes a real collated forward/backward
pass. Training starts only if CUDA is available, all model
tensors and device-map entries are CUDA-resident, at least 100 Linear4bit
modules exist, all 58 LoRA targets are present, and peak allocated CUDA memory
AND peak allocator-reserved CUDA memory are each strictly below 5.3 GiB. A
smoke failure means reduce
max_image_tokens/max_pixels together and preregister the new cap--do not bypass
the gate.

Environment
-----------
No dependency was installed while adding this model. The pinned host override
uses the already installed torch 2.13.0+cu130 and torchvision 0.28.0; it does
not force the old starter stack or a CUDA downgrade. Add the remaining pinned
packages to that isolated environment, then:

  .venv-vlm/bin/python -m pip install -r model27/requirements.txt

The dependency versions are pinned. Confirm the PyTorch wheel matches the host
driver before loading the checkpoint.

Smoke first
-----------
This uses only the verified local snapshot; network access is forcibly disabled:

  PYTHONPATH=src .venv-vlm/bin/python model27/train.py --fold 0 --smoke-only

The report is written to outputs/model27/smoke/fold0.json. Smoke-only does not
create or modify the fold training directory. Model22 paths are explicitly
protected from all Model27 writes.

Deterministic five-fold training commands
-----------------------------------------
Run one at a time on the 6 GiB GPU:

  PYTHONPATH=src .venv-vlm/bin/python model27/train.py --fold 0
  PYTHONPATH=src .venv-vlm/bin/python model27/train.py --fold 1
  PYTHONPATH=src .venv-vlm/bin/python model27/train.py --fold 2
  PYTHONPATH=src .venv-vlm/bin/python model27/train.py --fold 3
  PYTHONPATH=src .venv-vlm/bin/python model27/train.py --fold 4

Each output directory must be absent or empty. Every epoch performs greedy
generation on the complete held-out fold (do_sample=false, num_beams=1) and
scores it with road_htr.metrics.competition_error, the exact local challenge
metric. outputs/model27/foldN/best_adapter is selected only by maximum held-out
multi_score; validation CSV/JSON files preserve sample order and exact internal
spaces. GPU kernels can still have platform-level numerical differences even
with fixed seeds and deterministic PyTorch settings.

OOF inference / independent verification
----------------------------------------
The epoch callback already records predictions used to select each best model.
To regenerate a fold independently:

  PYTHONPATH=src .venv-vlm/bin/python model27/infer.py \
    --input data/processed/train_folds.csv \
    --fold 0 \
    --adapter outputs/model27/fold0/best_adapter \
    --output outputs/model27/oof/fold0.csv

Repeat with matching fold/adapter numbers. Never choose hyperparameters from a
combined OOF score and then quote the same OOF as an unbiased comparison; any
further postprocessing must be cross-fitted (Model28 provides that protocol).

Test inference and deployment
-----------------------------
Generate one complete test candidate from each independently trained fold:

  PYTHONPATH=src .venv-vlm/bin/python model27/infer.py \
    --input data/raw/Test.csv \
    --adapter outputs/model27/fold0/best_adapter \
    --output outputs/model27/test/fold0.csv

Repeat for folds 1..4. Keep the five candidates separate and use the fixed
Model26 five-fold text medoid; do not select a fold using public leaderboard
feedback. Test prediction CSVs contain exactly ID and prediction so Model26 can
consume them directly; validation/OOF CSVs additionally retain their fold
column for provenance. Repeated internal spaces are preserved through both
paths. A direct single-fold submission is supported only when its prediction
IDs exactly match SampleSubmission.csv:

  PYTHONPATH=src .venv-vlm/bin/python model27/infer.py \
    --input data/raw/Test.csv \
    --adapter outputs/model27/fold0/best_adapter \
    --output outputs/model27/test/fold0.csv \
    --sample data/raw/SampleSubmission.csv \
    --submission outputs/model27/test/fold0_submission.csv

The writer preserves internal spaces and validates every submission ID. Empty
predictions fail submission creation instead of being silently replaced.
