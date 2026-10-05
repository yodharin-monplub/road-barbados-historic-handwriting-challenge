Model48 - organizer-aligned rank-16 language QLoRA at mid resolution
====================================================================

Status
------

Model48 is implemented and its complete CPU audit passes. It has not used the
GPU, has not passed the mandatory real CUDA smoke, has not been trained, and has
no local OOF or Zindi score. It is a prepared fallback for Model43 if Model43's
1024-token image profile exceeds the 5.3 GiB gate or proves unnecessarily slow.

Controlled hypothesis
---------------------

Model48 keeps Model43's organizer-aligned learning recipe unchanged:

* Qwen/Qwen2-VL-2B-Instruct at the pinned Apache-2.0 checkpoint revision;
* rank-16, alpha-32 LoRA on q/k/v/o and gate/up/down in every language layer;
* all 32 visual blocks and the visual merger frozen;
* the organizer starter's generic exact-transcription prompt, verbatim;
* three complete epochs, learning rate 3e-5, batch size 1, accumulation 16;
* NF4 double quantization, float16 compute, SDPA, gradient checkpointing, and
  paged 8-bit AdamW;
* greedy full-fold validation with max_new_tokens=48, repetition penalty 1.05,
  and no-repeat 4-grams.

The only experimental change is the factor-aligned visual interval: 512..640
merged image tokens instead of Model43's 896..1024. The 640-token ceiling gives
twice Model39's 320-token visual budget while using 62.5% of Model43's maximum.
As a simple visual-token cap-squared proxy, that is 39.0625% of Model43's
quadratic workload. Actual CUDA savings must still be measured by the smoke.

Model43 provenance isolation
----------------------------

Model43 was already training when Model48 was created. Its source, config, and
output were therefore left byte-for-byte untouched. Model48 owns a separate
module, config, wrappers, paths, model-number fields, manifests, and tests. It
imports Model43's audited pure architecture, resize, collator, and scheduling
utilities read-only, but it does not call Model43's top-level train function.
The Model48 manifest is written as model_number=48 before optimization starts.

Exact adapter ledger
--------------------

The pinned checkpoint exposes 28 language layers and seven selected matrices
per layer, exactly 196 LoRA targets. There are no visual or merger targets.
Rank * (in_features + out_features), evaluated for every safetensor header,
gives:

  target modules:                  196
  LoRA tensors:                    392
  trainable parameters:     18,464,768
  targeted base parameters: 1,310,195,712

Both checkpoint-header and live module-graph audits must agree on every path and
dimension. Any extra trainable tensor, visual target, target-count drift, or
parameter-count drift aborts.

Aspect-preserving 640-token image policy
----------------------------------------

Every complete crop is uniformly resized with Lanczos into an aspect-matched
letterbox. Canvas dimensions are multiples of Qwen's 28-pixel merged patch
factor. There is no cropping, fragment concatenation, or glyph stretching.
The background is the median corner colour. The grid maximizes real content
area in the fixed 512..640 interval and never exceeds 501,760 input pixels.

The audited geometry over all 5,472 images is:

  effective tokens: minimum 520, median 636, maximum 640
  content fraction: minimum 0.8071428571, median 0.9761904762, maximum 1.0
  aspect error:      maximum 0.0043259118
  resize direction: 4,094 upscale, 1,378 downscale, 0 unchanged

No crop is lost. The smallest realized grid is 520 rather than 512 because the
best aspect-matched integer factor grid for each observed crop happens to land
above the configured lower bound.

Sequence and context receipts
-----------------------------

The pinned tokenizer was run against all 4,095 labels. Conservatively assigning
the full 640-image-token ceiling to every row gives:

  longest training template: 731 tokens (ID SXNZ5kUCcAxRUpCN)
  longest target:             34 tokenizer tokens
  generation prompt:          695 tokens
  prompt + 48 generation:     743 tokens
  configured train/gen caps:  768 / 816 tokens
  pinned model context:        32,768 tokens

The actual highest planned workload was selected using each crop's realized
grid and encoded through the real pinned processor. It produced 640 image
tokens, 728 total tokens, 695 prompt tokens, and 33 supervised assistant tokens
for ID WwlTCykxjP3c4kfo. Nothing is truncated.

Fold and optimizer schedule
---------------------------

Each of the five deterministic outer folds contains 3,276 train and 819 held-out
rows with disjoint IDs. The exact ceil schedule retains the final partial
accumulation group:

  204 full 16-row groups + one 12-row group = 205 updates per epoch
  205 updates * 3 epochs = 615 optimizer updates
  3,276 rows * 3 epochs = 9,828 row presentations

Callbacks permit validation only at complete boundaries 205, 410, and 615 and
fail if Trainer uses its usual floor-derived 612-step schedule.

CPU audit and tests
-------------------

From the repository root:

  PYTHONPATH=src .venv-vlm/bin/python model48/audit.py
  PYTHONPATH=src .venv-vlm/bin/python -m unittest \
    tests.test_model48_qwen_starter_midres -v

The machine-readable receipt is outputs/model48/dry_run.json. It verifies the
local checkpoint and organizer-starter hashes, dependency versions, exact target
headers, image geometry, real processor encoding, context, folds, and schedule.

Mandatory GPU smoke
-------------------

Run this only after the current GPU job is finished:

  PYTHONPATH=src .venv-vlm/bin/python model48/train.py --fold 0 --smoke-only

The report will be outputs/model48/smoke/fold0.json. It must show 196 injected
targets, 18,464,768 trainable parameters, 392 LoRA tensors, NF4 modules, full
single-device CUDA residency, and both peak allocated and peak reserved memory
at or below 5.3 GiB. Do not raise or bypass the threshold.

Training and inference after smoke
----------------------------------

Train the first challenger fold:

  PYTHONPATH=src .venv-vlm/bin/python model48/train.py --fold 0

Infer held fold 0 from the chosen adapter:

  PYTHONPATH=src .venv-vlm/bin/python model48/infer.py \
    --input data/processed/train_folds.csv \
    --fold 0 \
    --adapter outputs/model48/fold0/best_adapter \
    --output outputs/model48/fold0/oof.csv

All 819 held rows are decoded and scored after each complete epoch. A single
fold is a conditional model-selection result, not unbiased five-fold OOF.

Decision rule and risks
-----------------------

Use Model48 only if Model43 fails its memory gate, its runtime is prohibitive,
or an honest fold-0 comparison shows no benefit from 1024 tokens. Mid resolution
may miss fine faded strokes that Model43 preserves. Conversely it may reduce
interpolation blur and paper-noise magnification, while the smaller sequence can
train faster and leave safer allocator headroom. The frozen visual tower also
limits domain adaptation. Only complete held-fold transcription metrics—not
memory estimates or parameter counts—can decide whether this becomes a
submission candidate.
