Model50 - label-free adaptive-resolution Model43 successor
==========================================================

Status
------

Model50 is implemented in its own module and has passed its CPU config,
checkpoint-header, provenance, architecture, geometry, tokenizer, real processor,
fold, and exact-schedule audits. Its standalone worst-case high-tier CUDA smoke
also passed at 4.614 GiB peak allocated / 4.838 GiB peak reserved. An initial
launch failed closed while an unrelated approximately 1 GiB GPU workload was
active. The clean fold-0 run was later stopped after two complete epochs and
481/615 optimizer updates because its exact held-fold score remained far below
the CPU fallback. The machine-readable audits are
outputs/model50/dry_run.json and outputs/model50/smoke/fold0.json. Do not use
Model50 for a submission until a real fold improves the exact held-fold
multi-score.

Measured fold-0 result
----------------------
Epoch 1 scored 0.359866622 raw. Epoch 2 improved to 0.394881794 raw; the frozen
Model46 context repair raised that to only 0.396277996. Model66 scores
0.819849390 on the same fold. Continuing the final partial epoch was rejected
in favor of the already-demonstrated Kraken Model36 path. The epoch-2 adapter,
predictions, metrics, and transfer receipt remain under outputs/model50 and
outputs/model46/model50_epoch002_transfer.

Hypothesis
----------

Model43 spends roughly 900--1024 visual tokens on every crop. Its full geometry
audit shows two obvious source-raster families: 1,378 of 5,472 images have a
long side of at least 1,600 pixels, while most remaining images are roughly
1,000-pixel resized crops. The native-high-resolution subset is where retaining
more genuine raster detail is most defensible; uniformly upscaling the resized-
low family to 1,000 tokens is expensive and cannot recreate missing pixels.

Model50 therefore keeps Model43's strongest organizer-aligned architecture and
training protocol, but preregisters a two-tier visual budget. The intended gain
is substantially faster experimentation while reserving detail for source images
that contain it and for uncommon geometries likely to suffer at the low budget.

Frozen, label-free tier policy
------------------------------

The tier is selected from source width and height before tokenization:

* high tier, 896--1024 tokens, if the longer source side is at least 1,600 px;
* also high tier if long:short aspect is at most 6:1 (compact/multiline shape);
* also high tier if long:short aspect is at least 25:1 (extremely elongated);
* low tier, 512--640 tokens, for every other image.

The boundaries are inclusive and frozen as policy
source_geometry_v1_long1600_aspect6_25. The selector API accepts only width,
height, fixed thresholds, and fixed token bounds. It cannot receive a target,
ID, fold, train/test flag, model prediction, or confidence. The same rule is
used during training, held-fold validation, and test inference. There is no
post-inference routing, per-row prediction selection, or confidence gate.

Both tiers retain the complete image. Each image is uniformly Lanczos-scaled
onto a median-corner-colour letterbox, never cropped or stretched. Canvas axes
are multiples of Qwen's 28-pixel merged-patch factor, and the runtime verifies
that the real processor emits exactly the planned grid.

Observed geometry and compute proxy
-----------------------------------

The frozen rule was audited over all 5,472 supplied JPGs:

* high: 1,673 rows (30.57383041%); token min/median/max 903/1017/1024,
  mean 993.6103;
* low: 3,799 rows (69.42616959%); token min/median/max 530/630/640,
  mean 612.5078;
* high reasons: 1,372 native-only, 289 elongated-only, 6 compact-only, and
  6 native-plus-hard-geometry;
* overall adaptive min/median/max 530/636/1024, mean 729.0254;
* Model43 baseline min/median/max 900/1017/1024, mean 998.8072;
* maximum content aspect error 0.351849%, with no crop or stretch.

The referenced partitions are balanced: train assigns 1,264/4,095 (30.8669%)
to high and test assigns 408/1,374 (29.6943%) to high. Native-high proportions
are 25.2503% and 24.9636%, respectively. The image archive contains three JPGs
not referenced by either CSV; they remain in the all-image geometry audit but
cannot enter training or inference.

One complete dataset pass contains 3,989,227 adaptive visual-token
presentations versus 5,465,473 under Model43, saving 1,476,246 or 27.0104%.
A visual-token-square upper proxy falls by 43.5992%. These are geometry-only
compute proxies, not measured wall-clock claims; dataloading, frozen vision,
language attention, backpropagation, and validation generation do not all scale
identically. Given the observed roughly 78 seconds/update for Model43, a naive
linear extrapolation is roughly 57 seconds/update, but only a timed CUDA run can
establish the real value.

Architecture and optimization contract
----------------------------------------

Model50 retains Model43's organizer-starter-aligned core:

* pinned Qwen/Qwen2-VL-2B-Instruct checkpoint and Apache-2.0 license;
* generic exact-transcription prompt, with no geometry hint or OCR draft;
* rank 16, alpha 32, dropout 0.05 LoRA on exactly 196 full language paths:
  q/k/v/o and gate/up/down across all 28 language layers;
* complete 32-block visual tower and visual merger frozen;
* 392 trainable tensors and exactly 18,464,768 trainable parameters;
* NF4 double quantization, float16 compute, SDPA, batch size 1, gradient
  accumulation 16, and non-reentrant gradient checkpointing;
* three complete epochs at learning rate 3e-5;
* greedy generation, 48 new-token cap, repetition penalty 1.05, and no-repeat
  4-grams.

The isolated implementation is
src/road_htr/qwen2_vl_adaptive_resolution.py. It imports audited read-only
Model43 architecture/schedule primitives but does not edit, monkeypatch, or
write through Model43. Model50 has its own config, output tree, wrappers, tests,
and manifests.

Sequence and fold audit
-----------------------

The pinned tokenizer and actual per-row adaptive plans found:

* worst training sequence: 1,108 tokens, ID WwlTCykxjP3c4kfo;
* its image grid: 1,020 tokens in the high tier;
* longest target: 34 tokenizer tokens, ID SXNZ5kUCcAxRUpCN;
* maximum-budget generation prompt: 1,079 tokens;
* generation upper bound: 1,127 tokens;
* training/generation caps: 1,152 / 1,200, below Qwen's 32,768 context.

The real slow Qwen processor reproduced both audited grids exactly: a 1,024-
token high sample and a 640-token low sample. No processor truncation is used.

Every outer fold has 3,276 training rows and 819 held rows. The exact ceil
schedule keeps the final 12-row accumulation remainder:

* ceil(3276 / 16) = 205 optimizer updates per epoch;
* 205 * 3 = 615 optimizer updates total;
* 3,276 * 3 = 9,828 row presentations.

Training rejects a partial epoch, missing validation epoch, floor-derived
612-step schedule, overlapping folds, or any unexpected output-directory reuse.

Audit and tests
---------------

From the repository root:

  PYTHONPATH=src .venv-vlm/bin/python model50/audit.py
  PYTHONPATH=src .venv-vlm/bin/python -m unittest \
    tests.test_model50_qwen_adaptive_resolution -v

The CPU audit writes outputs/model50/dry_run.json. The model-specific suite has
11 tests and also passes in .venv-kraken for environment-independent coverage.

Mandatory GPU smoke and training
--------------------------------

Run only after the GPU is free:

  PYTHONPATH=src .venv-vlm/bin/python model50/train.py --fold 0 --smoke-only

The smoke must stay below 5.3 GiB allocated and reserved on the real 6 GiB GPU.
It exercises the actual worst adaptive batch through forward, backward, and
optimizer work. Do not raise or bypass the cap.

After a successful smoke, train fold 0:

  PYTHONPATH=src .venv-vlm/bin/python model50/train.py --fold 0

Held-fold or test inference uses the same immutable geometry rule:

  PYTHONPATH=src .venv-vlm/bin/python model50/infer.py \
    --input data/processed/train_folds.csv \
    --adapter outputs/model50/fold0/best_adapter \
    --output outputs/model50/fold0/oof.csv --fold 0

Decision rule and risks
-----------------------

* Compare exact fold-0 weighted WER, weighted CER, multi-score, truncation, and
  repetition diagnostics with Model43 before spending on all folds.
* The low tier can discard useful interpolation-scale cues even though it cannot
  discard native pixels that were absent. Adaptive training may also make batch
  compute less uniform. These are empirical risks, not resolved by CPU audits.
* The aspect safeguards deliberately add 295 non-native images to the high tier;
  they are fixed geometry exceptions, not error-fitted row IDs.
* A single fold is a challenger decision, not a robust OOF estimate and not a
  basis for a one-shot leaderboard submission.
