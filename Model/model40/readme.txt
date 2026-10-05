Model40: full-vision, higher-resolution Qwen2-VL challenger
==========================================================

Status
------
Implemented and CPU-audited only. No GPU process was started while Model39 was
running, and no Model39 source or output was changed. Model40 has no validation
score and is not a submission candidate until its real 6 GiB smoke passes and
a complete fold-0 run beats the simpler alternatives.

Hypothesis
----------
Model39 adapts the final eight vision blocks at a 320-token image cap. Model40
tests the deliberately more visual extreme: every vision block is adapted and
the complete crop is presented at up to 640 post-merger image tokens. This may
help distinguish faded, tightly spaced historical glyphs and recover complete
secondary lines that low resolution suppresses.

The prompt now asks for every complete handwritten line in reading order and
ignores only fragments clipped by the crop boundary. It does not instruct the
model to discard complete neighboring lines. Labels remain edge-trimmed only;
original spelling, capitalization, punctuation, line breaks, and repeated
internal spaces remain supervised exactly.

Exact adapter architecture
--------------------------
Base: Qwen/Qwen2-VL-2B-Instruct
Pinned revision: 895c3a49bc3fa70a340399125c650a463535e71c
Licence: Apache-2.0

Rank-8 LoRA is injected with one anchored exact-path matcher into:

  * 196 language modules: q/k/v/o and gate/up/down in all 28 layers;
  * 128 visual modules: qkv/proj/fc1/fc2 in all 32 vision blocks;
  * 2 visual-merger MLP matrices.

The 326 matrices produce exactly 652 LoRA tensors and 14,610,432 trainable
parameters, calculated from each pinned checkpoint matrix as
rank * (in_features + out_features):

  language:  9,232,384
  vision:    5,242,880
  merger:      135,168
  total:    14,610,432

The targeted frozen base matrices contain 1,973,420,032 parameters. Offline
safetensors-header checks and runtime module-graph checks must independently
agree on every path, dimension, target count, trainable tensor, and parameter.

Aspect-preserving 640-token visual policy
-----------------------------------------
Each full RGB crop is uniformly scaled with Lanczos inside a factor-28-aligned
canvas. The canvas is selected from 512..640 visual tokens to maximize actual
handwriting pixels under the cap; ties prefer less padding, then more tokens.
Padding uses the median corner color. There is no crop, glyph stretching,
thumbnail distortion, or line-band selection. Because the canvas is already
factor-aligned and within 501,760 pixels (640 * 28 * 28), Qwen leaves its
dimensions unchanged. Integer content rounding is limited to 0.5% relative
aspect error.

The dry geometry audit over all 5,472 images reports:

  * 4,094 uniformly upscaled and 1,378 downscaled;
  * no unchanged or cropped images;
  * effective Qwen grids from 520 to 640 tokens (median 636);
  * real-content pixel fraction from 0.807143 to 1.0 (median 0.976190);
  * maximum relative content aspect error about 0.004326.

Some aspect ratios use fewer than 640 tokens or up to 19.29% border padding.
This maximizes real content resolution while keeping glyph geometry intact;
stretching the handwriting merely to fill the grid is explicitly forbidden.

Exact sequence/context budget
-----------------------------
The 640 image tokens conflict with Model39's old 512-token total cap, so Model40
raises the training sequence cap to 768. This is supported by the pinned
Qwen configuration's 32,768-position context window. The offline audit uses the
pinned tokenizer and every one of the 4,095 labels. With a conservative full
640 image tokens, the largest training template is 748 tokens. The generation
prompt is 712 tokens; adding at most 48 generated tokens gives 760, below the
conservative 816 context cap. Runtime collator checks independently reject any
actual image or sequence over its cap.

Generation uses greedy decoding, max_new_tokens=48, repetition_penalty=1.05,
and no_repeat_ngram_size=4. The training-set maximum target length is 34 Qwen
tokens, leaving generation headroom while blocking the long repetition failure
seen with the former 160-token ceiling.

Exact epoch schedule
--------------------
There are 3,276 batch-one training rows per fold and accumulation 16. Model40
inherits the corrected ceil arithmetic as isolated code:

  204 full accumulation groups + one 12-row remainder = 205 updates/epoch
  205 updates * 3 complete epochs = 615 updates
  3,276 rows * 3 epochs = 9,828 row presentations

Training explicitly sets max_steps=615 and dataloader_drop_last=false. A custom
Trainer mixin corrects Transformers 4.49's floor-based epoch bookkeeping, and a
guard requires validation boundaries exactly at steps 205, 410, and 615. The
schedule is persisted in dry-run, manifest, and completion artifacts.

Memory contract and real smoke
------------------------------
The base remains offline NF4 with double quantization, float16 compute, SDPA,
non-reentrant gradient checkpointing, explicit single-GPU placement, batch size
1, and paged 8-bit AdamW. All frozen vision weights remain frozen; only their
LoRA branches train. The larger image grid and all-block visual adapters make
this materially riskier than Model39 on the RTX 4050 6 GiB.

Training is impossible until a worst-case post-resize, complete-label real
forward/backward pass verifies that both CUDA peak allocated and allocator
reserved memory are strictly below 5.3 GiB. Do not bypass this gate.

Offline audit (no CUDA)
-----------------------

  PYTHONPATH=src .venv-vlm/bin/python model40/audit.py

This writes outputs/model40/dry_run.json with checkpoint/starter hashes, all
326 target dimensions, tokenizer sequence limits, all-image resize geometry,
five disjoint fold splits, and exact optimizer schedules.

Real smoke, only after the GPU is free
--------------------------------------

  PYTHONPATH=src .venv-vlm/bin/python model40/train.py --fold 0 --smoke-only

The smoke report is outputs/model40/smoke/fold0.json. It must show exactly 326
injected targets, 14,610,432 trainable parameters, 652 LoRA tensors, full CUDA
residency, NF4 modules, and both peaks below 5.3 GiB.

Fold-0 training after a smoke pass
----------------------------------

  PYTHONPATH=src .venv-vlm/bin/python model40/train.py --fold 0

All 819 held-out rows are greedily decoded after each complete epoch and scored
with the exact challenge metric. Continue folds 1-4 only if this conditional
fold-0 challenger is competitive. A single tuned fold is not unbiased OOF.

Inference example
-----------------

  PYTHONPATH=src .venv-vlm/bin/python model40/infer.py \
    --input data/processed/train_folds.csv \
    --fold 0 \
    --adapter outputs/model40/fold0/best_adapter \
    --output outputs/model40/oof/fold0.csv

Principal risks
---------------
The doubled visual grid increases attention activations and runtime, so the
smoke may fail despite modest adapter storage. Upscaling can magnify paper noise
and compression artifacts, while letterbox padding adds blank context for the
most extreme aspect ratios. Adapting all early vision blocks can overfit only
3,276 training rows. Finally, asking for every complete line is more faithful
to the richer crop but may conflict with annotations if they intentionally
select only one line. Fold-0 evidence must decide; capacity alone is not proof.
