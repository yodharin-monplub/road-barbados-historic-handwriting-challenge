Model39: broad architecture-bound Qwen2-VL QLoRA challenger
===========================================================

Status
------
Rejected after the complete epoch-1 fold-0 gate. The corrected run trained one
full epoch and greedily decoded all 819 held-out rows. Raw multi-score was only
0.354774323; the frozen Model46 text-repair recipe reached 0.357686046. Only 13
rows were exact and the outputs were fluent but visually ungrounded rather than
empty or merely truncated. The run was deliberately terminated after the last
observed global step 209/615 to reallocate the GPU to high-resolution Model43.

This is not a submission candidate and no complete.json exists. The exact
decision receipt is outputs/model39/fold0/stopped.json. The epoch-1 adapter and
validation vector are retained only for diagnosis and reproducibility.

Why this model exists
---------------------
The organizer VLM starter targets q/k/v/o attention and gate/up/down MLP
projections. Model27 deliberately narrowed that to q/v plus the visual merger
to establish a low-memory baseline. Model39 restores the organizer's broader
language capacity with exact full paths and adds a bounded high-level visual
adapter. Historical handwriting errors are often both visual (similar glyphs,
faded strokes) and linguistic (names and period spelling), so adapting only q/v
can be an avoidable bottleneck.

Exact adapter architecture
--------------------------
Base: Qwen/Qwen2-VL-2B-Instruct
Pinned revision: 895c3a49bc3fa70a340399125c650a463535e71c
Licence: Apache-2.0

Every target is represented as an anchored full-path regex; suffix matching is
forbidden. The rank-8 adapter contains exactly:

  * 196 language modules: q/k/v/o and gate/up/down in all 28 layers;
  * 32 visual modules: qkv/proj/fc1/fc2 in blocks 24 through 31;
  * 2 visual-merger MLP modules.

That is 230 adapted matrices and 460 LoRA tensors. Counts are derived as
rank * (in_features + out_features) for every actual checkpoint matrix:

  language:  9,232,384
  vision:    1,310,720
  merger:      135,168
  total:    10,678,272 trainable parameters

The offline audit reads the safetensors headers and independently verifies all
230 dimensions. At model load, the complete instantiated graph is audited:
all 196 language projections, all four projections in every one of the 32
vision blocks, and both merger linears must match the pinned shapes. It then
checks the exact injected paths and exact trainable parameter/tensor counts.
Architecture or PEFT target drift fails closed.

Why only the last eight visual blocks
-------------------------------------
This is a preregistered capacity/memory compromise. Later visual blocks should
carry the most task-specific glyph/line representations, while adapting all 32
would add 96 more targets and unnecessary optimizer/activation risk on the
6 GiB RTX 4050. The full vision base remains frozen; only the 32 selected LoRA
branches train. The visual merger is retained to align visual features with the
language width.

Memory and optimization contract
--------------------------------
The base is offline NF4 with double quantization, float16 compute, SDPA,
explicit one-GPU placement, non-reentrant gradient checkpointing, batch size 1,
and paged 8-bit AdamW. Full images retain aspect ratio and are limited to 320
merged visual tokens and 512 total tokens. Labels are edge-trimmed only;
capitalization, punctuation, spelling, and repeated internal spaces survive.

Rank 8 gives over four times Model27's trainable capacity while keeping adapter
weights modest. It is expected to fit, but that expectation is not evidence.
The same worst-case real forward/backward smoke gate requires both peak CUDA
allocated and peak CUDA reserved memory to remain strictly below 5.3 GiB.
Training cannot begin if the smoke fails. Do not bypass the gate; reduce the
image-token cap in a new numbered model if needed.

Exact epoch/update schedule
---------------------------
The installed Transformers 4.49 Trainer derives epoch-based max_steps with
floor division. For 3,276 batch-one rows and accumulation 16 it budgets only
204 updates per epoch, or 612 total, even though its inner loop performs a
205th update for the final 12 rows. That mismatch would stop epoch 3 early.

Model39 computes the schedule from the actual selected train-row count before
loading CUDA:

  full groups per epoch:       204 x 16 rows
  remainder group per epoch:     1 x 12 rows
  optimizer updates per epoch: 205
  total updates for 3 epochs:  615
  total row presentations:  9,828

TrainingArguments receives explicit max_steps=615 and
dataloader_drop_last=false. A Model39-only Trainer mixin replaces upstream
floor-based epoch/resume bookkeeping with the same ceil schedule. A guard
callback checks boundaries 205, 410, and 615 before each expensive full-fold
validation, and checks the final global step again. Any partial/extra epoch or
dataloader-length drift fails closed. The computed schedule is recorded in the
dry audit, run manifest, and completion artifact; observed epoch boundaries are
also stored in complete.json.

Offline audit (safe while another GPU job runs)
-----------------------------------------------

  PYTHONPATH=src .venv-vlm/bin/python model39/audit.py

The report is outputs/model39/dry_run.json. It verifies source checkpoint and
starter hashes, tensor-header dimensions, exact counts, all five folds, and
disjoint train/validation IDs. It does not certify CUDA memory.

Required real smoke (run only after the GPU is free)
----------------------------------------------------

  PYTHONPATH=src .venv-vlm/bin/python model39/train.py --fold 0 --smoke-only

The completed report is outputs/model39/smoke/fold0.json. It shows 230 injected
targets, 10,678,272 trainable parameters, 460 trainable tensors, NF4 residency
entirely on CUDA, 2.954 GiB peak allocated, and 3.059 GiB peak reserved. Thus
memory was not the cause of the failed score.

Fold training and exact validation
----------------------------------

  PYTHONPATH=src .venv-vlm/bin/python model39/train.py --fold 0

The model trains only on folds 1-4. At every epoch boundary it greedily decodes
all 819 fold-0 rows and scores the exact challenge multi-score; there is no
sampled validation subset. Best-adapter selection uses that complete held-out
fold. Run folds 1-4 only if fold 0 is competitive, one at a time, with matching
commands. Hyperparameter comparisons on fold 0 are conditional model-selection
evidence, not unbiased five-fold OOF evidence.

Observed result: epoch 1 completed at step 205 and scored 0.354774323 over all
819 fold-0 rows. Model46's pre-frozen context, boundary, and confusion stages
improved it by only 0.002912 to 0.357686046. Training was stopped during epoch 2
because this failed the full-fold competitiveness gate; folds 1-4 must not be
run for Model39.

Generation is capped at 48 new tokens because the audited maximum target is 36
tokens. A deterministic 1.05 repetition penalty and four-token no-repeat gate
bound the phrase loops observed in rejected Model27. An initial Model39 process
using the inherited 160-token setting was stopped at step 26, before any
checkpoint or validation, and its manifest is preserved under
``outputs/model39/fold0_aborted_generation160``. The corrected run starts from
the pristine base; it does not resume those weights.

Inference example
-----------------

  PYTHONPATH=src .venv-vlm/bin/python model39/infer.py \
    --input data/processed/train_folds.csv \
    --fold 0 \
    --adapter outputs/model39/fold0/best_adapter \
    --output outputs/model39/oof/fold0.csv

For test inference, omit --fold and use data/raw/Test.csv. Keep five fold-model
candidates separate and use a fixed cross-fold ensemble/deployment rule rather
than selecting a fold from public leaderboard feedback.

Main risks
----------
Broad MLP adaptation can overfit 3,276 training rows, and the last-eight-block
vision choice is a plausible but unvalidated inductive bias. Fold-0 comparisons
against Model27 are correlated and model-selected. CUDA memory was certified by
the real smoke, but full-fold accuracy failed. Retain the established Kraken
candidate rather than assuming added capacity or more epochs would help.

Postmortem: the anticipated visual-resolution risk materialized. A 320-token
image cap preserved output length and language fluency but not enough glyph
evidence for this task. Model43 is the registered higher-resolution successor;
it changes the image representation and returns to the organizer-aligned frozen
visual tower instead of treating more epochs on this failed vector as evidence.
