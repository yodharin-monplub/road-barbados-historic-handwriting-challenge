Model43 - organizer-aligned rank-16 language QLoRA at high resolution
=====================================================================

Status
------

Model43 is implemented and has passed its CPU provenance, checkpoint-header,
target-shape, tokenizer/context, image-geometry, fold, schedule, and real CUDA
memory smoke audits. The smoke used the worst 1,024-token case and measured
4.631 GiB peak allocated / 4.857 GiB peak reserved. An uncheckpointed 16-step
fold-0 timing pilot ran at about 80 seconds/update and was deliberately stopped
to pursue more efficient successors. It has no held-fold OOF or Zindi score and
is not a submission candidate.

Hypothesis
----------

The organizer VLM starter points to a strong, simple recipe: Qwen2-VL-2B,
rank-16/alpha-32 LoRA on q/k/v/o attention and gate/up/down MLP projections,
three epochs, learning rate 3e-5, and a generic exact-transcription instruction.
Model43 retains that core while testing whether much more complete visual detail
fixes the likely under-resolution failure mode in Model39/40.

Compared with the organizer starter
-------------------------------------

Kept exactly in intent:

* Qwen/Qwen2-VL-2B-Instruct.
* LoRA rank 16, alpha 32, dropout 0.05.
* q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, and down_proj.
* Three epochs and learning rate 3e-5.
* The starter transcription prompt, verbatim:
  "Transcribe EXACTLY as seen. Preserve line breaks, punctuation, and spacing.
  Do not add explanations. Return verbatim transcription. No paraphrasing. No
  normalization."

Made explicit or safer:

* The seven suffix families are resolved to exactly 196 full language paths:
  28 language layers x 7 matrices. A regex is used only as an audited discovery
  check; PEFT receives an exact anchored path pattern. This prevents accidental
  matching of similarly named visual modules.
* The complete 32-block visual tower and visual merger are frozen. The organizer
  starter did not explicitly adapt them, so Model43 does not add a merger LoRA.
  This deliberately isolates the starter's language-adapter hypothesis and saves
  memory for image tokens. A merger adapter would confound that comparison and
  add 270,336 rank-16 parameters.
* The checkpoint revision, every local checkpoint file hash, starter archive,
  and the two starter member hashes are pinned and verified offline.
* The saved Qwen slow image processor is selected explicitly (`use_fast=False`),
  and its 896..1024-token pixel interval is checked after loading, avoiding a
  silent preprocessing change if a Transformers default changes.
* NF4 double quantization, float16 compute, SDPA, batch size 1, gradient
  accumulation 16, non-reentrant gradient checkpointing, and paged 8-bit AdamW
  are fixed for the 6 GiB card.
* Validation uses the existing deterministic five full folds. Training and held
  fold IDs are checked disjoint, and labels retain internal spaces and line
  breaks. Validation runs only at a verified complete-epoch boundary.
* Greedy decoding is capped at 48 new tokens with repetition penalty 1.05 and
  no-repeat 4-grams to contain the repetition failure observed previously.

Exact adapter accounting
------------------------

Each LoRA matrix contributes rank * (in_features + out_features). The ledger is
derived from the actual pinned safetensor headers, not estimated:

* target modules: 196 language, 0 vision, 0 merger
* trainable tensors: 392 (one A and one B tensor per target)
* trainable parameters: 18,464,768
* targeted frozen base-matrix parameters: 1,310,195,712

The runtime repeats the shape/layout audit before injection and then independently
counts every trainable tensor after injection. Any visual trainable, non-LoRA
trainable, missing target, extra target, shape drift, or count drift aborts.

High-resolution full-image policy
---------------------------------

Every complete crop is kept. Model43 uniformly scales it with Lanczos into a
median-corner-colour letterbox; it never crops, stretches, or concatenates image
fragments. Canvas width and height are exact multiples of Qwen's 28-pixel merged
patch factor. An aspect-matched grid is chosen in the 896..1024 effective-token
interval, maximizing real content area while staying under the hard cap.

The full 5,472-image geometry audit found:

* effective image tokens: minimum 900, median 1017, maximum 1024
* content fraction: minimum 0.8511904762, median 0.9744897959, maximum 1.0
* maximum content aspect error: 0.0033229182
* 4,100 crops upscale, 1,372 downscale, none are cropped or stretched

This is 3.2x Model39's 320-token cap and 1.6x Model40's 640-token cap. It is also
an intentional full-image policy rather than the starter's aspect-dependent
thumbnail heuristic. The organizer config lists max_pixels=1,500,000, but its
pre-thumbnailing means that number is not an effective per-example token promise;
Model43 audits the actual aligned token grid for every image.

Sequence and schedule audits
----------------------------

At the conservative 1024-image-token ceiling, the pinned tokenizer found:

* longest training sequence: 1115 tokens (ID SXNZ5kUCcAxRUpCN)
* longest target: 34 tokenizer tokens
* generation prompt: 1079 tokens
* generation upper bound: 1079 + 48 = 1127 tokens
* configured training/generation envelopes: 1152 / 1200 tokens
* pinned Qwen context: 32768 tokens

A real CPU processor encode of the worst static record produced exactly 1024
image tokens, 1115 total tokens, and 36 supervised assistant tokens. Thus the
static calculation and processor grid agree without truncation.

For every fold there are 3276 train and 819 validation rows. Transformers 4.49
would derive 204 updates by floor division even though the final group contains
12 valid rows. Model43 supplies and guards the exact ceil schedule:

* ceil(3276 / 16) = 205 optimizer updates per epoch
* 205 * 3 = 615 total optimizer updates
* 3276 * 3 = 9828 row presentations

The Trainer bookkeeping is replaced with these audited values, drop_last is
false, and callbacks fail if an epoch ends anywhere other than steps 205, 410,
and 615. The schedule is written to run_manifest.json and complete.json.

CPU audit and tests
-------------------

From the repository root:

  PYTHONPATH=src .venv-vlm/bin/python model43/audit.py
  PYTHONPATH=src .venv-vlm/bin/python -m unittest tests.test_model43_qwen_starter_highres -v

The machine-readable CPU report is outputs/model43/dry_run.json. All 15
model-specific tests pass in .venv-vlm. The 14 environment-independent tests
also pass in .venv-kraken; its VLM-runtime check is intentionally skipped because
that environment does not install Transformers. Training must use .venv-vlm.

Mandatory GPU smoke (run only when the GPU is free)
---------------------------------------------------

  PYTHONPATH=src .venv-vlm/bin/python model43/train.py --fold 0 --smoke-only

This loads the actual 4-bit model, verifies all runtime targets, trainable tensors,
NF4 modules, CUDA residency, and runs forward/backward/optimizer work on the
audited worst training record. It fails closed if either peak allocated or peak
reserved CUDA memory exceeds 5.3 GiB. Do not bypass or raise this threshold.

Training and inference after a successful smoke
-----------------------------------------------

Train one leakage-safe fold (fold 0 is the first challenger):

  PYTHONPATH=src .venv-vlm/bin/python model43/train.py --fold 0

Infer the held fold from its best adapter:

  PYTHONPATH=src .venv-vlm/bin/python model43/infer.py \
    --input data/processed/train_folds.csv \
    --adapter outputs/model43/fold0/best_adapter \
    --output outputs/model43/fold0/oof.csv \
    --fold 0

Infer test data and create a submission (adjust the input/sample paths to the
actual repository files if their names differ):

  PYTHONPATH=src .venv-vlm/bin/python model43/infer.py \
    --input data/raw/Test.csv \
    --adapter outputs/model43/fold0/best_adapter \
    --output outputs/model43/fold0/test_predictions.csv \
    --sample data/raw/SampleSubmission.csv \
    --submission outputs/model43/fold0/submission.csv

Risks and decision rule
-----------------------

* 1024 visual tokens plus a rank-16 language adapter may still exceed 5.3 GiB on
  the real card. CPU audits cannot certify CUDA allocator behaviour. If smoke
  fails, do not start training; lower the visual interval/cap in a new model so
  this experiment remains reproducible.
* Upscaling can make fine pen strokes easier for the visual encoder, but may add
  interpolation blur. The letterbox avoids the more damaging alternatives of
  glyph stretching or crop loss.
* A frozen visual tower limits domain adaptation. That is deliberate here: the
  experiment asks whether the organizer-aligned language capacity plus better
  pixels is enough. Model40 remains the complementary visual-adaptation branch.
* A single fold is suitable for deciding whether to continue, not for claiming a
  robust leaderboard gain. Compare exact held-fold weighted WER, weighted CER,
  multi-score, repetition rate, and truncation diagnostics before making one
  submission.
