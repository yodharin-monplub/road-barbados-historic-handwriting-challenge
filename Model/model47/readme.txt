MODEL 47 — HIGH-RESOLUTION QWEN2-VL WITH RECEIPTED RAW MODEL8 DRAFTS

Status
------
Model47 is an isolated, CPU-audited challenger. It has no OOF score, no
leaderboard score, and no claimed improvement. No Model47 GPU model has been
loaded, smoked, trained, or used for inference. Training is fail-closed until
the complete canonical Model32 receipt-v2 train draft bundle exists; test
inference also requires the corresponding test bundle. The current CPU report
is outputs/model47/dry_run.json and records score=null and gpu_started=false.

Hypothesis
----------
Model47 keeps Model43's organizer-aligned, high-resolution Qwen2-VL recipe but
adds Model8's frozen zero-shot OCR output as a literal untrusted draft. Qwen
still sees the full image and is told to copy the draft only when the pixels
support it. This may provide useful character and word structure without
giving the model any challenge label-derived feature.

The draft is inserted with literal string replacement between newline-isolated
markers. Braces in OCR output are data, not formatting instructions. The
loader rejects chat-control substrings, unclean edges/line controls, more than
160 UTF-8 bytes, or more than 160 Qwen tokens. A genuine empty Model8 result is
not missing data: receipt v2 binds its ordered ID and preserves a blank between
the markers. In the audited train source, PO7QQLWIFOT65BTz is such a case.
Model47 never invents replacement text, substitutes another crop, or uses the
label to fill/select a draft.

Immutable composition and leakage contract
------------------------------------------
Model47 hash-pins and imports rather than copies:

* Model43 config and implementation for Qwen architecture, letterboxing,
  exact epoch scheduling, checkpoint/header audits, and CUDA gates.
* Model32 config, implementation, and producer wrapper for the only accepted
  canonical raw Model8 receipt contract.

The accepted producer is one frozen PP-OCRv6 medium checkpoint, CPU float32,
batch one, full-image view, raw unconstrained decoding, no fold, and an ID-only
input projection. It never receives Target. Receipt verification binds the
ordered source IDs, every source image hash, checkpoint, generator config and
code, CSV, confidence JSONL, source manifest, blank-output ledger, and all
relevant SHA256 values. Missing, partial, reordered, modified, noncanonical,
or label-dependent artifacts fail before CUDA allocation.

Each role requires all five canonical files under outputs/model32/drafts:

  train_model8_raw.csv
  train_model8_raw.ids.csv
  train_model8_raw.jsonl
  train_model8_raw.manifest.json
  train_model8_raw.receipt.json

The test bundle has the analogous test_model8_raw names. These files are
owned by Model32 and are read-only to Model47.

Architecture and exact parameter accounting
-------------------------------------------
The trainable adapter is identical to Model43 and the organizer starter's
language-side intent:

* Qwen/Qwen2-VL-2B-Instruct at pinned revision
  895c3a49bc3fa70a340399125c650a463535e71c.
* Rank 16, alpha 32, dropout 0.05 LoRA on q_proj, k_proj, v_proj, o_proj,
  gate_proj, up_proj, and down_proj in all 28 language layers.
* Exactly 196 target modules and 392 trainable LoRA tensors.
* Exactly 18,464,768 trainable parameters, derived from pinned checkpoint
  tensor shapes as rank * (in_features + out_features) for every module.
* Exactly 1,310,195,712 frozen base-matrix parameters are targeted.
* The complete visual tower and visual merger remain frozen; there are zero
  visual or merger adapter targets.

This is much broader than Model32's inherited 58-target adapter. Compared with
Model43, Model47 changes the prompt and sequence caps only—not adapter capacity,
image policy, optimizer, schedule, or decoding.

Image, sequence, and processor gates
-----------------------------------
The full image is aspect-preservingly resized with Lanczos into a
median-corner-colour letterbox. Canvas dimensions are exact multiples of
Qwen's 28-pixel merged-patch factor. The factor-aligned policy uses 896..1024
effective visual tokens and never crops or stretches. The shared geometry
audit over all 5,472 images reports 900 minimum, 1,017 median, and 1,024 maximum
effective tokens, with minimum content fraction 0.8511904762.

Model47 raises only its own envelopes to 1,344 training tokens and 1,392
prompt-plus-generation tokens. The pinned Qwen context is 32,768. Before a
draft receipt exists, a conservative proof treats every one of the 160
permitted UTF-8 bytes as one byte-fallback token and adds 16 boundary tokens.
The pinned tokenizer proves all 256 byte-alphabet symbols exist. Results are:

* largest label-conditioned scaffold at 1,024 image tokens: 1,161 tokens;
* conservative maximum training sequence: 1,337 (7 tokens spare);
* generation scaffold: 1,125 tokens;
* conservative maximum generation prompt: 1,301 tokens;
* prompt plus 48 generated tokens: 1,349 (43 tokens spare);
* longest canonical target: 34 tokens, leaving 14 of the 48-token cap.

That proof establishes a safe envelope, not actual receipt readiness. Once a
receipt exists, Model47 additionally tokenizes every exact raw draft and exact
row geometry. It then encodes every intended row with the real pinned Qwen
processor before loading the CUDA model: all 3,276 outer-training rows with
targets plus all 819 validation prompts for training, all 4,095 train rows for
OOF inference, or all 1,374 test prompts for test inference. No truncation is
allowed. A cap violation or incomplete row count aborts.

Schedule, decoding, and memory
-----------------------------
Each outer fold has 3,276 training and 819 validation rows. With batch size 1,
gradient accumulation 16, and three epochs, Model47 explicitly uses:

  ceil(3276 / 16) = 205 optimizer updates per epoch
  205 * 3 = 615 total updates
  3276 * 3 = 9828 row presentations

max_steps=615 overrides Transformers' floor-derived 612-step behavior. Epoch
callbacks are guarded at steps 205, 410, and 615 so each validation follows a
complete data pass. Learning rate is 3e-5. Greedy decoding uses 48 new tokens,
repetition penalty 1.05, and no-repeat 4-grams.

NF4 double quantization, float16 compute, SDPA, non-reentrant gradient
checkpointing, paged 8-bit AdamW, and one explicit CUDA device are inherited.
A real forward/backward/optimizer smoke on the exact longest processor-encoded
training row must keep both peak allocated and peak reserved CUDA memory
strictly below 5.3 GiB. CPU audits cannot establish that result.

CPU audit and tests
-------------------
Run from the repository root without exposing a GPU:

  CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv-vlm/bin/python model47/audit.py
  CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv-vlm/bin/python -m unittest \
    tests.test_model47_qwen_starter_highres_guided -v

The audit is expected to report a receipt blocker until the two canonical
Model32 bundles have been generated. It still verifies Model43/Model32 hash
pins, canonical data, starter/checkpoint provenance, all 196 tensor shapes,
all 5,472 image geometries, the conservative sequence proof, and all five
205/615 fold schedules. It does not manufacture a score.

Create canonical drafts (CPU only)
----------------------------------
First inspect the producer plans:

  CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv-kraken/bin/python \
    model32/prepare_drafts.py --input data/processed/train_folds.csv \
    --output outputs/model32/drafts/train_model8_raw.csv --dry-run

  CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv-kraken/bin/python \
    model32/prepare_drafts.py --input data/raw/Test.csv \
    --output outputs/model32/drafts/test_model8_raw.csv --dry-run

Then remove or resolve any pre-existing partial bundle through the coordinated
Model32 workflow before running the same commands without --dry-run. The
producer refuses to overwrite artifacts. Draft recognition is CPU-only and
slow. Do not fabricate a receipt or hand-edit any generated file. Rerun the
Model47 audit afterward; it must verify both receipt-v2 bundles and complete
the exact every-row tokenizer and processor preflights.

Mandatory GPU smoke and training
--------------------------------
Only after the train receipt and CPU audit pass, and only when the GPU is free:

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv-vlm/bin/python \
    model47/train.py --fold 0 --smoke-only

If and only if that real smoke is strictly below both 5.3 GiB limits:

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv-vlm/bin/python \
    model47/train.py --fold 0

Validation generates the entire held fold after each exact epoch. It reports
the complete Model47 vector and the complete unchanged Model8-draft vector.
The saved adapter is chosen only by Model47's whole-fold score; there is no
per-row oracle or target-aware draft choice. Train folds 1..4 only if the
predeclared fold-0 comparison justifies the cost.

OOF and test inference
----------------------
Held-fold inference requires a matching excluded-fold adapter lineage:

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv-vlm/bin/python model47/infer.py \
    --input data/processed/train_folds.csv \
    --drafts outputs/model32/drafts/train_model8_raw.csv \
    --draft-receipt outputs/model32/drafts/train_model8_raw.receipt.json \
    --fold 0 --adapter outputs/model47/fold0/best_adapter \
    --output outputs/model47/oof/fold0.csv

One fold's test candidate and submission:

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv-vlm/bin/python model47/infer.py \
    --input data/raw/Test.csv \
    --drafts outputs/model32/drafts/test_model8_raw.csv \
    --draft-receipt outputs/model32/drafts/test_model8_raw.receipt.json \
    --adapter outputs/model47/fold0/best_adapter \
    --output outputs/model47/test/fold0.csv \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model47/fold0.csv

Risks
-----
The 1,024-image-token, rank-16 configuration may exceed 5.3 GiB despite the
static accounting; only the real smoke decides. Model8 drafts can be wrong or
blank and may anchor Qwen to an error. The extra prompt consumes context and
compute, while letterbox upscaling may blur thin strokes. Finally, one fold is
a challenger screen rather than a robust performance estimate. Compare exact
held-fold WER, CER, multi-score, repetition, and truncation diagnostics before
spending five-fold compute or making one leaderboard submission.
