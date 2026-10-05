MODEL 32 — OCR-GUIDED QWEN2-VL CHALLENGER

Status
------
The isolated implementation, lineage gates, CPU draft builder, and unit tests
are present. One CPU draft attempt was stopped after 100/4,095 rows when its
genuine blank output exposed the old contract bug; those partial files were
archived outside the canonical draft directory. No complete canonical draft
receipt, Qwen adapter training, or Qwen inference exists. Consequently Model32
has no claimed OOF score yet.

Hypothesis
----------
Model27 asks Qwen2-VL to read the image from scratch. Model32 gives the same
model the complete image plus a zero-shot OCR draft and asks it to correct the
draft using the pixels. The draft should supply useful word/line structure;
Qwen's visual reasoning can concentrate on uncertain characters and obvious
historical spelling errors. The assistant target is still the exact label and
only assistant tokens receive loss.

The prompt calls the draft untrusted, places the image first, asks for only a
transcription, and preserves capitalization, punctuation, spelling, and
genuine internal double spaces. Draft braces are inserted with literal string
replacement, not format evaluation. Neither labels nor predictions are
whitespace-collapsed.

Why Model25 drafts are forbidden for OOF
----------------------------------------
Model25 is cross-fitted for its own prediction row, but it is not nested for a
different outer Model32 fold. For outer fold F, a Model25 draft on training row
G can have been selected/fitted using labels from F. Feeding those drafts into
Model32 training would indirectly leak the outer validation labels. Model32's
loader therefore accepts exactly one source kind and rejects Model25, any
label-fitted postprocessor, and any charset-constrained Model8 output.

The accepted source is the frozen, organizer-approved Model8 PP-OCRv6 medium
checkpoint run zero-shot on the full image with its raw greedy decoder:

  * checkpoint SHA256 is pinned;
  * fold is null, so one identical static model handles every row;
  * image_variant is exactly full;
  * constrain_charset is exactly false;
  * internal spaces are preserved;
  * inference is CPU float32, batch one;
  * Model32 projects the source table to a one-column ordered-ID CSV before it
    invokes the pinned inference path, so Target is physically unavailable.

Thus every draft is a function only of the frozen external checkpoint, image
pixels, and fixed inference code. It is safe for every outer fold without 20
pair-excluded draft refits. The receipt binds exact ordered IDs, source input,
draft CSV, generic inference manifest, generator config/code, checkpoint, and
all SHA256 values. Receipt schema v2 also binds the count, ordered ID list, and
canonical digest of genuine empty Model8 outputs. A present empty output is
inserted as an exact blank between DRAFT markers; this is a fixed,
label-independent policy. Missing/reordered rows, absent values, dirty edge
whitespace, altered ledgers, constrained/partial outputs, and label-dependent
artifacts still fail closed.

Create the label-independent drafts (CPU only)
----------------------------------------------
Inspect both plans without writing or importing Kraken:

  PYTHONPATH=src python3 model32/prepare_drafts.py \
    --input data/processed/train_folds.csv \
    --output outputs/model32/drafts/train_model8_raw.csv \
    --dry-run

  PYTHONPATH=src python3 model32/prepare_drafts.py \
    --input data/raw/Test.csv \
    --output outputs/model32/drafts/test_model8_raw.csv \
    --dry-run

Then use the Kraken environment for the actual CPU recognition. This is slow
but does not occupy CUDA:

  CUDA_VISIBLE_DEVICES='' .venv-kraken/bin/python model32/prepare_drafts.py \
    --input data/processed/train_folds.csv \
    --output outputs/model32/drafts/train_model8_raw.csv

  CUDA_VISIBLE_DEVICES='' .venv-kraken/bin/python model32/prepare_drafts.py \
    --input data/raw/Test.csv \
    --output outputs/model32/drafts/test_model8_raw.csv

Each command creates its prediction CSV, ID-only .ids.csv input, Model8
.manifest.json, confidence .jsonl, and Model32 .receipt.json under
outputs/model32/drafts. The JSONL IDs/candidates must reproduce the CSV, and
the receipt also binds every ordered source-image hash and its explicit empty
draft ledger. Existing artifacts are never overwritten. The config defaults
name the corresponding train_model8_raw.csv and test_model8_raw.csv paths.

Pinned Model27 contract
-----------------------
Model32 imports Model27 rather than copying its model-loading internals. It
hash-pins Model27's config and implementation before use, then changes only the
prompt and output directory. The inherited runtime remains:

  * Qwen/Qwen2-VL-2B-Instruct revision
    895c3a49bc3fa70a340399125c650a463535e71c;
  * complete local 13-file allowlist, offline-only loading, no
    trust_remote_code;
  * bitsandbytes NF4 4-bit, double quantization, float16 compute, SDPA, and one
    explicit CUDA device (never device_map=auto);
  * rank-16 LoRA on exactly 28 language q_proj, 28 language v_proj, and two
    verified visual-merger matrices: exactly 58 targets;
  * frozen vision backbone, at most four million trainable parameters;
  * batch one, accumulation 16, non-reentrant gradient checkpointing;
  * complete-image smart resize, at most 320 visual and 512 total tokens, with
    no truncation;
  * a real worst conservative image+label+draft forward/backward smoke;
  * peak allocated and allocator-reserved memory each strictly below 5.3 GiB.

The collator hard-fails any over-cap row. Model32 additionally includes draft
length when selecting its representative smoke row. The strict memory ceiling
is checked after the smoke and again against the complete training/inference
peak; it is never relaxed because this prompt is longer.

Static audit and future GPU training
------------------------------------
After creating the train draft receipt, a dry run verifies every file, hash,
ID, split, base checkpoint, starter provenance, and output destination without
loading Qwen or CUDA:

  PYTHONPATH=src .venv-vlm/bin/python model32/train.py --fold 0 --dry-run

When the GPU is free, run the mandatory smoke first:

  PYTHONPATH=src .venv-vlm/bin/python model32/train.py --fold 0 --smoke-only

Only a passing smoke permits training:

  PYTHONPATH=src .venv-vlm/bin/python model32/train.py --fold 0

Repeat folds 1..4 only if the fold-0 result is competitive. Each fold trains
on the other four folds and records that excluded fold in run_manifest.json.
Every epoch performs deterministic greedy generation over the whole held-out
fold and computes the exact challenge weighted WER/CER/multi-score.

Comparison policy: no row oracle
--------------------------------
For every validation epoch Model32 writes the Qwen challenger vector and one
unchanged return-the-Model8-draft vector. It reports both whole-fold exact
metrics and their multi-score delta. The emitted vector is always the complete
challenger vector. It never chooses Qwen for rows it got right and Model8 for
rows it got wrong. The diagnostic whole-fold winner is not a row-level gate.

The best adapter is chosen only by the full-fold challenger score, as in
Model27; the draft comparison does not alter any row. Before promotion, merge
all five independently generated OOF vectors and make one global choice
between the complete Model32 vector and the complete static Model8 vector.
Do not tune the prompt or training settings on the same five-fold score and
then describe that score as an untouched final estimate.

OOF and test inference
----------------------
After a fold run, regenerate its held-out predictions with strict adapter and
draft lineage checks:

  PYTHONPATH=src .venv-vlm/bin/python model32/infer.py \
    --input data/processed/train_folds.csv \
    --drafts outputs/model32/drafts/train_model8_raw.csv \
    --draft-receipt outputs/model32/drafts/train_model8_raw.receipt.json \
    --fold 0 \
    --adapter outputs/model32/fold0/best_adapter \
    --output outputs/model32/oof/fold0.csv

Fold F inference rejects every adapter except the Model32 fold-F adapter whose
manifest says F was excluded. Adapter files, training draft lineage, config,
implementation, best-epoch receipt, and memory smoke are all hash-checked.
The sidecar fold0.draft.csv is the fixed return-the-draft baseline, and
fold0.metrics.json contains the exact whole-fold comparison.

For one test candidate from fold 0:

  PYTHONPATH=src .venv-vlm/bin/python model32/infer.py \
    --input data/raw/Test.csv \
    --drafts outputs/model32/drafts/test_model8_raw.csv \
    --draft-receipt outputs/model32/drafts/test_model8_raw.receipt.json \
    --adapter outputs/model32/fold0/best_adapter \
    --output outputs/model32/test/fold0.csv \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model32/fold0.csv

Generate five separate test vectors and use a frozen whole-vector/ensemble
policy selected from OOF evidence, never leaderboard feedback or a hidden
per-row oracle. All mutable artifacts are restricted to outputs/model32 and
submissions/model32; Model22, Model25, and Model27 files are read-only.
