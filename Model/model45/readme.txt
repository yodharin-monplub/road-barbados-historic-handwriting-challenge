MODEL 45 — HIGH-RESOLUTION FULL-VISION QWEN + RECEIPTED OCR DRAFT

Status
------
Implemented and CPU-audited only. Model45 has no OOF, validation, public, or
private score. No CUDA process, smoke, training, inference, or submission was
started. The canonical Model32 draft receipts are incomplete (the interrupted
train attempt was archived outside the canonical draft directory), so every
GPU-capable entry point deliberately fails closed before model allocation.

High-upside hypothesis
----------------------
Model45 combines two complementary ideas. Model40 presents the complete crop
with an aspect-preserving 520–640-token letterbox and adapts all language and
all 32 vision blocks. Model32 supplies a raw, frozen Model8 transcription as an
untrusted clue. Fine visual detail should help Qwen reject incorrect OCR
characters, while the draft can provide spelling and line structure that the
visual model would otherwise need to rediscover.

The prompt retains Model40's instruction to transcribe every complete line in
reading order. It warns that the draft may cover only one line, so draft
anchoring cannot legitimately suppress other complete visible lines. Draft
replacement is literal, not str.format; braces or punctuation emitted by OCR
cannot trigger another interpolation. The assistant target remains exact and
only assistant tokens receive loss.

Pinned full-vision architecture
-------------------------------
Model45 hash-pins finalized Model40's config and implementation. It inherits:

  * Qwen/Qwen2-VL-2B-Instruct at revision
    895c3a49bc3fa70a340399125c650a463535e71c;
  * offline NF4, double quantization, float16 compute, SDPA, explicit one-GPU
    placement, non-reentrant checkpointing, batch one, accumulation 16;
  * rank-8 LoRA on 196 language matrices, 128 matrices across every visual
    block, and two visual-merger matrices;
  * exactly 326 targets, 652 LoRA tensors, and 14,610,432 trainable parameters;
  * full-image Lanczos letterboxing with no crop or glyph stretching, a
    configured 512–640 token range, and audited observed geometry 520–640;
  * learning rate 3e-5, three epochs, greedy max-48 generation,
    repetition_penalty 1.05, and no_repeat_ngram_size 4;
  * the corrected accumulation schedule: 205 updates per complete epoch and
    exactly 615 total, with boundaries required at 205, 410, and 615.

Strict label-independent draft lineage
--------------------------------------
The only accepted drafts are Model32's canonical raw Model8 outputs. Model45
hash-pins Model32's config/implementation and reuses its full receipt verifier.
The receipt proves that the frozen medium checkpoint saw an ordered ID-only
CSV plus image pixels, never Target; decoding was full-view, unconstrained,
batch-one, CPU float32, with no label-fitted postprocessor.

Every ordered ID, image hash, draft CSV, ID-only projection, confidence JSONL,
source manifest, generator config/code, checkpoint, and checksum must match.
Receipt schema v2 binds the count, ordered ID list, and digest for genuine
empty OCR outputs; a present blank remains blank between DRAFT markers.
Model25, charset-constrained output, partial artifacts, missing/reordered rows,
tampered blank ledgers, and label-derived corrections are rejected. Model41
output is not a draft dependency.

CPU audit now
-------------

  PYTHONPATH=src .venv-vlm/bin/python model45/audit.py \
    --output outputs/model45/dry_run.json

Without complete draft receipts, this still verifies the canonical data and checkpoint
hashes, all 326 safetensors target paths/shapes, 32,768-token model context,
all 5,472 image geometries, five disjoint 3,276/819 folds, and the 205/615
schedule. It records the exact missing draft sidecars and marks the guided
tokenizer audit as blocked; it never imports the CUDA model.

Generate canonical drafts on CPU
--------------------------------

  CUDA_VISIBLE_DEVICES='' .venv-kraken/bin/python model32/prepare_drafts.py \
    --input data/processed/train_folds.csv \
    --output outputs/model32/drafts/train_model8_raw.csv

  CUDA_VISIBLE_DEVICES='' .venv-kraken/bin/python model32/prepare_drafts.py \
    --input data/raw/Test.csv \
    --output outputs/model32/drafts/test_model8_raw.csv

Model32 refuses to overwrite existing artifacts. Only the train bundle is
needed to audit/smoke/train fold 0; test drafts are needed for test inference.

Sequence/context policy
-----------------------
Model45 does not guess a larger context budget. It retains Model40's 768-token
training cap and 816-token generation-context cap. Once a complete receipt is
available, the CPU audit tokenizes every guided row with the pinned tokenizer
and that row's exact planned letterbox token count. It records maximum training
sequence, prompt-plus-48 context, target length, and responsible IDs. If a cap
is insufficient, the dry report records the exact observed minimum and token
shortfall but marks CUDA blocked. Smoke/training still abort. A future cap
increase is forbidden unless that complete receipted all-row audit provides
the exact evidence; changing this config also changes its manifest hash and
therefore cannot silently reuse an adapter.

Before CUDA model allocation, the real processor independently encodes all
3,276 selected training examples and all 819 held-out prompts after actual
letterboxing. It rejects an image above 640 tokens, a training sequence above
768, or a prompt-plus-48 context above 816. The exact largest training sequence
becomes the memory-smoke example.

Mandatory real smoke
--------------------

  PYTHONPATH=src .venv-vlm/bin/python model45/train.py --fold 0 --smoke-only

This command cannot reach CUDA until the train draft receipt and both all-row
sequence audits pass. It then verifies 326 injected exact targets, 652 tensors,
14,610,432 trainable parameters, NF4 modules, full CUDA residency, and performs
a real forward/backward. Peak allocated and allocator-reserved memory must each
be strictly below 5.3 GiB. The same ceilings are enforced after full training.
The combination is substantially more memory-intensive than Model41; smoke
failure means Model45 is infeasible on this 6 GiB GPU and must not be bypassed.

Training and OOF inference
--------------------------
Only after a passing smoke:

  PYTHONPATH=src .venv-vlm/bin/python model45/train.py --fold 0

Each complete epoch greedily decodes all 819 held-out rows using the exact
challenge metric. The best adapter is selected only by the complete challenger
vector. The raw draft is reported as a whole-fold baseline and never selected
per row.

  PYTHONPATH=src .venv-vlm/bin/python model45/infer.py \
    --input data/processed/train_folds.csv \
    --drafts outputs/model32/drafts/train_model8_raw.csv \
    --draft-receipt outputs/model32/drafts/train_model8_raw.receipt.json \
    --fold 0 \
    --adapter outputs/model45/fold0/best_adapter \
    --output outputs/model45/oof/fold0.csv

Do not make a test submission unless genuine fold-0 evidence beats the safer
candidate. A single selected fold is conditional evidence, not unbiased OOF.

Principal risks
---------------
The 640-token grid plus longer guided prompt may breach sequence or VRAM limits.
The raw draft can anchor Qwen to systematic Model8 mistakes, and letterbox
upscaling can magnify paper noise. All-block vision LoRA may overfit 3,276 rows.
Finally, the “every complete line” prompt can conflict with annotations if the
labels intentionally represent only a central line. These are empirical risks;
architecture capacity alone is not a score claim.
