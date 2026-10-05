MODEL 41 — BROAD QWEN2-VL + LABEL-INDEPENDENT OCR DRAFT

Status
------
Implementation and CPU/header audits are available. The canonical Model32
draft CSVs and receipts have not been generated yet, so Model41 deliberately
has no OOF score. A structural audit may pass while reporting that blocker;
smoke, training, and inference fail closed until every draft receipt verifies.

Hypothesis and isolated comparison
----------------------------------
Model39 supplies much more adaptation capacity than Model27: rank-8 LoRA on
all 196 language projections, the four projections in each of visual blocks
24..31, and both visual-merger matrices (230 exact targets, 10,678,272
trainable parameters). Model41 keeps that entire audited runtime and adds the
raw zero-shot Model8 OCR string as an explicitly untrusted hint. This combines
broad language correction and visual grounding while asking Qwen to spend its
capacity on ambiguities in a plausible first draft.

Image geometry stays at 320 merged visual tokens. Training hyperparameters,
seed, three epochs, corrected ceil accumulation schedule (205 updates/epoch,
615 total), and generation controls (48 new tokens, greedy, repetition 1.05,
no-repeat 4-gram) are inherited exactly from Model39. Thus fold 0 is an
interpretable prompt-only comparison against Model39 rather than a geometry or
optimizer confound.

Leakage and receipt contract
----------------------------
The only accepted draft source is Model32's frozen Model8 medium checkpoint,
run once on the full image with raw unconstrained greedy decoding. Its
inference input is physically projected to the ordered ID column, so no Target
field reaches the OCR process. No label-fitted postprocessor, Model25 output,
charset constraint, outer-fold model, row oracle, or leaderboard choice is
allowed.

Model41 hash-pins the complete Model39 config/code and Model32 draft config/code.
It re-verifies Model32's receipt, including exact ordered IDs, source-image
hashes, draft CSV, ID-only source CSV, confidence JSONL, source manifest,
generator config/code, frozen checkpoint, and all checksums. Receipt schema v2
also binds every genuine blank through its count, ordered ID list, and digest.
The prompt inserts the draft by one literal replacement; OCR braces cannot
trigger formatting, and a present blank remains blank between the markers.

Structural audit now (CPU only)
-------------------------------
This imports neither the Qwen CUDA model nor PEFT. It verifies the local
checkpoint headers, all 230 target shapes, 10,678,272 calculated parameters,
canonical data hashes, five disjoint folds, and the ceil optimizer schedule.
It also lists each missing draft-sidecar path without pretending Model41 is
trainable:

  PYTHONPATH=src .venv-vlm/bin/python model41/audit.py \
    --output outputs/model41/dry_run.json

The same fold-specific structural audit is available through:

  PYTHONPATH=src .venv-vlm/bin/python model41/train.py --fold 0 --dry-run

Generate the canonical drafts (CPU only)
----------------------------------------
Use Model32's single canonical producer; Model41 never invents a weaker second
receipt format:

  CUDA_VISIBLE_DEVICES='' .venv-kraken/bin/python model32/prepare_drafts.py \
    --input data/processed/train_folds.csv \
    --output outputs/model32/drafts/train_model8_raw.csv

  CUDA_VISIBLE_DEVICES='' .venv-kraken/bin/python model32/prepare_drafts.py \
    --input data/raw/Test.csv \
    --output outputs/model32/drafts/test_model8_raw.csv

Existing draft artifacts are never overwritten. Draft generation can be slow
on CPU. Only the train receipt is needed for fold training; the test receipt is
needed later for test inference.

Exact sequence and memory gates
-------------------------------
Before allocating the training model, Model41 encodes every one of the 3,276
selected training rows and every held-out prompt with the real processor. Any
row above 320 image tokens or 512 total tokens aborts. It records maxima and
the exact ID attaining each. It then audits the instantiated 230-target LoRA,
NF4 quantization, CUDA residency, and exact trainable tensor/parameter counts.

A real forward/backward on the exact largest processor-emitted training
sequence (image+prompt+draft+label) must remain strictly
below both 5.3 GiB allocated and 5.3 GiB reserved. Complete-run peaks are
checked again. This larger adapter has not yet passed that GPU smoke, so run:

  PYTHONPATH=src .venv-vlm/bin/python model41/train.py --fold 0 --smoke-only

Fold-0 training and OOF inference
--------------------------------
Only after the smoke passes:

  PYTHONPATH=src .venv-vlm/bin/python model41/train.py --fold 0

Every epoch performs exact deterministic greedy generation on all 819 held-out
rows. Best-adapter selection uses only the complete challenger vector's exact
challenge multi-score. The raw Model8 draft vector is reported as a whole-fold
baseline and is never selected row by row. The ceil schedule guard rejects a
204-step pseudo-epoch or any run that does not finish at steps 205/410/615.

Regenerate the fold-0 OOF vector with adapter and receipt lineage checks:

  PYTHONPATH=src .venv-vlm/bin/python model41/infer.py \
    --input data/processed/train_folds.csv \
    --drafts outputs/model32/drafts/train_model8_raw.csv \
    --draft-receipt outputs/model32/drafts/train_model8_raw.receipt.json \
    --fold 0 \
    --adapter outputs/model41/fold0/best_adapter \
    --output outputs/model41/oof/fold0.csv

Test candidate after OOF promotion
----------------------------------
Do not make a submission from this model until fold-0 evidence beats the safer
candidate. For a test vector from the fold-0 adapter:

  PYTHONPATH=src .venv-vlm/bin/python model41/infer.py \
    --input data/raw/Test.csv \
    --drafts outputs/model32/drafts/test_model8_raw.csv \
    --draft-receipt outputs/model32/drafts/test_model8_raw.receipt.json \
    --adapter outputs/model41/fold0/best_adapter \
    --output outputs/model41/test/fold0.csv \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model41/fold0.csv

Main risks
----------
The raw Model8 draft may anchor Qwen to systematic OCR errors, and the longer
prompt may exceed the 512-token cap for pathological drafts. The broad adapter
also has a materially higher VRAM footprint than Model27 and is not promoted
until the strict smoke succeeds. Finally, one fold is selection evidence, not
an unbiased five-fold estimate; train folds 1..4 only if fold 0 shows a clear,
credible gain over Model39 under the exact same evaluation protocol.
