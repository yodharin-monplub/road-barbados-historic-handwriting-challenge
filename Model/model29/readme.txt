MODEL 29 — leakage-safe PP-OCRv6 CTC prefix beam + challenge-text LM

Status
------
Implementation and synthetic tests are complete. No GPU job has been started,
no installed Kraken package was patched, and nothing under outputs/model22 is
written. Genuine Model29 OOF scoring starts only after each Model22 fold has a
selected converted .safetensors checkpoint and its probability export exists.

What this model changes
-----------------------
Model22's standard inference remains untouched and greedy. Model29 is a
separate, opt-in decoder experiment. Its exporter installs a project-side
decoder callback for one process: Kraken computes its normal temperature-
scaled softmax, the callback copies the full [time, classes] matrix to CPU,
then delegates the original objects to Kraken's stock greedy decoder. This
avoids a second forward pass and produces the same ordinary Kraken record.

Do not use bbox record.logits for this purpose. In Kraken 7.1 that field holds
decoded tuples for bbox lines, not the full matrix. Model29 also fixes capture
batch size at one because the decoder callback has no image IDs and Kraken may
reorder extracted lines in a multi-line batch.

Every ID receives a compressed float32 NPZ containing every timestep and every
codec class (class 0 is blank). Per-ID atomic receipts record ID, shape,
source-image hash, and SHA-256. The final index and manifest bind the input CSV,
checkpoint, verified Model22 training/conversion lineage, configuration,
implementation, ID order, and exact class-to-text mapping. Every row also
checks that independently collapsing the captured tensor reproduces Kraken's
greedy record exactly. Interrupted exports resume only when that complete
signature still matches.

Leakage-safe OOF protocol
-------------------------
The immediately runnable protocol freezes ``default_fixed_candidate`` in
config.yaml before inspecting Model29 OOF. For outer fold f, its character LM
is fitted on every fold except f, so changing labels in f cannot change the LM
or predictions applied to f (covered by a unit test).

Naively tuning candidates on ordinary OOF stores is not fully safe: fold-g's
checkpoint trained on fold f and can carry information from f into its tuning
emissions. Model29 refuses that shortcut. Optional tuned mode requires a
separate store for every ordered (f,g) pair, produced by a checkpoint whose
signed provenance says both f and g were excluded from model training. Its LM
also excludes f and g. Only then are candidates scored on g != f and the final
LM fitted on all folds except f.

The fixed five-member grid includes untouched greedy decoding, acoustic-only
prefix beam search, two character-LM weights, and one optional word-boundary
prior. Decoding keeps only the eight highest-probability classes plus a
guaranteed blank per timestep and twelve prefixes, so memory is bounded even
though the exported artifact retains full probabilities. Ties prefer the
earlier declared candidate. Exact challenge WER, CER,
and normalized multi-score are computed globally and per fold using the shared
competition_error implementation.

Step 1: export each held-out fold
--------------------------------
Wait for Model22 checkpoint evaluation and use a converted checkpoint whose
epoch-selection rule was frozen without consulting the fold being exported
(for example, choose fold f's epoch from the other folds' epoch results). Do
not use each fold's own best validation epoch for a claimed unbiased OOF score.
Run once per fold, changing the checkpoint, receipts, fold, and destination.
Example for fold 0:

  .venv-kraken/bin/python model29/export_probabilities.py \
    --input data/processed/train_folds.csv \
    --checkpoint outputs/model22/checkpoint_evaluation/fold0/converted/SELECTED_CHECKPOINT.safetensors \
    --training-manifest outputs/model22/fold0/run_manifest.json \
    --conversion-receipt outputs/model22/checkpoint_evaluation/fold0/converted/SELECTED_CHECKPOINT.conversion.json \
    --training-excluded-fold 0 \
    --fold 0 \
    --output-dir outputs/model29/probabilities/fold0

Replace the placeholder with the externally selected converted weight path;
the matching conversion receipt has the same stem. The exporter verifies that
the source checkpoint belongs to the stated Model22 fold run and that training
excluded the exported fold. It cannot prove the statistical independence of
an operator's epoch-selection rule, so record that rule separately. Repeat for
folds 1 through 4.
A non-mutating compatibility check is available by appending --dry-run.

Step 2: exact fixed-candidate OOF decode
----------------------------------------
Once outputs/model29/probabilities/fold0 through fold4 are complete, this runs
the prespecified ``char4_a004`` candidate and reports it against greedy:

  .venv-kraken/bin/python model29/decode_oof.py \
    --truth data/processed/train_folds.csv \
    --probabilities-root outputs/model29/probabilities \
    --output-dir outputs/model29/oof_fixed_char4

The destination must be new or empty and cannot be under outputs/model22.
Artifacts are predictions.csv, greedy_predictions.csv, report.json, receipt.json,
and one full selection audit per outer fold under folds/. Promote Model29 only
if the held-out global gain is positive, fold behavior is acceptable, and the
receipt checks all five genuine OOF probability stores.

To override the prespecified candidate without claiming an unbiased choice,
add for example:

  --fixed-candidate char4_a004

Optional true nested tuning scaffold
------------------------------------
This remains staged because producing 10 unordered pair-excluded inner PP-OCR
checkpoints (and exporting their 20 ordered outer/tuning-fold stores) is a
separate GPU experiment. Each store at
``inner_probabilities/outerF/foldG`` must come from a checkpoint whose training
manifest records both F and G in ``preparation.excluded_folds`` and must pass
both values as repeated ``--training-excluded-fold`` export arguments. Once
those stores exist, the exact remaining CPU command is:

  .venv-kraken/bin/python model29/decode_oof.py \
    --truth data/processed/train_folds.csv \
    --probabilities-root outputs/model29/probabilities \
    --tune \
    --tuning-probabilities-root outputs/model29/inner_probabilities \
    --output-dir outputs/model29/oof_nested_tuned

The decoder fails closed if any inner checkpoint did not exclude both its
outer and tuning fold. Ordinary five-fold OOF stores cannot be substituted.

Run only one exporter process per destination directory. Individual files are
written atomically and interrupted work is resumable, but concurrent writers
would duplicate expensive inference and are intentionally unsupported.

Deployment boundary
-------------------
Do not build a test submission before OOF approval. After approval, freeze one
candidate from the OOF evidence, train/select the all-data Model22 checkpoint
without leaderboard tuning, export Test.csv probabilities by omitting --fold,
and decode with an LM fitted on all permitted training labels. A final-test
decoder is intentionally not run or claimed here while five-fold OOF evidence
is unavailable.
