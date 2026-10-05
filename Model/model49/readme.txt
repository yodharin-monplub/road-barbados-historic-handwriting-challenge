Model49 — frozen greedy no-repeat 4-gram versus 8-gram decoder

Purpose
  Compare two complete fold-0 Qwen2-VL prediction vectors from one completed
  Model39 best adapter.  Both policies are deterministic greedy decoding with
  num_beams=1, num_return_sequences=1, max_new_tokens=48,
  repetition_penalty=1.05, and use_cache=true.  The sole difference is
  no_repeat_ngram_size: 4 for the registered control and 8 for the challenger.
  No row-wise mixing, oracle choice, sampling, checkpoint change, epoch change,
  prompt change, image change, or post-processing is permitted.

Why test this
  A CPU audit of all 4,095 labels with the pinned Qwen tokenizer found 26
  targets containing a repeated token 4-gram.  Four are in fold 0, so the
  no-repeat-4 rule can make their exact truth structurally impossible to emit.
  One training target contains a repeated token 8-gram, but it is in fold 2;
  fold 0 has zero affected labels at n=8.  Therefore n=8 removes this avoidable
  fold-0 ceiling while max_new_tokens=48 still bounds runaway generation.  It
  is a challenger to measure, not an assumed improvement.

Fail-closed lineage
  config.json pins the Model39 config and runtime hashes, the exact fold CSV,
  fold 0, the expected 819-row vector, the local tokenizer/version audit, and
  outputs/model49.  A decode or readiness audit requires exactly
  outputs/model39/fold0/best_adapter plus both run_manifest.json and
  complete.json.  The completion receipt must bind fold 0, the same best
  adapter and best epoch, and all 615 optimizer updates.  Every adapter file is
  hashed before and after both vectors are generated.  Model49 loads that
  adapter once, then runs both policies through the same in-memory model.

Static CPU contract audit (allowed before training completes)
  .venv-vlm/bin/python model49/audit.py

  This validates pinned sources, data, checkpoint provenance, tokenizer label
  counts, and vector size.  Its status explicitly says the adapter was not
  checked and no score is claimed.  It cannot generate predictions.

CPU readiness audit after Model39 completes
  .venv-vlm/bin/python model49/decode.py --dry-audit \
    --adapter outputs/model39/fold0/best_adapter

  This fails if complete.json is absent or inconsistent.  It reads and hashes
  the adapter but never loads model weights and therefore does not use a GPU.

GPU decode comparison after the readiness audit passes
  .venv-vlm/bin/python model49/decode.py \
    --adapter outputs/model39/fold0/best_adapter \
    --output-dir outputs/model49/model39_fold0_best

  The new output directory receives both full CSV vectors, immutable receipts,
  exact whole-vector challenge metrics, and the challenger-minus-control score
  delta.  Model49 does not create a submission and never selects rows between
  vectors.  Do not run this command until the Model39 completion receipt exists.
