Model44 — frozen whole-vector decoding-policy challenger

Purpose
  Challenge a completed Model39 broad Qwen2-VL adapter using exactly two
  pre-registered decoding policies: the audited Model39 greedy baseline and
  organizer-starter-aligned 3-beam decoding.  This is a decoding comparison,
  not an ensemble: every policy produces one prediction for every row and no
  per-row choice, threshold, learned selector, or oracle mix exists.

Frozen panel
  greedy: num_beams=1, max_new_tokens=48, repetition_penalty=1.05,
          no_repeat_ngram_size=4, do_sample=false.
  beam3_starter_aligned: identical controls, num_beams=3.
  num_return_sequences=1 and use_cache=true are explicit.  No length penalty,
  early stopping or EOS override is added: those knobs are frozen omissions,
  so Model39's pinned base generation configuration supplies its normal token
  termination behavior without a post-hoc tuning degree of freedom.

Lineage and isolation
  config.json pins Model39 config, inference/runtime/schema/submission source
  hashes, raw input hashes, and output root.  A usable adapter must be exactly
  outputs/model39/foldN/best_adapter and have Model39 run_manifest.json plus
  complete.json binding that fold.  Model44 hashes all adapter files, its
  manifests, base checkpoint and source set both before and after generation;
  any change fails the TOCTOU gate.  It writes only new directories below
  outputs/model44 and records initial/final receipts, per-vector CSV hashes,
  whole-vector metrics when labels exist, and runtime.

CPU dry audit (no model/adapter load, no GPU)
  python model44/decode.py --dry-audit --input data/processed/train_folds.csv --fold 0

The dry audit deliberately says no score is claimed unless a completed adapter
is supplied.  It verifies Model39 provenance and the full local checkpoint.

GPU run after Model39 completes
  python model44/decode.py --input data/processed/train_folds.csv --fold 0 \
    --adapter outputs/model39/fold0/best_adapter --output-dir outputs/model44/fold0

The command emits greedy.csv and beam3_starter_aligned.csv, with separate
whole-vector metrics in report.json.  It does not declare a winner or create a
submission; selecting a deployment policy requires a separately pre-registered
out-of-fold protocol.
