Model51 - exact tail-logit high-resolution Qwen2-VL training
================================================================

Status
------

Model51 is implemented; its CPU audits and mathematical equivalence tests pass.
Its real worst-case CUDA smoke also passed at 4.421 GiB peak allocated / 4.584
GiB peak reserved. A two-step uncheckpointed timing pilot took about 75
seconds/update and was stopped in favor of the geometry-adaptive Model50. It
has no held-fold OOF or Zindi score and is not a submission candidate.

Controlled change from Model43
------------------------------

Model51 keeps Model43's complete learning experiment unchanged:

* the pinned Apache-2.0 Qwen2-VL-2B-Instruct checkpoint;
* the 896..1024 merged-token, aspect-preserving full-image letterbox policy;
* rank-16/alpha-32 LoRA on exactly seven projection families in all 28 language
  layers: 196 modules, 392 tensors, and 18,464,768 trainable parameters;
* the fully frozen 32-block visual tower and frozen visual merger;
* batch size 1, accumulation 16, learning rate 3e-5, cosine schedule, three
  epochs, non-reentrant gradient checkpointing, and paged 8-bit AdamW;
* full determinism and the exact 205/615 optimizer-update schedule;
* the organizer starter prompt and full-fold greedy validation policy.

The only change is how training logits are materialized before the same shifted
cross entropy. Validation, generation, adapter serialization, and inference use
the ordinary full LM head with no hook.

Why the stock loss wastes work
------------------------------

The pinned Transformers 4.49.0
Qwen2VLForConditionalGeneration.forward signature has neither logits_to_keep
nor num_logits_to_keep. Its stock implementation always:

1. projects every final hidden position to the 151,936-token vocabulary;
2. upcasts every logit to float32;
3. creates a contiguous one-position-shifted logit tensor; and
4. asks cross entropy to ignore all prompt labels marked -100.

The Model43 worst training sequence is 1,115 positions, but only the final 36
positions are supervised. Projecting prompt logits cannot affect the loss or
gradient because their shifted labels are ignored.

Exact 37-position construction
------------------------------

All 4,095 labels were encoded through the pinned chat template. Every full
training template has the generation prompt as an exact prefix, and its
supervised assistant suffix contains:

  minimum: 8 tokens
  median:  17 tokens
  mean:    17.6383394383 tokens
  p95:     25 tokens
  maximum: 36 tokens

Because causal language modelling shifts logits by one position, N terminal
labels need N+1 trailing hidden positions under the stock loss. Model51
therefore keeps a fixed 37-position label window. The first position is always
ignored; the remaining 36 positions cover the maximum possible suffix.

The CPU collator fails closed unless every batch has size one, at least one
supervised token, no supervised token outside the last 36, and one contiguous
supervised suffix ending at the final sequence position. These data-dependent
checks happen before transfer to CUDA, so they add no per-batch GPU
synchronization. A transient pre-hook then passes only the final 37 hidden
positions into the unchanged tied LM head. The hook is removed immediately
after that one training forward.

Equivalence evidence
--------------------

The unit tests compare the stock full-head loss against Model51 for suffixes of
1, 8, 17, and 36 tokens. They verify matching selected logits, scalar loss,
input-hidden gradients, and LM-head weight gradients within floating-point
tolerance. Separate tests prove that non-contiguous, empty, oversized, and
batch-size-two labels fail; the hook is removed even after use; and an ordinary
eval forward still emits full-sequence logits.

This is the same mathematical objective. A differently shaped GEMM can select
a different kernel and therefore need not be bit-identical on CUDA. The held
fold remains the authority for model quality.

Static tensor-size audit
------------------------

At the conservative 1,115-position maximum and vocabulary 151,936, the source
audits these materialized tensor sizes:

                                      stock head       Model51 head
  projected positions                    1,115                  37
  shifted loss positions                 1,114                  36
  float16 logits                   338,817,280 B        11,243,264 B
  float32 logits                   677,634,560 B        22,486,528 B
  float32 contiguous shift         677,026,816 B        21,878,784 B

The projected-position count falls by 96.6816%. These are exact tensor byte
counts from the pinned source dimensions, not a prediction of allocator peak or
wall-clock speed. PyTorch can release, alias, or reuse storage, and transformer
and vision work remains unchanged. Only the real CUDA smoke can establish the
actual memory reduction; only measured update timing can establish speed.

Isolation and provenance
------------------------

Model43 was active when Model51 was created. Its source was not edited and is
pinned here at SHA256:

  4da6449f5a4241deaa803305b53013480012717d0ff1b54100da2d7eb89aebda

Model51 owns a separate config, source module, wrappers, smoke/output paths,
manifest fields, and tests. It imports Model43's audited model-loading,
geometry, scheduling, and validation utilities read-only. The transient hook
does not wrap or rename the tied output module, so saved PEFT adapters retain
the standard structure and load through the unmodified inference path.

Recoverability warning
----------------------

Like Model43, save_strategy is deliberately "no" and held-fold validation runs
only after each complete epoch. Before an epoch finishes validation and saves a
new best_adapter, the run_manifest is the only artifact; there is no optimizer,
scheduler, RNG, or adapter checkpoint from which to resume. Stopping a run
before that boundary discards its training progress.

CPU audit and tests
-------------------

From the repository root:

  PYTHONPATH=src .venv-vlm/bin/python model51/audit.py
  PYTHONPATH=src .venv-vlm/bin/python -m unittest \
    tests.test_model51_qwen_starter_highres_efficient -v

The machine-readable receipt is outputs/model51/dry_run.json. It records the
full 4,095-row suffix audit, actual worst processor encode, Qwen forward
signature and source hash, static materialization counts, geometry, folds, and
schedule. All Model51-specific tests pass in the pinned VLM environment.

Mandatory real GPU smoke
------------------------

Run only when the GPU is free:

  PYTHONPATH=src .venv-vlm/bin/python model51/train.py --fold 0 --smoke-only

The smoke loads the real NF4 model, repeats all target/residency audits, encodes
the worst fold-0 training record, exercises Model51's actual 37-position head
through forward and backward, and fails unless both peak allocated and peak
reserved memory are strictly below 5.3 GiB. Do not bypass or raise this limit.

The smoke does not prove a speedup. If it passes, compare several steady-state
optimizer updates against Model43 under the same thermal/power conditions
before deciding whether a restart is worth losing uncheckpointed progress.

Training after a successful smoke
---------------------------------

  PYTHONPATH=src .venv-vlm/bin/python model51/train.py --fold 0

At epoch boundaries the normal validation callback performs full-head greedy
generation over all 819 held rows, writes exact challenge metrics, and saves the
best standard PEFT adapter. Training completion additionally checks that the
tail loss ran exactly 9,828 times, once for every row presentation.

Inference from a completed adapter
----------------------------------

  PYTHONPATH=src .venv-vlm/bin/python model51/infer.py \
    --input data/processed/train_folds.csv \
    --adapter outputs/model51/fold0/best_adapter \
    --output outputs/model51/fold0/oof.csv \
    --fold 0

Test inference accepts the same --sample and --submission arguments documented
for Model43. Inference never installs the tail hook.
