MODEL 31 — leakage-safe ByT5 OCR error corrector
=================================================

Status
------
Model31 is an implemented challenger, not a claimed score. The pinned local
snapshot and its isolated dependencies are installed and verified. A real
two-row forward/backward GPU smoke passed after explicitly initializing the
configured CUDA allocator: 593,920 trainable LoRA parameters, 0.877 GiB peak
allocated, and 0.885 GiB peak reserved. No fold training or submission has yet
been run.

Why this model
--------------
The visual OCR often produces near-correct historical English. ByT5 operates on
UTF-8 bytes and is therefore a natural spelling/punctuation correction model
without a subword vocabulary. It sees only an OCR string and emits a corrected
string; it never sees an image.

Leakage contract
----------------
Training input must be genuine complete OOF OCR, initially
outputs/model25/allfold_oof.csv. For an outer fold f:

  1. inner fold (f + 1) mod 5 is held out;
  2. the other three folds train the selection adapter;
  3. every epoch is greedily decoded and scored on that genuine inner fold;
  4. the epoch and a maximum source-to-candidate edit ratio are selected there;
  5. a fresh adapter is refit for that many epochs on all four folds != f;
  6. the frozen adapter and frozen gate are applied once to outer fold f.

Outer targets are read only after predictions have been written. The report
shows identity, raw-neural, and gated fold scores explicitly. Identity versus
neural is never chosen using an individual row's target. Combined OOF is valid
only after all five independent outer runs finish.

Optional train-side corruptions
-------------------------------
config.yaml preregisters deterministic synthetic OCR corruptions but leaves
them disabled for the first baseline. To test them, copy the config before any
run and set synthetic.enabled=true. Each selection/refit stage learns character
substitution, deletion, and insertion frequencies only from that stage's
genuine training OOF pairs. It never fits on inner or outer validation rows;
validation remains entirely genuine. The learned distributions and augmented
pairs are hashed in progress.json.

Base-model provenance
---------------------
Base: google/byt5-small
Pinned revision: 6f07f879d308b7b762708b50c83d41b27e329992
Licence: Apache-2.0
Model page: https://huggingface.co/google/byt5-small
Revision: https://huggingface.co/google/byt5-small/tree/6f07f879d308b7b762708b50c83d41b27e329992

That immutable revision is the safetensors conversion of canonical revision
68377bd. model.safetensors is pinned to 1,198,571,496 bytes and SHA256
1dfcb862fbe12b97e63e1cdb8b3920a97d0ef4ed3a11e431829ebeacac3fc026.
Loading forces HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1,
local_files_only=true, trust_remote_code=false, and use_safetensors=true. The
five required local files and architecture are checked before imports/model
loading; every file hash is copied into the run lineage.

The revision-pinned files have been fetched with this exact command shape:

  hf download google/byt5-small \
    config.json generation_config.json model.safetensors \
    special_tokens_map.json tokenizer_config.json \
    --revision 6f07f879d308b7b762708b50c83d41b27e329992 \
    --local-dir checkpoints/model31/google-byt5-small-6f07f87

Runtime and memory contract
---------------------------
Only rank-8 LoRA matrices train: q/v in 12 encoder self-attention blocks, four
decoder self-attention blocks, and four decoder cross-attention blocks (exactly
40 modules, at most one million trainable parameters). The base is float16;
batch size is 2, accumulation 8, and non-reentrant gradient checkpointing is
mandatory. A real worst-length forward/backward smoke must keep both allocated
and allocator-reserved memory strictly below 5.3 GiB before a fold run starts.

All audited labels are ASCII and <=120 bytes (maximum exactly 120); current
Model25 OOF sources max at 108. Training rejects anything over 120 bytes rather
than truncating. Test sources over the limit use identity. Empty, >120-byte, or
too-distant neural outputs also use identity. Token caps are 122 including
special tokens, and generation is deterministic greedy decoding.

CPU/static audit (works before dependencies or weights exist)
-------------------------------------------------------------
  PYTHONPATH=src python3 model31/train.py --fold 0 --audit-only

Repeat for folds 0..4. This prints disjoint split hashes, byte maxima, and the
synthetic-row plan without touching outputs or CUDA.

Environment and required smoke
------------------------------
Use `.venv-vlm`, which contains the existing torch 2.13.0 CUDA build and the
exact remaining pinned packages. The local snapshot byte count, weight hash,
architecture, and resolved package versions have passed verification:

  PYTHONPATH=src .venv-vlm/bin/python model31/train.py --fold 0 --smoke-only

Smoke writes outputs/model31/smoke.json. A changed config, source OOF, truth
CSV, base snapshot, or implementation invalidates it.

Five outer runs and exact OOF
------------------------------
  PYTHONPATH=src python model31/train.py --fold 0
  PYTHONPATH=src python model31/train.py --fold 1
  PYTHONPATH=src python model31/train.py --fold 2
  PYTHONPATH=src python model31/train.py --fold 3
  PYTHONPATH=src python model31/train.py --fold 4
  PYTHONPATH=src python model31/evaluate.py

Every epoch boundary has a safetensors adapter receipt plus hash-pinned
optimizer/scheduler/scaler state. --resume accepts only an exact run lineage
and a complete receipted epoch boundary. It refuses stale, modified, or
unreceipted artifacts and never overwrites a nonempty run.

Switching to Model22 OOF
------------------------
After complete genuine Model22 OOF exists, copy config.yaml, change both
data.train_predictions and output_dir, then repeat audit, smoke, all five folds,
and evaluation. Do not reuse a smoke receipt or Model31 run from Model25 input.

Test candidates and submission
------------------------------
Each outer adapter gives one independent correction of the complete Model25
test prediction file:

  PYTHONPATH=src python model31/infer.py \
    --run outputs/model31/fold0 \
    --output outputs/model31/test/fold0.csv

Repeat for fold1..fold4. Inference verifies the current base, config, fold-run
lineage, adapter receipt, input IDs, and output hash. A single candidate can be
written as a schema-validated submission by adding:

  --sample data/raw/SampleSubmission.csv \
  --submission submissions/model31_fold0.csv

For deployment, prefer the fixed Model26 five-source medoid over choosing a
fold from leaderboard feedback. Do not claim an OOF score for that five-model
test medoid: four adapters have seen each training row.
