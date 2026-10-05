Model107: automatic, fold-isolated StackMix handwriting augmentation
===================================================================

Recombine whole-word crops from the provided training images, preserving exact
historical text, punctuation and capitalization. A CTC Viterbi alignment locates
characters; crops with minimum character confidence <0.5 or mean confidence
<0.95 are rejected. No manual word boxes, transcriptions or targeted edits.
The aligner checkpoint, crop sources and target text must all exclude the
validation fold. Training excludes the existing known-bad IDs without changing
the supplied CSV. No test images enter this method.

Organizer permission:
https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34514

Pilot commands (from workspace root):
  .venv-kraken/bin/python model107/prepare.py bank --device cuda --limit 400 \
    --output outputs/model107/fold0/bank
  .venv-kraken/bin/python model107/prepare.py synthesize \
    --bank outputs/model107/fold0/bank/bank.json --count 200 \
    --output outputs/model107/fold0/pilot

For the complete bank omit --limit (previous verified rows are reused).
For training generate 819 lines (3276 real / 4), giving 80% real / 20% synthetic
dataset rows under the normal seeded training shuffle. Sampling is approximate
per batch, not a guarantee of an exact ratio in each individual optimizer step.
The LightOnOCR baseline, seed, augmentation and 400-step training budget are fixed.
Fold 0 is a development screen, not untouched confirmation data.
Require +0.001 over Model92 before expanding to other folds. Do not promote a
new submission unless its cross-fitted ensemble gains >=0.0005 over Model104
with a positive paired-bootstrap confidence interval.

Every generated row records its original target ID, donor IDs, text spans,
crop boxes, hashes and seed. These are checked again before training.
The pilot has no CV or leaderboard score until OCR training/evaluation finishes.

Initial pilot (2026-09-07)
-------------------------
CUDA was verified outside the restricted sandbox: RTX 4050 Laptop GPU, 6 GB.
The first 400 eligible training lines yielded 3030 accepted word crops and
1445 low-confidence word rejections. Recombination produced 200 lines covering
195 eligible target transcripts. An automatic same-aligner round-trip check
read 49/200 lines exactly, with no blanks and diagnostic multi_score 0.90811645.
This is NOT a CV score, is not independent of the aligner, and does not prove
that all synthetic labels are correct. It is a pipeline consistency check.

Training commands after the complete bank is prepared:
  .venv-kraken/bin/python model107/prepare.py synthesize \
    --bank outputs/model107/fold0/bank/bank.json --count 819 \
    --output outputs/model107/fold0/training
  .venv-lighton/bin/python model107/train.py --prepare-only
  .venv-lighton/bin/python model107/train.py
  .venv-lighton/bin/python model107/infer.py \
    --adapter outputs/model107/train/fold0/checkpoint-400 \
    --input data/processed/train_folds.csv --fold 0 \
    --output outputs/model107/fold0_oof.csv
  python3 model107/evaluate.py --predictions outputs/model107/fold0_oof.csv \
    --output outputs/model107/fold0_report.json

The shared VLM loader accepts an optional synthetic_manifest parameter; normal
models without it retain their previous data path. Validation explicitly rejects
synthetic manifests. The 200-row pilot is not accepted as the 819-row train mix.

Complete bank: 3276 eligible lines, 24840 accepted word crops, 12067 rejected
low-confidence words, 3 unalignable lines. The 819-line training augmentation
covers 1289 fully synthesizable target transcripts. Twelve alignment/provenance
tests, three shared VLM checkpoint tests, thirteen Kraken regression tests and
two Vast stop-state tests passed before cloud launch.

Cloud supervision: scripts/monitor_model107.py checks every 600 seconds, copies
the fold-0 prediction files, training manifest and final adapter, verifies their
hashes and the 3276/819 training mix, stops Vast instance 50076619, and scores the
development fold locally. The PC must remain awake and online. Logs/status:
outputs/model107/supervisor.log and supervisor_status.json.

Model107 fold-0 training was launched on the existing Vast RTX 4090 after all
835 uploaded code/artifact files passed hash verification. A real Hugging Face
dataset-load test confirmed 4095 training rows (3276 original plus 819 synthetic)
and 819 validation rows, and decoded a synthetic image successfully. The local
service road-model107-supervisor.service completed successfully; the instance
was stopped after all downloaded artifacts passed verification.

Fold-0 result: 0.8950117648 versus Model92 0.8924811370 (+0.0025306277).
This passes the +0.001 development gate. Confirmation folds 1-4 now use the
same frozen recipe and their own fold-excluded Model36 epoch-7 aligners.
Run model107/prepare_remaining.py locally before starting rented compute.
No full-OOF improvement or new submission is claimed until confirmation passes.

Recovery and conditional completion (2026-09-07)
------------------------------------------------
An SSH launch AND-list retained its output pipes and timed out, although the
four-fold GPU job continued. scripts/resume_model107.py now attaches without
retraining, checks every 600 seconds, downloads each finished fold, verifies
hashes/provenance, and stops the instance after completion. The future launcher
has been corrected to detach its streams properly.

Deployment-only augmentation uses a separately proven all-data aligner and is
never used in CV. It contains 4095 real / 1023 synthetic examples. The final
500-step refit requires a passing full-OOF promotion_decision.json. The optional
scripts/deploy_model107.py --wait service enforces that gate, verifies downloaded
test predictions and adapter, stops the instance, and prepares (does not upload)
submissions/model107_stackmix_selector.csv. Existing submissions are preserved.

Current monitoring files: outputs/model107/confirmation_status.json and
deployment_status.json. Keep the local PC awake for cloud supervision.
