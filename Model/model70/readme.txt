Model70 - monitored continuation for the Model36/35 cross-fold candidate

Purpose
-------
Model70 prevents idle gaps in the long CUDA experiment. It attaches to the
already-running fold-2 training job, waits for exactly eight checkpoints and
CUDA release, then performs the remaining work in this fixed order:

1. Exact checkpoint evaluation for fold 2.
2. Fixed epoch-7 all-row inference and concurrent Model35 CPU postprocessing.
3. Eight-epoch CUDA training, evaluation, inference, and CPU postprocessing for
   folds 3 and 4.
4. Provenance-verified five-fold Model35 combination.
5. Model69 leave-one-fold-out checkpoint selection.

Safety and scope
----------------
Every destination must be new, the fold-2 attachment times out after two hours,
the checkpoint count must equal eight, and any failed subprocess stops later
stages. The runner does not train an all-data deployment model, alter the
frozen checkpoint or terminal-stage policies, create a test submission, access
the network, or upload to Zindi.

Run
---
  env PYTHONUNBUFFERED=1 PYTHONPATH=src .venv-kraken/bin/python \
    model70/orchestrate_remaining_folds.py --attach-fold 2

Recovery after an interrupted fold-3 process
--------------------------------------------
Preserve the incomplete output directory under a new archival name, then use:

  env PYTHONUNBUFFERED=1 PYTHONPATH=src .venv-kraken/bin/python \
    model70/orchestrate_remaining_folds.py --restart-fold 3

This mode requires the canonical fold-3 output directory to be absent. It
reruns fold 3 deterministically from the pinned base rather than treating a
weights-only restart as equivalent to the original optimizer schedule.

Status
------
Implemented while Model36 fold 2 was training. Syntax and CLI are checked
before launch. The original attached process was found stopped after
fold-3 epoch 6; the cause was not established. Its seven checkpoints are
retained as an interrupted audit, and
the clean restart mode avoids silently changing the optimizer schedule.
Model70 is orchestration and does not itself claim an OOF score; its outputs
remain attributable to Models35, 36, and 69.

2026-09-05 recovery progress
----------------------------
The clean fold-3 restart completed all eight checkpoints and exact checkpoint
evaluation. Frozen epoch 7 scored 0.808359660 raw OOF on 819 rows. Its all-row
inference then completed successfully. Model70 started fold-3 Model35 CPU
postprocessing and fold-4 Model36 GPU training concurrently. These active jobs
must finish before the five-fold combine and Model69 exploratory analysis.

Final deployment continuation, 2026-09-05
-----------------------------------------
The cross-fold pipeline completed successfully. The frozen Model35 confusions
stage scores 0.841721630 over all five held-out folds and passes the strict
global, confirmation-fold, and per-fold gate against Model66. The promoted
all-data Model36 run was then started separately in screen session
road_model36_all_data.

orchestrate_deployment.py can bind to that exact live wrapper PID and wait for
its /proc start identity to disappear. It then re-runs the OOF gate, verifies
and exports only fixed epoch 7, infers all training and test rows from the same
weights, and invokes Model35's guarded builder. All downstream destinations
must be fresh and every existing CLI remains fail-closed. The runner performs
no upload and contains no network or Zindi command.

  env PYTHONUNBUFFERED=1 PYTHONPATH=src .venv-kraken/bin/python \
    model70/orchestrate_deployment.py --training-pid <all-data-wrapper-pid>

Its status is written atomically to
outputs/model70/deployment_continuation_status.json. Its final successful
state is validated_candidate_not_uploaded.

Deployment completion, 2026-09-05
---------------------------------
The continuation completed every guarded step and exited successfully. The
atomic status is validated_candidate_not_uploaded, and the log ends with
"DEPLOYMENT CONTINUATION COMPLETE; NOTHING UPLOADED". It produced and validated
outputs/model35/deployment_epoch07/submission.csv with SHA256
d676ac5dffb6bd37b5614243b4b1acb1ddadaadbee11b6ac379d68513caf2b3c.
The final 38-test deployment-critical suite passed after artifact creation.
