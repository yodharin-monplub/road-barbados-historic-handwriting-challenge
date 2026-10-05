Model121: unchanged beam-pair confirmation on folds1-4
===================================================
Research continuation, NOT a claim that Model120 passed its original gate.
Model120 scored+0.000521 vs original Model112 but+0.000474 vs matched greedy,
missing its required+0.0005 on both. Both individual OCR members improved
substantially with beam3. This borderline result motivates an explicitly
documented follow-up rather than more tuning on fold0 or immediate submission.

Freeze exactly Model120's recipe: both Model73 and Model107 decoded with3 beams,
penalty1.05; retain Model92 and Kraken greedy predictions, Model104 anchor and
Model112 family-word vote. No member-subset search. Re-decode both members
greedily on the same local runtime as the paired baseline for each fold.
Use each fold's own adapter verified against its original OOF receipt.
No retraining or external/test data. Configuration files are the pinned Model120
configs; the fold and adapter are explicitly supplied by the Model121 runner.

Run16 passes:4 folds x2 members x2 decoders. Evaluate all3276 confirmation rows
after completion; do not tune/stop based on intermediate fold scores.
Primary promotion gate frozen before new predictions:
- aggregate gain>=0.0005 vs BOTH original Model112 and local-greedy pair;
- nonnegative beam gain on at least3 of4 folds vs EACH baseline;
- paired3000-resample row-bootstrap95% lower bound>0 vs EACH baseline.
Bootstrap seed20260921. This measures conditional uncertainty, not clean
generalization: folds have upstream recipe/selector selection history and
individual rows need not be independent manuscript pages. Fold0 is excluded
from the confirmation aggregate and bootstrap.

No automatic test predictions/submission, even if confirmation passes. Keep
best public Model1120.911129378 untouched. Broad goal remains unachieved.
Local CUDA only; expected roughly2.5hours; no rented GPU or cloud charges.
Each pass waits safely for CUDA; background process checks every600s.

Run with .venv-lighton/bin/python:
  model121/experiment.py prepare
  model121/experiment.py check
  scripts/run_model121.py
Progress: outputs/model121/status.json. Results: outputs/model121/report.json.

Driver-update interruption (2026-09-11)
--------------------------------------
Three passes completed safely: fold1 Model73 greedy/beam and Model107 greedy.
While the third process was running, installed NVIDIA libraries changed from
610.43.02 to610.57.04 while the loaded kernel module remained610.43.02.
Existing inference completed using its mapped old libraries. New CUDA startup
failed with error803; nvidia-smi failed with NVML mismatch and exit18.
The system reports a restart required. No reboot or driver change was performed
by this experiment. Supervisor stopped before fold1 Model107 beam began.
Preserve all completed files and receipts. After the user reboots, recheck CUDA
and source hashes. For a matched driver control, preserve/archive and re-decode
fold1 Model107 greedy under the new driver before its beam pass. The already
finished Model73 greedy/beam pair used the same driver and need not be repeated.
Record the driver transition in a recovery receipt; do not claim every fold
ran under one identical driver. No full confirmation result exists yet.

Prepared recovery command (only after CUDA works again):
  .venv-lighton/bin/python scripts/resume_model121.py --run
Without --run it performs read-only file/protocol checks. With --run it first
checks no original supervisor/GPU task is live, nvidia-smi works, and an actual
CUDA tensor allocation/readback succeeds. Only then does it archive (not delete)
the third pass's4 artifacts under pre_driver_restart, save a recovery receipt,
repeat that greedy pass under the new driver and finish the remaining work.
The completed Model73 pair is kept. No reboot or driver installation is done.
