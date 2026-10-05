MODEL 75 — GUARDED MODEL73 REMAINING-FOLD CONTROLLER
====================================================

Purpose
-------
Model75 prevents RunPod idle gaps while completing Model73 folds 2–4 under
the checkpoint-400 policy frozen on fold 0 and confirmed on fold 1. For each
fold it verifies the complete checkpoint file set and trainer state, validates
the exact trainer PID/command, sends SIGTERM only to that trainer, waits for a
clean stop, decodes the heldout fold, creates a Model74 signed score receipt,
and only then starts the next fold.

Safety
------
The controller fails closed on a missing/mismatched process, incomplete
checkpoint, existing prediction destination, failed inference, or failed
hash-bound scoring. It never creates a submission and never deletes the Pod.
Pod deletion remains a separate action after local artifact recovery.

Status
------
Created after Model73 scored 0.8906962580 on fold 0 and 0.8954270287 on the
independent fold-1 confirmation. The controller completed folds 2--4, produced
all three signed Model74 receipts, wrote
`outputs/model75_folds_2_4_complete`, and exited. It never submitted or deleted
the pod.
