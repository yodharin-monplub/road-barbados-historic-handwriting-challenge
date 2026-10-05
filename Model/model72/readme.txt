MODEL 72 — FOLD-SAFE CORRECTION FOR MODEL71
===========================================

Purpose
-------
Model72 applies the established same-fold-model correction protocol to the
native-height Model71 recognizer. For held-out fold f, the exact Model71 fold-f
weights first predict all training images. Correction recipes are fitted only
from folds other than f and are then applied once to fold f. The held-out
labels are used only to score the resulting predictions.

Selection policy
----------------
The Model71 epoch is selected once from the exact raw challenge metric on fold
0 after all ten checkpoints have been decoded. Model72 may then measure the
fixed context -> boundaries -> confusions recipe on that same fold. The epoch
and terminal-stage policy must be frozen before folds 1--4. Per-fold epoch or
terminal-stage selection is prohibited.

Provenance
----------
run_fold.py uses the same fail-closed lineage verifier as Model35. Model71 is
an explicitly supported OCR family, and outputs/model71 is protected against
postprocessor writes. The verifier requires identical model/config/weight
identity, a fixed completed training run, and canonical train/validation fold
manifests. Model71 decoding is batch-sensitive: the all-row fit-side run used
batch one while the signed OOF run used batch four. Heldout predictions are
therefore taken only from the signed OOF artifact; all-row predictions are
used only on non-heldout fit rows. The declared batch sizes, difference count,
and difference-ID digest are recorded. A single blank fit-side decode receives
the fixed label-independent `?` fill; any heldout blank fails closed.

Status
------
The audited fold-0 report completed. Raw Model71 scored 0.8450843486068216;
the boundary stage was best at 0.8679556058552770, ahead of context
(0.8649328438461055), confusions (0.8677559722724811), approximate joins
(0.8660283885120055), and supervised errors (0.8659930811431867). The
boundary terminal policy is therefore frozen for confirmation folds. This is
only one heldout fold, not a complete OOF score or submission. Model35 remains
the promoted candidate until confirmation/all-fold evidence exists.
