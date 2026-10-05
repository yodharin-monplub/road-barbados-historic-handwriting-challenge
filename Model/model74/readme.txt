MODEL 74 — SIGNED MODEL73 FOLD SCORER AND FUSION SCREEN
=======================================================

Purpose
-------
Model74 turns a Model73 heldout decode into a provenance-bound score receipt.
It verifies the truth, prediction, inference-manifest, and complete LoRA
adapter-tree hashes; requires exact heldout IDs; records blank outputs; and
then computes the official local multi-score. It prevents an ad hoc CSV or
wrong-fold result from being promoted.

After the standalone fold-0 score is known, Model74 may also host a strictly
heldout comparison/fusion screen against Model71/72. Any learned fusion rule
must be frozen on fold 0 and confirmed on another fold before five-fold OOF or
test use. No submission is created automatically.

Status
------
The evaluator signed all five fixed checkpoint-400 heldout artifacts. Every
fold contains exactly 819 expected IDs and no blank prediction. Fold scores are
0.8906962580, 0.8954270287, 0.8887612944, 0.8835489034, and 0.8888979614.
Model76 independently combined the receipts into the exact 0.8894664158 global
OOF result. No submission was created.
