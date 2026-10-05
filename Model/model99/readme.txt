Model99: conservative character-level ROVER
============================================

Model99 tests whether the five held-fold OCR witnesses can repair individual
characters inside words where Model87's exact word vote is inconclusive. It
aligns every source to the Model87 word-ROVER base, then accepts character
substitutions, insertions, or deletions only after an exact fixed vote.

Vote thresholds 3, 4, and 5 are screened on development fold 0 only. The
chosen threshold is frozen before scoring confirmation folds 1--4. This uses
only already-generated competition-image predictions and needs no GPU.

Run:

  PYTHONPATH=src python3 model99/evaluate.py

No test prediction or submission is created unless confirmation improves the
Model87 baseline.

Result
------

Fold 0 selected the 3-of-5 threshold. On untouched folds 1--4 it improved
Model87 from 0.8983867801 to 0.8990592792 (+0.0006724991). The complete OOF
score is 0.8989966964 versus 0.8983390396 for Model87. It changes 386/4,095
OOF rows and 114/1,374 test rows, and is promoted as Model100 candidate A.
