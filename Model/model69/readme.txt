Model69 - leave-one-fold-out checkpoint epoch selection

Purpose
-------
Model36 preregistered epoch 7 from fold 0. Fold 1 instead peaks at epoch 5,
showing that one global development fold is not a stable checkpoint selector.
Model69 preserves all eight checkpoints and selects the epoch for held-out fold
f using the mean exact OOF score on the other four folds only.

Selection scope and limitations
------------------------------
The label-derived score for fold f is excluded when selecting fold f's epoch.
All checkpoint prediction hashes and reported scores are independently checked.
Ties select the earliest epoch. After all five folds exist, the deployment
epoch is the maximum all-fold mean and can be used for a separate all-data fit.

This excludes the held-out fold's score directly, but does NOT provide fully
nested validation: the other fold models used for epoch selection were trained
on the held-out fold's labels. The result is exploratory checkpoint-selection
evidence. Earlier descriptions calling this fully leakage-safe were too strong.
A strictly independent estimate requires inner models trained without the
outer fold, or an independent holdout. Do not promote a submission solely from
Model69's reported improvement. The fixed Model35/36 policy provides separate
confirmation on folds 1--4; fold 0 was its policy-development fold.

Run after folds 0--4 are evaluated
----------------------------------
  env PYTHONPATH=src .venv-kraken/bin/python model69/select_crossfold_epochs.py \
    --fold-leaderboard 0=outputs/model22/checkpoint_evaluation/fold0/leaderboard.json \
    --fold-leaderboard 1=outputs/model36/checkpoint_evaluation/fold1/leaderboard.json \
    --fold-leaderboard 2=outputs/model36/checkpoint_evaluation/fold2/leaderboard.json \
    --fold-leaderboard 3=outputs/model36/checkpoint_evaluation/fold3/leaderboard.json \
    --fold-leaderboard 4=outputs/model36/checkpoint_evaluation/fold4/leaderboard.json \
    --output-dir outputs/model69/oof

Status
------
Implemented and unit-tested while Model36 fold 2 trains. It cannot claim a
score until the preregistered eight-checkpoint matrices for folds 2--4 exist.
