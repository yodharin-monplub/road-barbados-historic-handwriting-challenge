Model67 - exact-context deletion repair

Purpose
-------
Remove a one-token OCR insertion only when the same left/current/right token
context is repeatedly aligned as a deletion in out-of-fold training pairs.

Leakage controls
----------------
For every held-out fold, rules are fitted on the other four folds only.
The frozen rule requires support >= 2 and empirical precision >= 0.8.
The stage reads Model66 predictions and is accepted only if its OOF evidence is
positive and stable.  Test rules are fitted only from OOF prediction/truth pairs.

Commands
--------
OOF:
  env PYTHONPATH=src .venv-kraken/bin/python model67/contextual_deletion_repair.py oof \
    --predictions outputs/model66/oof/allfold_oof.csv \
    --output-dir outputs/model67/oof

Test candidate (run only if OOF accepts the stage):
  env PYTHONPATH=src .venv-kraken/bin/python model67/contextual_deletion_repair.py apply \
    --training-predictions outputs/model66/oof/allfold_oof.csv \
    --predictions outputs/model66/test_candidate/test.csv \
    --output-dir outputs/model67/test_candidate \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model67_contextual_deletions.csv

Result
------
REJECTED. Five-fold OOF changed 2/4095 rows and reduced multi_score from
0.8170712445762045 to 0.8170310013797448 (-0.0000402431964597). Fold 3
regressed and the other folds were unchanged. No test candidate was generated;
Model66 remains the release candidate.
