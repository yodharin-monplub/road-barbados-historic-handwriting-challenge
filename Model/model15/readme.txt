MODEL 15 — cross-fitted recurring OCR-error repair

Purpose
-------
Learn high-precision word substitutions that recur between OOF OCR predictions
and their ground-truth transcriptions. This targets systematic recognition
errors that a label-only lexicon cannot identify.

Leakage control
---------------
For validation, each outer fold is split into five deterministic inner groups.
All base predictions in an outer fold were generated without that fold's
labels, and each evaluated inner group uses mappings learned only from the
other inner groups. A candidate must meet support, precision, length, and edit
distance gates. Never tune this model against test labels or leaderboard rows.

Tune
----
  python model15/tune_supervised_errors.py \
    --truth data/processed/train_folds.csv \
    --predictions outputs/model14/allfold_oof.csv \
    --output-config outputs/model15/allfold_config.json \
    --output-predictions outputs/model15/allfold_oof.csv

Apply to test
-------------
  python model15/apply_supervised_errors.py \
    --train data/processed/train_folds.csv \
    --training-predictions outputs/model14/allfold_oof.csv \
    --test data/raw/Test.csv \
    --predictions outputs/model14/test.csv \
    --config outputs/model15/allfold_config.json \
    --output outputs/model15/test.csv \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model15.csv

Status
------
Implementation and unit tests are complete. On the isolated fold-0 rehearsal,
the best inner-cross-fitted rule changed 23 tokens and improved multi-score
from 0.806518 to 0.807270. Promotion still requires a positive aggregate gain
on the full five-fold OOF set, with the per-fold deltas audited.
