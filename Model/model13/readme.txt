MODEL 13 — fold-safe valid-word confusion repair

Purpose
-------
Correct contextually implausible one-edit words even when the OCR token is a
valid training word (for example, a likely "tho"/"the" confusion). Candidate
frequency and the two adjacent word transitions must beat the source score by
a validation-selected safety margin. The vocabulary and all counts come only
from the supplied training transcriptions.

Leakage control
---------------
OOF rows use models built from the other four folds. Test inference uses all
eligible labels only after parameters are frozen. This model uses no external
dictionary, manual map, image inspection, or image metadata, and logs every
automated replacement.

Tune
----
  python model13/tune_confusions.py \
    --truth data/processed/train_folds.csv \
    --predictions outputs/model12/fold0_oof.csv \
    --fold 0 \
    --output-config outputs/model13/fold0_confusions.json \
    --output-predictions outputs/model13/fold0_oof.csv

Apply to test
-------------
  python model13/apply_confusions.py \
    --train data/processed/train_folds.csv \
    --test data/raw/Test.csv \
    --predictions outputs/model12/test.csv \
    --config outputs/model13/confusions.json \
    --output outputs/model13/test.csv \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model13.csv

Status
------
Implementation and the 60-setting fold-0 sweep are complete. Context weight 2
and score margin 10 changed 170 tokens, improving weighted WER from 3.531495 to
3.482314, weighted CER from 6.122760 to 6.107284, and multi-score from 0.797193
to 0.799383. Confirm all five folds before using this higher-risk language
correction as a final private-leaderboard candidate.
