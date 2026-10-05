MODEL 14 — fold-safe approximate joined-word repair

Purpose
-------
Split an unknown OCR token when it is within one or two character edits of a
frequent joined training bigram. For example, a noisy form of "presenceof" can
be mapped to "presence of" only when that exact adjacent pair is sufficiently
common in the supplied labels. This complements Model12, which handles only
exact joins.

Leakage and safety
------------------
OOF phrase counts and candidates exclude each row's fold. Known whole words
are untouched, short tokens are gated, all replacements are logged, and no
external corpus, dictionary, image metadata, or manual map is used. The final
test rule must be frozen from OOF validation before test application.

Tune
----
  python model14/tune_approximate_joins.py \
    --truth data/processed/train_folds.csv \
    --predictions outputs/model13/fold0_oof.csv \
    --fold 0 \
    --output-config outputs/model14/fold0_approximate_joins.json \
    --output-predictions outputs/model14/fold0_oof.csv

Apply to test
-------------
  python model14/apply_approximate_joins.py \
    --train data/processed/train_folds.csv \
    --test data/raw/Test.csv \
    --predictions outputs/model13/test.csv \
    --config outputs/model14/approximate_joins.json \
    --output outputs/model14/test.csv \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model14.csv

Status
------
Implementation, tests, and the 96-setting fold-0 sweep are complete. It
selected maximum distance 2, minimum token length 6, and minimum bigram count
5. It changed 80 tokens (49 at distance two), improving weighted WER from
3.482314 to 3.346919, weighted CER from 6.107284 to 5.942944, and multi-score
from 0.799383 to 0.806518. Five-fold confirmation remains required.
