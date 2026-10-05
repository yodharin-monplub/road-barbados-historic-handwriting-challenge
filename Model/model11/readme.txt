MODEL 11 — fold-safe contextual two-edit repair

Purpose
-------
Extend Model10 with a train-only word bigram language model and carefully
guarded edit-distance-two candidates. Words shorter than six characters remain
limited to one edit. A candidate is retained only when it has adjacent context
support in the training labels or sufficient training frequency. No external
dictionary, test-image inspection, manual replacement table, or image metadata
is used.

Leakage control
---------------
Each OOF line is decoded with unigram and bigram counts built from the other
four folds. The first-pass context is Model10's deterministic one-edit output,
also built from those same fold-excluded labels. Test application uses all
eligible training labels only after the hyperparameters have been frozen.

Tune
----
  python model11/tune_context.py \
    --truth data/processed/train_folds.csv \
    --predictions outputs/model9/fold0_confidence_constrained_oof.csv \
    --fold 0 \
    --output-config outputs/model11/fold0_context.json \
    --output-predictions outputs/model11/fold0_oof.csv

When all five OOF folds exist, omit --fold and use that result for the final
private-leaderboard candidate.

Apply to test
-------------
  python model11/apply_context.py \
    --train data/processed/train_folds.csv \
    --test data/raw/Test.csv \
    --predictions outputs/model9/test.csv \
    --config outputs/model11/context.json \
    --output outputs/model11/test.csv \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model11.csv

Status
------
Implementation and tests are complete. The packaged 128-setting fold-0 sweep
selected minimum frequency 2, context weight 16, distance penalty 64, and
fallback frequency 5. On 819 rows it reached weighted WER 3.669149, weighted
CER 6.192340, and multi-score 0.790825. It changed 1,711 tokens, including 397
distance-two repairs. Treat this as validation evidence, not a hidden-test or
prize guarantee; freeze from five-fold OOF before the final submission.
