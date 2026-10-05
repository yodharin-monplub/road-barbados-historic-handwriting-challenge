MODEL 10 — fold-safe train-only lexical repair

Purpose
-------
Correct likely one-character OCR errors using only words observed in the
provided Train.csv labels. Candidate generation is deterministic and limited
to ASCII alphabetic tokens at edit distance one. Punctuation and the OCR
token's capitalization pattern are preserved. No external dictionary, manual
replacement table, test-image inspection, or image metadata is used.

Leakage control
---------------
During OOF tuning, every validation row uses a lexicon built from the other
four folds. With combined OOF predictions, each row automatically receives the
lexicon excluding its own fold. For test inference the already-frozen rule uses
all eligible training labels. Every changed sentence is logged to JSONL by
code; do not hand-edit that file.

Tune
----
  python model10/tune_lexicon.py \
    --truth data/processed/train_folds.csv \
    --predictions outputs/model9/fold0_confidence_constrained_oof.csv \
    --fold 0 \
    --output-config outputs/model10/fold0_lexicon.json \
    --output-predictions outputs/model10/fold0_oof.csv

After all-fold OOF is available, omit `--fold` and freeze parameters from all
five folds rather than the initial fold alone.

Apply to test
-------------
  python model10/apply_lexicon.py \
    --train data/processed/train_folds.csv \
    --test data/raw/Test.csv \
    --predictions outputs/model9/test.csv \
    --config outputs/model10/lexicon.json \
    --output outputs/model10/test.csv \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model10.csv

Status
------
The corrector, fold-safe tuner, change audit, test application, and submission
writer are implemented. The packaged 50-setting sweep selected minimum
frequency 3 and ambiguity ratio 1.0. On the 819-row fold-0 validation set it
improved weighted WER from 4.642130 to 3.951156, weighted CER from 7.043966 to
6.572657, and the normalized multi-score from 0.742542 to 0.775617. It changed
1,343 tokens across 648 lines; 372 lines improved and 147 were harmed by
character edit distance. Five-fold stability is still required before treating
this as the final private-leaderboard rule.
