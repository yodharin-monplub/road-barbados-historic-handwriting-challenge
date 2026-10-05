MODEL 12 — fold-safe word-boundary repair

Purpose
-------
Repair OCR tokens such as "andseales" -> "and seales" and "reco very" ->
"recovery" using only exact words and adjacent word pairs observed in the
provided training transcriptions. Known whole words are never split, and a
merge is attempted only when at least one source piece is unknown. Punctuation
boundaries are protected. No external text, image metadata, or hand-authored
correction list is used.

Leakage control
---------------
Every validation row is transformed using unigram and bigram counts from the
other four folds. Test inference uses all eligible training labels only after
the split/merge thresholds are frozen. All changed lines and operations are
written to JSONL for reproducibility; never hand-edit test predictions.

Tune
----
  python model12/tune_boundaries.py \
    --truth data/processed/train_folds.csv \
    --predictions outputs/model11/fold0_oof.csv \
    --fold 0 \
    --output-config outputs/model12/fold0_boundaries.json \
    --output-predictions outputs/model12/fold0_oof.csv

After five-fold Model11 OOF is ready, omit --fold and freeze the final rule on
all five held-out partitions.

Apply to test
-------------
  python model12/apply_boundaries.py \
    --train data/processed/train_folds.csv \
    --test data/raw/Test.csv \
    --predictions outputs/model11/test.csv \
    --config outputs/model12/boundaries.json \
    --output outputs/model12/test.csv \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model12.csv

Status
------
Implementation and all 49 fold-0 settings are complete. The sweep selected
split count 1 and merge frequency 1, making 62 splits and five merges. On 819
rows it improved weighted WER from 3.669149 to 3.531495, weighted CER from
6.192340 to 6.122760, and multi-score from 0.790825 to 0.797193. Five-fold
confirmation is required before this becomes a final private candidate.
