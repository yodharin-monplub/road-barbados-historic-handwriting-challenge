MODEL 26 — preregistered five-fold deployment medoid

Purpose
-------
Combine exactly five complete test transcriptions, one from each independently
fine-tuned fold model. The rule is fixed: every fold has weight 0.2 and the
distance is 0.45 normalized word edit distance plus 0.55 normalized character
edit distance. There is no parameter fitting, score input, or leaderboard
feedback in this model.

Selection rule
--------------
For each test ID:

1. An exact three-of-five (or stronger) string majority always wins.
2. Otherwise, select the source string with the smallest equal-weight mean
   distance to all five source strings.
3. If medoid costs tie, sum available top-candidate CTC confidence only for
   sources that emitted each tied string and prefer the larger sum.
4. A remaining tie is resolved lexically so changing source argument order
   cannot change the result.

Confidence never changes a strict majority or a strict medoid decision. CSV
sources have no confidence; Kraken-style JSONL sources may provide it in the
first entry of ``scores``. Only each source's first candidate is used.
Boundary whitespace and generated line breaks are cleaned, but scored repeated
internal spaces are preserved in source candidates, medoid output, and the
submission file.

Why there is no ordinary OOF score
----------------------------------
For a training row in fold f, only model f is genuinely held out. The other
four fold models trained on that row. Combining all five predictions on Train
would therefore leak the target into four votes and produce an invalid OOF
estimate. Honest tuning of a five-model fold ensemble would require nested
model training. Model26 instead preregisters equal weights and a fixed 0.55
character mix. Do not run Model7's weight tuner on five-fold Train predictions.

Recommended inputs
------------------
First apply the frozen Model19 view selector and the validated text-repair
chain independently to each fold model's three-view test output. Supply those
five final CSVs to Model26. Each source must contain every test ID exactly once
and every top prediction must be nonempty. Repeated source files are rejected.

Run
---
  python model26/apply_medoid.py \
    --test data/raw/Test.csv \
    --source fold0=outputs/model22/fold0_test_final.csv \
    --source fold1=outputs/model22/fold1_test_final.csv \
    --source fold2=outputs/model22/fold2_test_final.csv \
    --source fold3=outputs/model22/fold3_test_final.csv \
    --source fold4=outputs/model22/fold4_test_final.csv \
    --output outputs/model26/test.csv \
    --agreement-output outputs/model26/test_agreement.jsonl \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model26_fivefold_medoid.csv

When ``--sample`` is supplied, both the prediction file and diagnostics follow
its row order, even if Test.csv is ordered differently. ``--submission`` is
optional but requires ``--sample``; the completed file is validated against
the sample schema and IDs before success is reported. Agreement diagnostics
record the decision reason, exact votes, fixed medoid cost, agreement, source
texts/confidences, and every unique candidate's support.

Limitations
-----------
This text medoid can select only a string emitted by one of the five models.
It does not average incompatible fold codec logits and does not synthesize a
ROVER transcription. Its test benefit cannot be claimed from ordinary OOF,
and no leaderboard or prize result is guaranteed.
