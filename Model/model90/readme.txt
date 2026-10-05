Model90: recurring OCR-error dictionary
=======================================

Model90 learns high-consistency token substitutions from Model88 OOF errors,
using only the supplied training labels. For each validation fold, mappings
are learned solely from the other four folds. Candidate mappings are screened
by edit distance, support, empirical precision, word length, and whether the
source token is absent from the training truth vocabulary.

The selected fold-safe rule is fitted on all Model88 OOF prediction/Target
pairs and applied to Model88 test predictions. No external corpus or public
leaderboard feedback enters the rule.

Run:
PYTHONPATH=src:. python3 model90/recurring_error_dictionary.py \
  --truth data/processed/train_folds.csv \
  --oof outputs/model88/word_rover_vs_kraken_selector/crossfit_predictions.csv \
  --train data/raw/Train.csv --test data/raw/Test.csv \
  --test-predictions outputs/model88/test.csv \
  --sample data/raw/SampleSubmission.csv \
  --output-dir outputs/model90/recurring_errors \
  --submission submissions/model90_recurring_errors.csv

Result
------
The fold-safe sweep selected the no-change behavior: 0 changed OOF/test rows
and score 0.8990278333490134, equal to Model88. Recurring token errors did not
generalize across held-out folds at the required support and precision. This
model is rejected and its identical submission should not be uploaded.
