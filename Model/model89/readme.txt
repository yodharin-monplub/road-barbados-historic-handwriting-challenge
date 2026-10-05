Model89: conservative training-only historical-language dictionary correction

Purpose
-------
Test a competition-safe version of the proposed dictionary postprocessor on
top of Model88. No external text, dictionary, labels, or OCR data enters the
predictions.

Method
------
For every held-out fold, build a vocabulary and word-bigram table from the
other four folds' Target labels. Consider only OCR words absent from that
fold-safe vocabulary. A replacement must be exactly one character edit away,
meet a training-frequency threshold, be unique or have a large frequency
margin over the runner-up, and have immediate left/right context support.
Short and optionally all-uppercase tokens are protected.

The configuration grid includes an effectively conservative set of minimum
length, frequency, context, ambiguity, and uppercase-protection thresholds.
Configuration evaluation uses leakage-safe OOF transformations. The selected
configuration is then fitted on all supplied Train.csv labels and applied to
Model88 test predictions.

Run
---
PYTHONPATH=src python3 model89/train_only_dictionary.py \
  --truth data/processed/train_folds.csv \
  --oof outputs/model88/word_rover_vs_kraken_selector/crossfit_predictions.csv \
  --train data/raw/Train.csv \
  --test data/raw/Test.csv \
  --test-predictions outputs/model88/test.csv \
  --sample data/raw/SampleSubmission.csv \
  --output-dir outputs/model89/conservative_dictionary \
  --submission submissions/model89_training_dictionary.csv

Artifacts
---------
outputs/model89/conservative_dictionary/report.json
outputs/model89/conservative_dictionary/crossfit_predictions.csv
outputs/model89/conservative_dictionary/crossfit_changes.jsonl
outputs/model89/conservative_dictionary/test.csv
outputs/model89/conservative_dictionary/test_changes.jsonl
submissions/model89_training_dictionary.csv

Result
------
Model88 OOF: 0.8990278333490157
Model89 OOF: 0.8991651361120013
Delta:       +0.0001373027629856

The selected rule changes 168 tokens across 164/4,095 OOF rows and 64/1,374
test rows. It improves folds 0, 1, 2, and 4 but regresses fold 3 slightly. A
5,000-sample paired row bootstrap interval is [-0.0001217969, +0.0004082647],
with 85% positive resamples. This is a useful experimental candidate, but the
gain is not stable enough to recommend spending a submission on it alone.
