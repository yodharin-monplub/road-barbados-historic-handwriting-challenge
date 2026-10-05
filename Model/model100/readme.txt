Model100: character ROVER plus Kraken confidence selector
========================================================

Model100 applies the exact frozen Model88 cross-fitted selector to Model99's
character-level ROVER candidate and corrected Model35 Kraken. Every evaluation
fold is excluded from the selector fit and from the target-text language model;
training-side language features are nested-cross-fitted. The selector config is
unchanged from Model88, so this is a direct comparison rather than a retune.

Run:

  PYTHONPATH=src python3 model100/evaluate.py \
    --truth data/processed/train_folds.csv \
    --candidate-a outputs/model99/character_rover_oof.csv \
    --candidate-b outputs/model35/allfold_epoch07_verified/predictions/03_confusions_oof.csv \
    --details-b outputs/model80/model35_epoch07_details.jsonl \
    --config model100/config.json \
    --output-dir outputs/model100/character_rover_vs_kraken

No test prediction or submission is created until the complete OOF comparison
shows a meaningful improvement over Model88.

Current OOF result
------------------

  Model88:  0.8990278333490157
  Model100: 0.8996529328792820
  delta:   +0.0006250995302663

Model100 improves Model88 on every canonical fold. On untouched confirmation
folds 1--4 the gain is +0.0006477557. A seeded 5,000-sample paired row
bootstrap on those folds gives a 95% percentile interval of
[+0.0002831294, +0.0010153688], positive in every resample. This is
conditional on the saved predictions and does not cover training randomness,
record correlation, model-selection effects, or test shift.

Deployment candidate
--------------------

  submissions/model100_character_rover_kraken_selector.csv
  SHA-256 e74cf477d10bdd1ffa59007db6fccabb2d6aadd68213b735f1a77bbb1b1e70e6

The test pipeline changes 115/1,374 rows relative to Model88; the final fitted
selector chooses corrected Kraken for 12 rows. Nothing is submitted to Zindi
automatically.
