Model129: round-2 pseudo-labels from per-fold ensemble votes
============================================================
Round 1 (Model125) labeled the test images with a single fold model. Round 2
labels them with a four-member word vote among ONLY that fold's models - each of
which excluded the fold being scored - so the labels are more accurate while the
held-out fold stays unseen. Organizer-approved (discussion 34459): the process
is strictly algorithmic, with no human transcription anywhere.

prepare_pseudo2.py builds those labels. Using the deployed Model124 test
ensemble instead would be wrong, because that vector mixes in models trained on
the fold being scored.

Results (complete five-fold OOF, 4,095 rows)
  member trained on round-1 labels (model125)   0.899227
  member trained on round-2 labels (model129)   0.900386   +0.001159
  five-member vote using it as anchor           0.906780   (was 0.905544)
  labels changed versus round 1: 283-316 lines per fold
Deployed as submissions/model129_ensemble5_maj.csv -> public 0.910015562.

Superseded by Model134's round-3 labels (member 0.902252, vote 0.910200,
public 0.913737765), which uses the Kraken-including ensemble as the labeler.

Run:
  python model129/prepare_pseudo2.py --members model125_pseudo model122_prod \
      model122_prod_lora model123_qwen3vl4b_lora --output-dir outputs/pseudo2
  python model122/train.py --config model129/config_pseudo2.json --fold N --predict-test
