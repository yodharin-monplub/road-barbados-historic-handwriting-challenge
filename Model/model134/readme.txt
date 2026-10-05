Model134: third pseudo-label round (labels from the Kraken-including ensemble)
==============================================================================
Each round relabels the 1,374 test images with the best ensemble available at
the time and retrains the main member on 3,276 real rows plus those pseudo
rows. Fold k's labels are always produced by a vote among ONLY the fold-k
models of each member, every one of which excluded fold k, so the held-out
score stays honest. Organizer-approved (discussions/34459).

Round 3 labels come from the five-member vote that includes the PP-OCRv6/Kraken
member (Model130); round 2's came from a four-member vote without it. 126-158
lines per fold differ from the round-2 labels.

Member progression (complete five-fold OOF, all 4,095 rows)
  no pseudo-labels (model122)                 0.896608
  round 1, single-model labels (model125)     0.899227   +0.002619
  round 2, 4-member vote labels (model129)    0.900386   +0.001159
  round 3, 5-member vote labels (model134)    0.902252   +0.001866
  fold 0 alone: 0.899821 -> 0.902607 -> 0.904140 -> 0.905961

Ensembles on the complete OOF (anchor first, min_votes 2)
  model129 anchor + prod + lora + qwen + kraken   0.909376  (submission #4, public 0.913145)
  model134 anchor + prod + lora + qwen + kraken   0.910200  <- selected, submission #5
  both pseudo members kept (6 voters)             0.909956
  six voters with min_votes 3                     0.908387
Keeping both pseudo members is slightly worse than keeping only the newest:
they are successive refinements of one recipe and their errors overlap, the
same dilution Model132's second Kraken showed more sharply.

Deployed as submissions/model134_ensemble5_pseudo3.csv
  SHA-256 9df715544e0c8b89e007d483d76f618021c8f7bb0b3a3417fe7e2df2e6af354f
  1374 rows, 82 lines changed versus the anchor's own fold consensus.

Diminishing returns are clear: +0.0026, +0.0012, +0.0019 on the member across
the three rounds, and only +0.0008 on the ensemble this time, versus +0.0031
public for adding one new architecture. A fourth round is not worth the GPU
time; a genuinely different recognizer would be.

Run:
  python model129/prepare_pseudo2.py --members model129_pseudo2 model122_prod \
      model122_prod_lora model123_qwen3vl4b_lora model130_kraken --output-dir outputs/pseudo3
  python model122/train.py --config model134/config_pseudo3.json --fold N --predict-test
