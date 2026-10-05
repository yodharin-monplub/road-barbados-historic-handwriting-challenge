MODEL 21 — supervised intrinsic-feature view ranker (controlled rejection)

Purpose
-------
Test whether a fold-held-out tree regressor can choose among full, center, and
enhanced OCR using only recognized-text statistics, confidence distribution,
pairwise OCR agreement, and view identity. No image pixels, image dimensions,
file metadata, external labels, or leaderboard feedback are used.

Validation design
-----------------
For each of the five outer folds, the ranker is trained on the other four
folds' candidate errors and predicts the held-out fold. Features include 13
confidence summaries, text/token ratios, pairwise character/word distances,
relative confidence and length, and view indicators. Extra Trees and histogram
gradient boosting were checked with conservative leaf sizes.

Result
------
The best checked variant scored 0.738739926 five-fold OOF, far below Model 19's
0.753781973 character-LM selector. It is rejected and is retained only as
negative evidence. A useful future ranker would need train-only language-model
features and nested feature construction, not merely more tree capacity.
