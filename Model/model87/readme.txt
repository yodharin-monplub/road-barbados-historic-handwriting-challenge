Model 87: held-fold-validated word ROVER
========================================

Model87 applies Model84's already-fixed, label-free word consensus to the five
held-fold Model83 sources. It replaces, inserts, or deletes an aligned word only
when at least three of five sources agree exactly; ambiguous positions retain
the fixed whole-line medoid. The five sources are full, enhanced, center,
center-enhanced, and corrected Kraken.

Exact five-fold OOF result
--------------------------

  Model73 full view:       0.8894664157724265
  Model83 fixed medoid:    0.8973904035303688
  Model87 word ROVER:      0.8983390396361067
  delta vs Model83:       +0.0009486361057379
  delta vs Model73:       +0.0088726238636802

The fixed rule improves folds 0, 1, 3, and 4. Fold 2 changes by only
-0.0000264640. It changes 150 of 4,095 OOF lines and 49 of 1,374 test lines.
A seeded 5,000-sample paired row bootstrap gives an incremental 95% interval
of [+0.0006052513, +0.0012904922], positive in every resample. This interval is
conditional on the saved predictions and excludes model-selection effects and
test distribution shift.

Deployment candidate
--------------------

  submissions/model87_model77_tta_word_rover.csv
  SHA-256 a412d1e868732cfb4223a792845287ebf89036e1d8ba59a82ed865a1425fccdd

Model87 never uploads or submits to Zindi automatically.
