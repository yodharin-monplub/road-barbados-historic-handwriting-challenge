Model126: outer-fold recurring-error repair - REJECTED (no gain left)
=====================================================================
Rebuild of the most productive stage of the lost Models 58/61/66 chain: learn
word substitutions from the (prediction, truth) pairs of the OTHER four folds,
apply them to the held-out fold, so no fold's own labels touch its own repair.
For test, the table is learned from all five folds.

Result on the Model124 four-member ensemble (complete OOF, 4,095 rows, base
0.905544):
  min_support 2, min_precision 0.9 :  22 rows changed, -0.000187
  min_support 3, min_precision 0.9 :   2 rows changed, -0.000031
  min_support 4-5, precision 0.8-0.9:  0-2 rows changed, 0.000000 to -0.000025
Earlier, on the weaker three-member ensemble (base 0.903271), the best variant
changed a single row for +0.000015.

Conclusion: this stage is exhausted. It earned its keep historically because the
base was far weaker - Models 11-15 lifted 0.7538 to 0.8109, and Model58 added
+0.0015 on a 0.8146 base - but recurring word-level errors with support >= 3 and
precision >= 0.8 have essentially been voted away by Model124. The remaining
errors are one-off misreadings of hard handwriting, not systematic OCR-to-truth
confusions, so a frequency-gated substitution table has nothing to fix and the
few rules it does find are as often wrong as right.

Not applied to any submission. Kept as a measured negative result so the stage
is not rebuilt again on the assumption that it "worked before".

Run:
  python model126/repair.py oof --predictions outputs/model124/oof_ensemble.csv \
      --min-support 3 --min-precision 0.9
