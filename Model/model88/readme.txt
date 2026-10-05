Model 88: word ROVER plus Kraken confidence selector
====================================================

Model88 uses Model87's held-fold word ROVER as candidate A and corrected
Model35 Kraken with genuine per-character confidence as candidate B. It applies
the already-frozen metric-weighted selector with outer-fold exclusion and
nested-cross-fit language features.

Exact five-fold OOF result
--------------------------

  Model73 full view:       0.8894664157724265
  Model83 fixed medoid:    0.8973904035303688
  Model87 word ROVER:      0.8983390396361067
  Model88 selector:        0.8990278333490157
  delta vs Model87:       +0.0006887937129090
  delta vs Model73:       +0.0095614175765892
  selected Kraken:        68 / 4095 OOF rows

Model88 improves Model87 on every canonical fold. The all-data deployment
selector chooses Kraken for 20 test rows, changing 12 texts relative to
Model87. No public-leaderboard label or score is used by the rule.
A seeded 5,000-sample paired row bootstrap gives an incremental 95% interval
of [+0.0003211950, +0.0011362737], positive in every resample. This is
conditional on the saved predictions and excludes training randomness,
model-selection effects, group correlation, and test distribution shift.

Recommended submission candidate
--------------------------------

  submissions/model88_model77_rover_kraken_selector.csv
  SHA-256 3ae11857fca0fa4397339c2f1593f02447cf347b0e7d58d9d56b48c179c8ef9f

Model88 never uploads or submits to Zindi automatically.
