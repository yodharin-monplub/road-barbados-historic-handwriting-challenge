Model 86: confidence selector over the five-view medoid
=======================================================

Model86 applies the already-frozen Model80 metric-weighted selector to a
stronger candidate A: Model83's five-source held-fold medoid. Candidate B is
the corrected Model35 Kraken prediction with its genuine per-character
confidence sequence. The outer-fold selector and target language model exclude
each scored fold, and selector-training language features are nested-cross-fit.

Exact five-fold OOF result
--------------------------

  Model73 full view:       0.8894664157724265
  Model83 fixed medoid:    0.8973904035303688
  Model86 selector:        0.8981558437436980
  delta vs Model83:       +0.0007654402133291
  delta vs Model73:       +0.0086894279712715
  selected Kraken:        67 / 4095 rows

Model86 improves Model83 on every canonical fold. No leaderboard result was
used to tune the selector or source set.

Deployment
----------

The all-data selector is `outputs/model86/deployment.json`. On test it selects
Kraken for 19 of 1,374 rows, changing 16 texts relative to Model85. The ready
submission is `submissions/model86_model85_kraken_selector.csv`, SHA-256:

  125272df97aa2e4566a8a569cbcbc58f4750702d9f776e97e5c2e570b24fe8ca

Model86 never uploads or submits to Zindi automatically.
