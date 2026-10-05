Model91: alternative-view selector screen
==========================================

Model91 tested the frozen, fold-safe Model79 logistic selector with Model88 as
candidate A and each held-fold LightOnOCR view as candidate B. All four tests
used outer-fold exclusion and nested-cross-fit training language features.

OOF scores:

  Model88 baseline:        0.8990278333
  versus full selector:    0.8988980611
  versus enhanced:         0.8990178758
  versus center:           0.8979751864
  versus center-enhanced:  0.8990297725

The +0.0000019392 center-enhanced result is effectively a tie caused by four
selected rows, and the other alternatives regress. Model91 is rejected and no
test candidate or submission is produced.
