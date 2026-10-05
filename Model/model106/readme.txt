Model106: six-source character ROVER with Model92
================================================

Model106 adds the independently augmentation-trained Model92 prediction as a
sixth held-fold witness to Model99's character-level ROVER.  The other five
witnesses and Model87 word-ROVER base are unchanged.

Vote thresholds 3, 4, 5, and 6 are screened on development fold 0 only.  The
winner is frozen before scoring confirmation folds 1--4.  Every source for a
fold was produced without that fold's labels.  No test output is created until
the confirmation gate passes and the matched Model105 all-data witness exists.

Run:

  PYTHONPATH=src python3 model106/evaluate.py

Result
------

Fold 0 selected threshold 4.  On untouched folds 1--4 it improved Model87 by
+0.0006632588.  Complete OOF is 0.8989967981 versus 0.8983390396 for Model87,
changing 221/4,095 rows.  This is only +0.0000001017 above Model99's existing
five-source character ROVER, so the sixth-witness vote itself adds essentially
no value.  Model104's learned selector remains the primary use of Model92.
