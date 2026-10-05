Model94: absolute-edit-guarded Model93 correction
==================================================

Model93's fold-0 output changed only 66 of 819 rows, but one accepted output
rewrote 66 characters and dominated the aggregate loss. Model94 freezes a
simple maximum of six character edits, chosen on development fold 0. It keeps
Model88 whenever Model93 exceeds that cap.

Fold-0 score:

  Model88 baseline:       0.8989699979153494
  Model93 aggressive:     0.8979901609957579
  Model94 edit-cap guard: 0.8997912221115281
  Model94 delta:         +0.0008212241961787

The threshold is frozen before inspecting folds 1--4. Those folds are the
confirmation set; no test submission is created unless confirmation passes.

Confirmation result
-------------------

Model94 did not promote. On untouched folds 1--4, the combined score moved
from 0.8990422722150929 to 0.8991318425846592, a gain of only
+0.0000895703695662. Fold deltas were +0.0008062851, -0.0005580059,
-0.0001828574, and +0.0002911958. Only two folds were nonnegative and the
5,000-sample paired bootstrap positive fraction was 0.6164. Required values
were +0.0005, three nonnegative folds, and 0.95. No test prediction or Zindi
submission was created.

Apply to any Model93 prediction artifact with:

  PYTHONPATH=src python3 model94/apply_guard.py \
    --config model94/config.json --primary PRIMARY.csv \
    --challenger MODEL93.csv --output MODEL94.csv

Confirmation is reported with:

  PYTHONPATH=src python3 model94/confirm.py --config model94/config.json \
    --truth data/processed/train_folds.csv \
    --primary outputs/model88/word_rover_vs_kraken_selector/crossfit_predictions.csv \
    --challenger outputs/model94/allfold.csv \
    --output outputs/model94/confirmation.json
