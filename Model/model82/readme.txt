Model 82: confidence selector on the Model81 deployment medoid
================================================================

Purpose
-------
Model82 applies the already-frozen Model80 logistic selector to Model81's
five-fold LightOnOCR medoid as candidate A and the corrected all-data Model35
Kraken prediction as candidate B. It uses genuine Model35 character confidence
details and the Model80 deployment language model. No parameter is refitted or
tuned from leaderboard feedback.

Evidence boundary
-----------------
Model80's leakage-safe OOF score is 0.8936202069 when candidate A is the
single held-out Model73 prediction. Model81 is a deployment-only five-model
medoid, so Model82 has no directly valid OOF score: applying the selector to
this stronger, distribution-shifted candidate is an extrapolation. Keep
Model81 as the primary candidate and treat Model82 as a distinct second-slot
experiment, never as a proven OOF improvement.

Build
-----

  PYTHONPATH=src python3 model82/apply.py --input data/raw/Test.csv \
    --candidate-a outputs/model81/medoid_test.csv \
    --candidate-b outputs/model35/deployment_epoch07/confusions_test.csv \
    --details-b outputs/model36/deployment_epoch07/test_predictions.jsonl \
    --model outputs/model80/deployment.json \
    --output outputs/model82/test.csv \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model82_model81_confidence_selector.csv

Nothing in Model82 uploads or submits to Zindi automatically.

Completed candidate
-------------------
The frozen selector chose Model35 on 48 rows, but 15 of those already matched
Model81, so the submission differs from Model81 on 33 / 1,374 rows. The changed
rows have mean Model81 agreement 0.9029873895 versus 0.9613489864 for unchanged
rows; 23 are strict-medoid decisions and 10 are exact-majority decisions.

Candidate: submissions/model82_model81_confidence_selector.csv
SHA-256: bd80f962d589a55ea13d2144195e9551f9c725a732bb86ba309aede0e4a34f07

This remains an extrapolative second-slot candidate, not a replacement for the
publicly validated Model81 score of 0.90725997.
