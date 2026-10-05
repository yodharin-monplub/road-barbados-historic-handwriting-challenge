Model 81: five-fold Model73 deployment ensemble
================================================

Model81 decodes the 1,374-row test set independently with each of the five
signed Model73 checkpoint-400 adapters. The checkpoints use the fixed policy
that scored 0.8894664158 globally. All test prediction CSVs and inference
manifests must be copied back and hash-verified before any consensus candidate
is built.

This is a deployment diversity candidate, not a new OOF claim: four of the five
fold models trained on any given labeled row, so a five-model majority cannot
be evaluated as ordinary OOF without leakage. A text medoid/consensus policy
must therefore be fixed independently and the single all-data Model77 output
retained as the primary OOF-calibrated candidate.

Remote inference pattern (run for folds 0--4):

  PYTHONUNBUFFERED=1 PYTHONPATH=src .venv-lighton/bin/python model73/infer.py \
    --adapter outputs/model73/foldF/checkpoint-400 \
    --input data/raw/Test.csv --output outputs/model81/foldF_test.csv

Model81 never uploads or submits to Zindi automatically.

After all five verified CSVs are local, build the fixed equal-weight medoid:

Fold1 emitted one declared empty prediction for ID fjODbAygJXuQoAcT. The fixed
medoid reader deliberately rejects empty sources. First replace only that row
with the independently fixed mixed-distance medoid of folds 0, 2, 3, and 4;
the command records all donor strings, costs, and hashes in a JSON receipt:

  PYTHONPATH=src python3 model81/impute_single_blank.py \
    --target outputs/model81/fold1_test.csv \
    --donor outputs/model81/fold0_test.csv \
    --donor outputs/model81/fold2_test.csv \
    --donor outputs/model81/fold3_test.csv \
    --donor outputs/model81/fold4_test.csv \
    --output outputs/model81/fold1_test_imputed.csv \
    --receipt outputs/model81/fold1_test_imputation.json

Then build the fixed equal-weight medoid using the imputed fold1 source:

  PYTHONPATH=src python3 model81/apply_medoid.py --test data/raw/Test.csv \
    --source fold0=outputs/model81/fold0_test.csv \
    --source fold1=outputs/model81/fold1_test_imputed.csv \
    --source fold2=outputs/model81/fold2_test.csv \
    --source fold3=outputs/model81/fold3_test.csv \
    --source fold4=outputs/model81/fold4_test.csv \
    --output outputs/model81/medoid_test.csv \
    --agreement-output outputs/model81/medoid_agreement.jsonl \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model81_fivefold_medoid.csv

Completed result
----------------

All five 1,374-row sources were independently hash-checked against their
inference manifests and local 12-file adapter trees. Fold1's sole declared
blank was imputed from the four other folds with the documented rule above.
The final candidate contains 1,374 unique IDs in exact SampleSubmission order,
has columns ID,Target, and has no blank predictions.

Selection breakdown: 773 exact majorities, 590 strict medoids, and 11 lexical
ties. Mean agreement is 0.9599472887 and median agreement is 0.9730153061.
The full receipt is outputs/model81/report.json. Candidate submission:
submissions/model81_fivefold_medoid.csv (SHA-256
4155f57b2260917d5f19c03c9e84a7e49300c5ae643c4d258bc76db1e5465878).

No leaderboard submission was made automatically, and no OOF score is claimed
for this deployment-only five-fold ensemble.
