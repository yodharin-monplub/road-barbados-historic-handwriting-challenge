Model 84: conservative five-fold word ROVER
============================================

Model84 improves on Model81's whole-line medoid by aligning the five fold
transcriptions at word level. It replaces, deletes, or inserts a word sequence
only when at least three of the five aligned sources agree exactly. Every
ambiguous position falls back to the publicly validated Model81 medoid. The
policy is fixed, label-free, and creates no manual transcription.

Because the five deployment folds are not leakage-safe on training rows,
Model84 has no ordinary OOF score. Model83 will validate this same fixed
consensus behavior with held-fold image views before broad promotion. Model84
is a deployment candidate and never submits automatically.

Build:

  PYTHONPATH=src python3 model84/apply.py --input data/raw/Test.csv \
    --base outputs/model81/medoid_test.csv \
    --source fold0=outputs/model81/fold0_test.csv \
    --source fold1=outputs/model81/fold1_test_imputed.csv \
    --source fold2=outputs/model81/fold2_test.csv \
    --source fold3=outputs/model81/fold3_test.csv \
    --source fold4=outputs/model81/fold4_test.csv \
    --output outputs/model84/test.csv \
    --diagnostics outputs/model84/changes.jsonl \
    --report outputs/model84/report.json \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model84_word_rover.csv

Completed candidate
-------------------
Model84 changes 47 / 1,374 rows: 44 word substitutions, five insertion events,
and two deletions. All 47 resulting lines are new consensus strings rather than
complete strings copied from one fold. Their mean Model81 agreement is
0.8760135642 versus 0.9629200732 for unchanged rows; 46 of 47 start from a
strict-medoid decision and one from a lexical tie. This is the intended hard
tail, but the held-fold Model83 screen is still required before submission.

Candidate: submissions/model84_word_rover.csv
SHA-256: b1096043aa43da772e1bf93ee3ba4596a3ab97635646842d9fd64acbeae55d9a
