MODEL 7 — N-best consensus, OOF weight tuning, and legal pseudo-labelling

Purpose
-------
Select the transcription that minimizes weighted disagreement across model
families. The candidate pool contains each model's N-best beams. Source weights
and the word/character consensus mix are tuned deterministically against true
out-of-fold (OOF) predictions using the exact local competition metric. This is
standard ensemble hyperparameter search, which the organizer permits; it is not
AutoML and never uses leaderboard feedback as labels.

Expected folder preparation
---------------------------
Put each source's five non-overlapping OOF JSONL files in its own directory,
for example outputs/oof/model1/ ... outputs/oof/model8/. Do not mix test JSONL
into those directories. Every source must predict every eligible OOF row.

Tune on combined OOF
--------------------
python model7/tune_ensemble.py \
  --truth data/processed/train_folds.csv \
  --source lighton_zero=outputs/oof/model1 \
  --source lighton_lora=outputs/oof/model2 \
  --source trocr=outputs/oof/model3 \
  --source ctc=outputs/oof/model4 \
  --source qwen3=outputs/oof/model5 \
  --source qwen25=outputs/oof/model6 \
  --source ppocr=outputs/oof/model8 \
  --output-config outputs/model7/ensemble.json \
  --output-predictions outputs/model7/oof.csv

The config records individual OOF scores, ensemble score, trial seed, source
weights, and consensus mix. Inspect score stability by fold before accepting it.

Apply to test
-------------
python model7/apply_ensemble.py \
  --test data/raw/Test.csv \
  --source lighton_zero=outputs/test/model1.jsonl \
  --source lighton_lora=outputs/test/model2.jsonl \
  --source trocr=outputs/test/model3.jsonl \
  --source ctc=outputs/test/model4.jsonl \
  --source qwen3=outputs/test/model5.jsonl \
  --source qwen25=outputs/test/model6.jsonl \
  --source ppocr=outputs/test/model8.jsonl \
  --config outputs/model7/ensemble.json \
  --output outputs/model7/test.csv \
  --agreement-output outputs/model7/test_agreement.jsonl \
  --sample data/raw/SampleSubmission.csv \
  --submission submissions/model7.csv

Optional automated pseudo-label stage
-------------------------------------
Zindi explicitly allows transductive pseudo-labelling when it is fully
algorithmic, reproducible, and contains no human-in-the-loop corrections. Build
a conservative augmented file with:

python model7/build_pseudolabels.py \
  --train data/processed/train_folds.csv \
  --agreement outputs/model7/test_agreement.jsonl \
  --output data/processed/train_plus_pseudo.csv \
  --min-agreement 0.88 \
  --min-exact-votes 2

Point a copied model2/model5/model6 config at that output and retrain. Keep the
plain ensemble as one of the final private candidates unless OOF simulation or
a held-out pseudo-label experiment shows a stable improvement. Never inspect
and hand-correct individual test predictions.

Status
------
Consensus tuning, application, agreement logging, submission creation, and
pseudo-label filtering are implemented. They await OOF predictions.
