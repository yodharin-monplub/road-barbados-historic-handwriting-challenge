MODEL 33 — leakage-safe metric-weighted candidate selector

Purpose
-------
Choose one complete line transcription per row between:

  candidate A: the conservative production candidate (currently Model25)
  candidate B: a genuinely OOF challenger (currently fine-tuned Model22)

The model never splices a label-derived token into a prediction. It only chooses
between two already-produced strings.

Legal inference features
------------------------
The fixed feature schema contains only information available when predicting:

  * both candidate strings and their length/character/token shapes;
  * character- and word-level disagreement between the two strings;
  * optional mean/lower-tail OCR confidence summaries for either candidate;
  * character n-gram and word-unigram language quality learned from training
    transcriptions that exclude the row being evaluated.

No image dimensions, file sizes, pixels, row IDs, test labels, or hand-written
row overrides are features. A confidence JSONL may be a source recognizer for a
postprocessed candidate; source-to-final text distance is then an explicit
feature. Missing confidence is supported and explicitly marked.

Exact objective
---------------
For selector training, each row is labelled according to which candidate has
the smaller additive contribution to the exact challenge mean error. The sample
importance is the absolute difference between those two contributions. This
preserves the competition's sqrt reference-length weights, WER/CER normalizers
(12 and 55), and the 50/50 WER/CER mixture. There is no row-oracle output.

Leakage control
---------------
Complete five-fold evaluation leaves out the canonical fold from both the
selector fit and target language model. Selector-training rows receive their LM
features through another nested cross-fit, so a row's own target never boosts
its language score.

Until Model22 has all five OOF folds, --single-fold 0 performs deterministic
five-way inner cross-fitting inside fold 0. Every reported fold-0 prediction is
therefore still made without that row's target. This score is a valid held-out
selector experiment, but it is not a complete five-fold OOF score.

Evaluate now on the available fold 0
------------------------------------
  python3 model33/evaluate.py \
    --truth data/processed/train_folds.csv \
    --candidate-a outputs/model25/allfold_oof.csv \
    --details-a outputs/model8/allfold_full_constrained.jsonl \
    --candidate-b outputs/model22/checkpoint_evaluation/fold0/predictions/checkpoint_07-0.9209.csv \
    --details-b outputs/model22/checkpoint_evaluation/fold0/predictions/checkpoint_07-0.9209.jsonl \
    --config model33/config.json \
    --single-fold 0 \
    --output-dir outputs/model33/fold0_inner_crossfit_rerun

Evaluate complete five-fold OOF later
-------------------------------------
First combine the five Model22 OOF fold CSV/JSONL artifacts without refitting
them on their held-out rows. Then omit --single-fold:

  python3 model33/evaluate.py \
    --truth data/processed/train_folds.csv \
    --candidate-a outputs/model25/allfold_oof.csv \
    --details-a outputs/model8/allfold_full_constrained.jsonl \
    --candidate-b outputs/model22/allfold_oof.csv \
    --details-b outputs/model22/allfold_oof.jsonl \
    --config model33/config.json \
    --output-dir outputs/model33/allfold_crossfit

Fit the deployable selector only after complete OOF exists
----------------------------------------------------------
  python3 model33/fit.py \
    --truth data/processed/train_folds.csv \
    --candidate-a outputs/model25/allfold_oof.csv \
    --details-a outputs/model8/allfold_full_constrained.jsonl \
    --candidate-b outputs/model22/allfold_oof.csv \
    --details-b outputs/model22/allfold_oof.jsonl \
    --config model33/config.json \
    --output-model outputs/model33/deployment.json

Apply to test
-------------
  python3 model33/apply.py \
    --input data/raw/Test.csv \
    --candidate-a outputs/model25/test.csv \
    --details-a outputs/model8/test_full.jsonl \
    --candidate-b outputs/model22/test.csv \
    --details-b outputs/model22/test.jsonl \
    --model outputs/model33/deployment.json \
    --output outputs/model33/test.csv \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model33_selector.csv

Promotion rule
--------------
Do not submit Model33 from the single-fold experiment. Promote only if complete
five-fold cross-fitted OOF improves on Model25 and the held-out gain is not
driven by a single fold. The output report records split ID hashes and explicit
label-exclusion assertions for auditability.

Status
------
Implementation and focused leakage tests are complete. The fixed configuration
was evaluated once with exact internal-space preservation on the available 819
fold-0 rows. The selector-level inner-cross-fitted result is:

  Model33 selector      0.842145141
  candidate A/Model25   0.817450092   delta +0.024695050
  candidate B/Model22   0.813019600   delta +0.029125541
  selected A/B rows     453 / 366

The five inner-partition selector scores are 0.843009598, 0.838987747,
0.842135634, 0.849187669, and 0.837415536. Each partition improves on its own
Model25 baseline; the gains are +0.030107839, +0.021545719, +0.020301769,
+0.019191423, and +0.032135100 respectively.

Artifact:

  outputs/model33/fold0_inner_crossfit_final/report.json

This is a genuinely held-out evaluation of the selector conditional on the two
supplied, fixed candidate artifacts: each selected row was excluded from the
selector and LM fits. It is not a complete five-fold OOF result and must not be
submitted yet. In addition, Model22's epoch-7 artifact came from the earlier
fold-0 checkpoint screen, so a fully nested end-to-end estimate requires the
remaining fixed-epoch Model22 folds (or an independently fixed checkpoint
policy). Treat 0.842145141 as strong evidence of complementarity, not as the
final submission estimate.
