Model111: image-conditioned candidate likelihood, with shuffled-image control
============================================================================
Difficulty 4/5: exact completion-token scoring and fold-safe evaluation.

Motivation: Model107 selector OOF increased to 0.9028869229, but the user reports
public 0.907264706, below Model104/105 public 0.909699454. Test labels are unknown.
First audit the 101 actual changed test lines and corresponding OOF changes.

Pilot protocol frozen before likelihood results:
- Canonical fold 0 only, 819 real images. Local CUDA, no new training/cloud start.
- Two judges: Model73 fold0 checkpoint400 and Model107 fold0 best_adapter. Both
  use the saved fold-excluded checkpoints, not any all-data deployment adapter.
- Clean candidate pair: raw Model73 and raw Model107 fold0 predictions. Model92
  fold0 lacks its original inference receipt locally, so do not silently use it
  as a fully provenance-verified candidate in this pilot.
- Also score Model104's OOF candidate for a separate DEVELOPMENT DIAGNOSTIC.
  Its reused upstream selector hierarchy is not fully nested. A result involving
  that candidate cannot establish an unbiased improvement or authorize submission.
- Feed each original image and each candidate to both OCR models. Teacher-force
  the exact candidate tokens plus EOS; exclude prompt/image tokens from loss.
  Average per-completion-token negative log likelihood across the two judges.
  Choose the lowest mean NLL; exact ties keep the primary (raw Model107 for the
  clean pair, Model104 for the deployment-related diagnostic).
- No threshold/weight fitting, no dictionaries, spelling changes, manual labels,
  external data, synthetic images, or test-label-dependent decisions.
- Negative control: repeat using a different fold0 image, assigned by a fixed
  label-independent cyclic permutation. Control rows never use their own image.
  This is a sensitivity check, not an image-free language model baseline.
- Also report a fixed conservative rule: replace the primary only if BOTH judges
  individually prefer the challenger. Do not pick a winning rule on fold0 and
  report its result as fresh validation. Mean NLL is the primary registered rule.
- Judge likelihood includes the OCR model's language prior; it is not a purely
  visual measurement, and judges may favor their own transcriptions.

Proceed to other folds only if the clean-pair mean-NLL rule improves over raw
Model107 by >=0.001 and beats its shuffled-image control. No automatic submission.
Fold0 has already been used for historical model development: this is a pilot,
not an untouched holdout. Later confirmation must exclude each evaluated row
from every fitted OCR, language model, and candidate-selection stage.

Commands:
  .venv-lighton/bin/python model111/audit.py
  .venv-lighton/bin/python model111/prepare.py
  .venv-lighton/bin/python model111/score.py --judge model73
  .venv-lighton/bin/python model111/score.py --judge model107
  .venv-lighton/bin/python model111/evaluate.py

Outputs: outputs/model111/audit_report.json, validation_changes.csv,
test_changes.csv, fold0/requests.json, fold0/model73_scores.jsonl,
fold0/model107_scores.jsonl, fold0/report.json. No CV result exists at creation.

Initial execution (2026-09-08)
------------------------------
Audit: 293 actual changed OOF lines; 155 improved, 118 worsened, 20 tied under
the competition metric's per-row contributions. There were 26 previously exact
transcriptions broken and 35 exact matches fixed. Net OOF gain +0.0014094806.
The 101 changed test lines comprise 23 case-only, 5 punctuation-only, 2 including
numeric changes, and 71 word/mixed changes. These have no correctness labels.
This audit does not identify the cause of the public-score decline.

Both judge checkpoints and prediction CSVs passed their saved inference-asset
SHA-256 receipts. Requests contain 819 fold0 IDs; 548 need likelihood scoring,
while 271 have identical text across all three candidates and need no decision.
The two-image Model73 CUDA smoke test completed successfully. Six new scorer
unit tests and 27 existing augmentation/VLM regression tests passed (33 total).

road-model111-local-pilot.service runs scripts/run_model111.py on local CUDA,
with sequential judges and 600-second progress checks. Each scored row is saved
for resumption; completed files receive checksums before evaluation. This run
does not start a cloud instance, fit a new model, or create/upload a submission.
Status: outputs/model111/status.json. Logs: model73.log, model107.log,
evaluation.log and supervisor.log under outputs/model111.

Pilot result and confirmation (2026-09-08)
-----------------------------------------
Fold0 primary mean-NLL score: 0.8986621211 versus raw Model107 0.8950117648
(+0.0036503563). Shuffled-image rule: 0.8943664686. Pilot gate passed.
The Model104-related diagnostic scored 0.9027757522 versus 0.9021704215;
this remains a non-fully-nested ensemble diagnostic, not a promotion result.

The user approved unchanged confirmation on folds 1-4. confirm.py imports the
EXACT original likelihood/make_prefix functions from score.py; common.choose
is unchanged. Their source hashes must match the completed pilot receipts.
No pilot request, score, or report is overwritten. All eight additional judge
checkpoints and candidate prediction files passed saved inference SHA checks.
Each fold has 819 validation rows; respectively 559, 536, 551, and 551 rows need
scoring because at least two candidates differ. Same-fold shuffled controls use
the same fixed, label-independent hash/cyclic assignment as the pilot.

Confirmation decision is frozen in outputs/model111/confirmation_protocol.json:
on folds 1-4 combined require mean-NLL gain >=0.001 over raw Model107, gains on
at least 3 of 4 folds, positive paired-bootstrap lower bound and superiority to
the shuffled-image control. All-five-fold results are also reported separately.
These are conditions for confirming the raw-model method, NOT evidence that it
beats the best submitted ensemble. No automatic test/deployment/submission.

road-model111-confirmation.service runs scripts/confirm_model111.py. It waits
for a free local GPU before each judge, never interrupts other work, and checks
long jobs or GPU availability every 600 seconds. No cloud instance is started.
Status: outputs/model111/confirmation_status.json. Each fold's logs, requests,
scores and report remain in outputs/model111/foldN. Final combined output:
outputs/model111/full_oof_report.json and clean_real_mean_nll_full_oof.csv.
Twelve scorer/confirmation tests passed before launch.
