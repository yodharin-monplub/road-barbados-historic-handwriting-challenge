MODEL 35 — SAME-FOLD-MODEL CROSS-FITTED POSTPROCESSING
======================================================

Purpose
-------
Fit the Models11–15 sequence separately for every Model22/Model36 OCR fold
model. For outer fold f, the one fold-f OCR model must first predict all 4,095
training images. Model35 then:

  1. fits every recipe and setting from that model's predictions and labels on
     folds != f;
  2. carries the train-side output forward when fitting the next stage;
  3. freezes the resulting sequential recipe; and
  4. applies it exactly once to the same model's predictions on fold f.

This matters because Model28's top-level runner expects a complete OOF CSV in
which each row came from its own fold model. Calling Model28 directly on one
fold model's all-row inference would mix the wrong prediction lineage during
its outer loop. Model35 safely reuses only Model28's audited STAGES fit and
transform primitives.

What may be claimed
-------------------
Only fold f is OOF for a fold-f OCR model. Predictions and recipe-search scores
on folds != f are in-sample with respect to that OCR model. They are useful for
fitting, but they are NOT OOF scores and must never be reported as validation
performance. A global OOF score is produced only by model35/combine_oof.py
after it verifies one held-out slice from every canonical fold.

Required provenance
-------------------
run_fold.py fails closed unless all of the following agree:

  * the all-row CSV and its sibling inference manifest cover all truth IDs;
  * the inference used fold=None, exact-space preservation, and one model hash;
  * the signed held-out OOF CSV, its inference manifest, and evaluator receipt
    use the same model, config, image variant, charset setting, and spacing;
  * the held-out CSV checksum is signed by the evaluator receipt;
  * the receipt binds the model weights to the exact training-manifest hash;
  * the training/validation manifests contain exactly eligible folds != f and
    the complete fold f respectively;
  * the full inference reproduces every signed held-out OOF string exactly;
  * the checkpoint is the pre-fixed zero-based epoch 7 policy.

The output report records hashes of every input, weights, source manifests,
recipes, changes, and prediction artifacts. Writes into outputs/model22 and
outputs/model36 are forbidden. The reserved live directory
outputs/model35/fold0_epoch07_train_side_fit is also intentionally protected.

Provenance limitation: Kraken's ordinary all-row inference manifest records
the model/config/input hashes but does not sign the all-row CSV hash itself.
Model35 therefore requires it to be the CSV's exact sibling, hashes it at
intake, and cryptographically binds the held-out slice through the evaluator
receipt; the non-held-out fit-side CSV remains locally self-attested. As with
all local receipts, a malicious editor able to rewrite every artifact cannot be
excluded without external attestation.

Fold 0 command
--------------
The necessary all-row Model22 epoch-7 inference already exists. Use a fresh
Model35 output directory:

  PYTHONPATH=src .venv-kraken/bin/python model35/run_fold.py \
    --truth data/processed/train_folds.csv \
    --predictions outputs/model22/checkpoint_evaluation/fold0/train_and_validation_epoch07.csv \
    --inference-manifest outputs/model22/checkpoint_evaluation/fold0/train_and_validation_epoch07.manifest.json \
    --training-manifest outputs/model22/fold0/run_manifest.json \
    --heldout-oof-predictions outputs/model22/checkpoint_evaluation/fold0/predictions/checkpoint_07-0.9209.csv \
    --oof-evaluation-receipt outputs/model22/checkpoint_evaluation/fold0/predictions/checkpoint_07-0.9209.evaluation.json \
    --heldout-fold 0 \
    --checkpoint-epoch 7 \
    --checkpoint-policy fixed_zero_based_epoch_7_selected_once_on_fold0 \
    --policy-selection-fold 0 \
    --model-family model22 \
    --output-dir outputs/model35/fold0_epoch07_verified_rerun

Folds 1–4
---------
For each fixed-eight-epoch Model36 fold model, first run the existing exact
checkpoint evaluator on fold f and retain the zero-based epoch-7 artifact.
Then infer all rows with that exact converted weights file (omit --fold):

  PYTHONPATH=src .venv-kraken/bin/python model36/infer.py \
    --checkpoint <fold-f-epoch07.safetensors> \
    --input data/processed/train_folds.csv \
    --output outputs/model36/fold<f>/train_and_validation_epoch07.csv \
    --no-constrain-charset --batch-size 1

Invoke run_fold.py with the analogous fold-f all-row CSV/manifest, training
manifest, signed fold-f OOF CSV/evaluation receipt, --heldout-fold f,
--model-family model36, and exactly the same checkpoint policy string above.
Do not choose a different epoch or postprocessing recipe after viewing a fold's
held-out score.

Combine five verified held-out slices
--------------------------------------
Use only report.json files written by run_fold.py:

  PYTHONPATH=src .venv-kraken/bin/python model35/combine_oof.py \
    --truth data/processed/train_folds.csv \
    --fold-report outputs/model35/fold0_epoch07_verified/report.json \
    --fold-report outputs/model35/fold1_epoch07_verified/report.json \
    --fold-report outputs/model35/fold2_epoch07_verified/report.json \
    --fold-report outputs/model35/fold3_epoch07_verified/report.json \
    --fold-report outputs/model35/fold4_epoch07_verified/report.json \
    --output-dir outputs/model35/allfold_epoch07_verified

The combiner re-hashes every fold report and prediction, verifies exact fold
IDs, recomputes each reported held-out score, enforces one common checkpoint
policy, restores canonical truth order, and only then computes complete OOF
scores for raw and every Models11–15 stage.

CPU and dependencies
--------------------
Model35 is CPU-only and adds no dependency. It uses the existing dependency-
free challenge metric and Models11–15 correctors through
src/road_htr/crossfit_postprocess.py. No OCR inference, model conversion,
training, network access, or GPU operation occurs in Model35 itself.

Promotion rule
--------------
Do not submit or claim a complete Model35 score until all five provenance-
verified held-out slices have been combined. Promote only if the final complete
OOF score improves on both raw Model36 and the current fallback, with no fold
showing an unacceptable regression.

Status
------
Implementation, focused tests, and provenance-verified fold-0 and fold-1 runs
are complete. Each run took about 22 minutes on CPU and deliberately did not
touch the reserved active output directory named above.

Fold-0 held-out results:

  raw Model22 epoch 7       0.813019600
  after Model11/context     0.842682737
  after Model12/boundaries  0.846164941
  after Model13/confusions  0.846941606
  after Model14/joins       0.845484318
  after Model15/errors      0.846125810

Verified report:

  outputs/model35/fold0_epoch07_verified/report.json

Fold-1 held-out results:

  raw Model36 epoch 7       0.800785088
  after Model11/context     0.834679324
  after Model12/boundaries  0.838590393
  after Model13/confusions  0.839338500
  after Model14/joins       0.837242908
  after Model15/errors      0.837926264

Verified report:

  outputs/model35/fold1_epoch07_verified/report.json

The two-fold combined Model13/confusions score is 0.843140575, compared with
0.806903223 for the same two raw held-out slices. This is encouraging partial
OOF evidence, not a complete five-fold score.

The best fold-0 terminal point is currently Model13/confusions at 0.846941606,
but this is not a complete OOF score. If that terminal-stage choice is adopted,
it must be frozen now and used unchanged for folds 1–4; selecting a different
terminal stage after viewing each held-out fold would leak validation labels.
The terminal-stage policy is now explicitly frozen in
``model35/terminal_policy.json``: stop after Model13/confusions for every fold.
This decision was made from fold 0 before training or scoring folds 1–4. Never
replace it with per-fold stage selection. The 0.846941606 selection-fold score
remains conditional development evidence until the untouched folds confirm it.

2026-09-05 confirmation update
------------------------------
Fold 2 is also complete: raw epoch 7 scores 0.806714535 and the fixed
confusions stage scores 0.841263454. Folds 0--2 give partial OOF 0.842515813;
this includes the selection fold and is not a complete five-fold estimate.

Against the same-fold slices of Model66, the fixed candidate improves fold 1
by 0.021550684 and fold 2 by 0.025305502. Paired row bootstrap (2,000 draws,
seed 20260903) gives 95% percentile intervals [0.014671835, 0.028238284] and
[0.018943250, 0.032117361], respectively. Input hashes and full precision
results are saved in outputs/model35/confirmation_folds12_bootstrap.json.
These intervals condition on existing predictions; they do not account for
page correlation, training randomness, repeated model search, or test shift.
Folds 3--4 and the final comparison are still required before promotion.

Deployment preparation
----------------------
src/road_htr/model35_deployment.py provides the CPU fit/apply core for the
fixed context -> boundaries -> confusions sequence. It carries corrected
training predictions forward exactly as the fold pipeline does, accepts no
test labels, preserves internal spaces, and labels fitting scores in-sample.
Focused tests cover sequential fitting, input isolation, stage order, and row
counts. A synthetic integration test runs the real context, boundary, and
confusion correctors through both fold validation and deployment and requires
identical predictions and selected settings at every stage, including an
actual spelling correction. This validates code-path equivalence, not test
generalization. The artifact-verifying CLI is model35/build_submission.py. It binds
train and test inference to the same final all-data recognizer, verifies
complete OOF evidence, and validates submission IDs/schema before publishing
an output. Do not pass mixed-model OOF predictions as deployment fit inputs.

Do not use scripts/build_postprocessed_submission.py unchanged for this
candidate: it always applies approximate joins after confusions, whereas the
frozen Model35 terminal stage is confusions. Its default submission cleaner
also collapses internal whitespace. The Model35 deployment wrapper must use
clean_prediction_preserving_internal_spaces, emit the official transcript
column(s) in sample order, and validate exact sample IDs. Neither the legacy
builder nor the new low-level core alone verifies final model provenance.

Guarded submission CLI
----------------------
Run model35/build_submission.py --help for required artifact paths. This CPU
command requires the complete combined OOF report and revalidates its source
fold lineage, prediction hashes, global and per-fold scores. Its conservative
numeric gate requires strict improvement over raw Model36 and Model66 on each
fold, the complete OOF, and confirmation folds 1--4 together. A failed gate
requires further analysis; it does not authorize switching per-fold stages.
This gate is not a confidence interval or a claim of fully nested validation.

Deployment inputs must come from completed all-data Model36 training with all
eight checkpoints and a final best-weights artifact. A conversion receipt must
bind zero-based epoch 7 to the exact weights used by both all-training-row and
test inference. The Model36-specific inference wrapper adds
predictions_sha256 to final manifests; the CLI rejects unsigned final
predictions or changed CSVs. The shared historical Kraken CLI remains at its
pinned hash so older numbered models are not silently redefined. Source folds can
retain their older signed evaluation receipts; their policy is unchanged.

The output must be a new directory beneath outputs/model35. It contains stage
predictions, submission.csv in the official sample schema/order, fitting
diagnostics explicitly labeled non-OOF, checksums, and a report. Existing output
directories are never overwritten. No GPU job, rental, or upload is started.
The wrapper has fixture tests; real end-to-end deployment remains pending the
remaining folds and final all-data training/inference.

The source-lineage check compares freshly verified metadata with every field
recorded at recipe-fitting time, including the all-row prediction checksum.
This also detects changes to non-heldout predictions that a heldout-only
comparison cannot detect. Recipe checksums are revalidated as well. The real
completed fold-0, fold-1, and fold-2 reports passed this check on 2026-09-05;
older valid reports without the excluded-validation diagnostic remain usable.

model35/check_candidate.py runs the read-only numeric/provenance gate before
final all-data training, without requiring deployment predictions. It will
refuse the currently incomplete fold collection. The complete final-training,
fixed-checkpoint export, inference, and assembly commands are in model36/readme.txt.

Development-fold residual review, 2026-09-05
-------------------------------------------
outputs/model35/fold0_epoch07_verified/residual_analysis.json was produced by
scripts/analyze_predictions.py from the unchanged fixed-confusions predictions.
Its SHA256 is d0e1123ce4ab792938aa38ad4a02524d376110b5b21ab70ffb77f6532f970601.
It reproduces the conditional fold-0 score 0.846941606 across all 819 rows.
The worst-example ordering is character error rate, not global score impact.

Manual inspection of existing development images found two concrete failure
types worth investigating after the current five-fold experiment finishes:

1. Yy1NKXy9jbHmOrFK has a second text line containing the presence clause that
   the recognizer omits. tsVliOumuhe5DHRJ includes John Davis on the next line,
   also omitted, plus partial neighboring text at the top of the crop. A
   future layout-aware test should distinguish valid continuation lines from
   incidental neighboring lines; blindly concatenating every visible line
   could introduce errors. No layout rule or new OCR run was selected here.
2. xfH9BNTuAZfoQtrd visibly uses abbreviated Executors/Administrators, while
   its target contains both marked abbreviations and expanded words. This
   suggests an annotation-format issue in this example, not simply unreadable
   handwriting. Any expansion rule must be learned on train-side labels and
   independently validated, not inserted from heldout targets. Model24 already
   addresses caret restoration; expansion is a distinct hypothesis, not a
   demonstrated improvement over that model.

Aspect-ratio subgroup scores are descriptive only: 8--14 scores 0.815221925
(248 rows), 14--22 scores 0.855910381 (406 rows), and >=22 scores 0.872224379
(158 rows). The <8 subgroup has only seven rows. Length/content and image
quality may confound these comparisons; they do not justify an aspect-only
selection rule. Folds 1--4 were not inspected for this residual review. Current
epoch/terminal policies and all prediction artifacts remain unchanged.

Score-impact ranking, same development fold
-------------------------------------------
The analyzer now supports --sort-by score_loss_contribution. Row contributions
use the metric's actual character/word weights and full-evaluation denominators;
unit tests require their sum to equal the exact aggregate mean error, including
role-marker and internal-space handling. The original CER-ranked report above
is preserved. The new report is residual_impact_analysis.json in the same folder,
SHA256 b7d828ab3ea816887b8d24a73fa89c53bc149df6a4f69b9cd25fb5de0bd131b4.

The eight highest-impact rows contribute only 0.006172150 score loss out of
0.153058394 total (about 4%). Even perfect predictions on just those eight rows
would therefore give only that oracle improvement; this is not an achieved or
deployable gain. Most error is distributed across other rows. Approximately
73% of the total normalized loss comes from the word-error component, which
does not by itself identify whether spacing, spelling, names, or punctuation
caused it. Broad recognition/correction improvements remain necessary; the
earlier multiline and abbreviation examples alone cannot explain the gap.

Confirmation-fold uncertainty check (after all five folds finish)
---------------------------------------------------------------
scripts/bootstrap_stage_gains.py supports --exclude-fold 0 to resample folds
1--4 together while excluding the policy-selection fold. It reports included
folds and observed scores, uses the exact metric's role/space handling, and
rejects unknown prediction IDs and empty/invalid fold selections. The original
fold-1 and fold-2 comparisons in confirmation_folds12_bootstrap.json were rerun
with the updated utility: their deltas, intervals, and gain fractions reproduced
exactly. The older report's script checksum remains its historical snapshot.

Once the combined prediction file exists, run:

  .venv-kraken/bin/python scripts/bootstrap_stage_gains.py \
    --truth data/processed/train_folds.csv --exclude-fold 0 \
    --prediction model66=outputs/model66/oof/allfold_oof.csv \
    --prediction model35=outputs/model35/allfold_epoch07_verified/predictions/03_confusions_oof.csv \
    --samples 5000 --seed 20260903

Repeat against the combined 00_raw_oof.csv to evaluate correction gains. These
are conditional paired-row intervals, not training-seed or page-group intervals
and not a guarantee about hidden test data. The complete confirmation analysis
has not run yet; folds 3--4 are still needed.

Fold-3 confirmation result
--------------------------
The provenance-verified fold-3 report is complete. The fixed confusions stage
scores 0.843273606 versus raw Model36 0.808359660 and Model66 0.815051187 on
the same 819 rows. Its gains are +0.034913946 over raw and +0.028222419 over
Model66. A 2,000-draw paired row bootstrap against Model66 gives percentile
interval [0.022002091, 0.034331089], with all draws positive. Combined fixed
confusions predictions across folds 0--3 score partial OOF 0.842705334 versus
0.817162199 for Model66 on the same 3,276 rows. These are still partial results.

Full hashes and fold-1--3 intervals are recorded in
outputs/model35/confirmation_folds123_bootstrap.json. Fold 4 remains required
for the five-fold score, confirmation-fold aggregate interval, and promotion
gate. The fold-3 Model35 worker has exited successfully and may remain a zombie
until the orchestrator reaps it after blocking fold-4 training finishes; the
verified report, rather than process state alone, establishes completion.

The stage sequence also confirms the frozen endpoint across confirmation folds
1--3 together (2,457 rows): raw 0.805285633, context 0.836896928,
boundaries 0.840836186, confusions 0.841291978, approximate joins 0.839426390,
and supervised errors 0.839908427. Thus confusions is the best endpoint on
these confirmation rows as well as the original fold-0 selection data. The
exact aggregate metrics are saved in
outputs/model35/confirmation_folds123_stage_diagnostic.json. Fold 4 remains
required; no terminal stage was reselected from these confirmation results.

Five-fold completion and promotion, 2026-09-05
----------------------------------------------
All five provenance-verified held-out slices are complete. The frozen
Model13/confusions terminal stage scores 0.841721630 across all 4,095 rows,
versus 0.805638484 for raw Model36 and 0.817071245 for Model66. It improves on
Model66 in every fold: fold 0 0.846941606 vs 0.819849390, fold 1 0.839338500
vs 0.817787816, fold 2 0.841263454 vs 0.815957953, fold 3 0.843273606 vs
0.815051187, and fold 4 0.837792473 vs 0.816706849. The read-only numeric and
source-provenance gate in model35/check_candidate.py passes.

The untouched confirmation folds 1--4 score 0.840416017 for Model35 and
0.816376339 for Model66, a paired gain of 0.024039678. A 5,000-draw paired-row
bootstrap (seed 20260905) gives the 95% percentile interval
[0.020817211, 0.027261848], with every draw positive. Against raw Model36 on
the same confirmation rows, the gain is 0.036623764 with interval
[0.034414258, 0.038877713], again with every draw positive. These intervals
condition on the existing predictions and do not include training randomness,
page correlation, model-search multiplicity, or hidden-test distribution shift.

Authoritative artifacts and SHA256 values:

  outputs/model35/allfold_epoch07_verified/report.json
    47cdf95f98b86a6368671638dbbab7e12d14b15b0abaa5738d9b1d4e57e1c06b
  outputs/model35/allfold_epoch07_verified/predictions/03_confusions_oof.csv
    d2900d3fb208d75da74e93d4f261706638c1bcd1aff26f9171476147e9c8893d
  outputs/model35/allfold_epoch07_verified/bootstrap_confirmation_model66_vs_confusions_5000.json
    79af5e4f67b4ed35fc217566cee4f47ffc2ad780c42a8478160379e28b4f30f1
  outputs/model35/allfold_epoch07_verified/bootstrap_confirmation_raw_vs_confusions_5000.json
    5c22946e3c6781209c1451a4e1b29cfd9325b92dd7490f0ae1d951c8fd55471d

The diagnostic stages after confusions regress: approximate joins score
0.840208584 and supervised errors score 0.840715879 globally. They remain
excluded from deployment by terminal_policy.json. Model69's exploratory
cross-fold epoch selector scores only 0.803042175 raw and is not fully nested;
it is neither the deployment candidate nor a promotion gate.

This is strong conditional local evidence, not a public-score or prize
guarantee. At promotion time, the next authorized step was one all-data
Model36 training run, fixed epoch-7 export, signed train/test inference, and
guarded Model35 assembly; its completion is recorded below.

Final deployment candidate, 2026-09-05
--------------------------------------
The authorized all-data Model36 epoch-7 run and guarded Model35 assembly have
finished successfully. The single recommended submission is:

  outputs/model35/deployment_epoch07/submission.csv

It contains 1,374 rows in the official SampleSubmission.csv order, has one
Target column, and independently passes road_htr.submission validation. Its
SHA256 is:

  d676ac5dffb6bd37b5614243b4b1acb1ddadaadbee11b6ac379d68513caf2b3c

The deployment report is outputs/model35/deployment_epoch07/report.json with
SHA256 6237aced4d00998df47bd615ed833f89e01ee4cf97bcce41d75bf443638d11f5.
Both deployment status files say validated_candidate_not_uploaded. Nothing was
uploaded to Zindi. The 0.841721630 value is five-fold conditional OOF evidence,
not a claim about the unknown public or private leaderboard score.
