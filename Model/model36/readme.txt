MODEL 36 — FIXED EIGHT-EPOCH PP-OCRV6 CROSS-VALIDATION
======================================================

Purpose
-------
Model36 turns the completed Model22 fold-0 checkpoint audit into a fixed
cross-validation/deployment policy. The exact challenge metric selected
zero-based epoch 7 (eight completed epochs) on fold 0: raw multi-score
0.813019600 versus 0.731393742 for the frozen recognizer. Epoch 9 fell to
0.801538687, so Kraken's filename proxy is not used for this decision.

Protocol
--------
Folds 1, 2, 3, and 4 are each trained for exactly eight epochs. The optimizer,
seed, augmentation, preprocessing, compact codec, and ten-epoch cosine horizon
remain identical to Model22. Keeping cosine_epochs=10 makes their first eight
epochs the same optimization schedule used by the already-complete Model22
fold 0. No per-fold validation score is allowed to choose a different stopping
epoch for the claimed OOF result.

The fold-0 OOF component is
outputs/model22/checkpoint_evaluation/fold0/predictions/checkpoint_07-0.9209.csv.
The remaining components will live below outputs/model36/fold1 through fold4.
The three Model22 geometry exclusions remain model-local training exclusions;
external validation must still predict all 819 canonical rows per fold.

Train
-----
  PYTHONPATH=src .venv-kraken/bin/python model36/train.py --fold 1
  PYTHONPATH=src .venv-kraken/bin/python model36/train.py --fold 2
  PYTHONPATH=src .venv-kraken/bin/python model36/train.py --fold 3
  PYTHONPATH=src .venv-kraken/bin/python model36/train.py --fold 4

Deployment after OOF validation
-------------------------------
Only after the fixed five-fold OOF candidate and its fold-safe fusion beat the
current 0.817071245 Model66 OOF fallback, train all eligible labels for eight epochs:

  PYTHONPATH=src .venv-kraken/bin/python model36/train.py --all-data

Then infer Test.csv with model36/infer.py and apply only transformations fitted
without target-fold leakage. A prize or leaderboard score cannot be guaranteed;
the final submission decision must be based on the exact local metric and fold
stability, not the public leaderboard.

Status
------
Configuration and wrappers are ready. Fold 0 is inherited from the byte-pinned
Model22 epoch-7 artifact. Fold 1 completed all eight epochs on the local RTX
4050; exact checkpoint evaluation found epoch 5 at 0.804964212 and the frozen
epoch-7 policy at 0.800785088. Fold 2 completed training and exact evaluation:
epoch 7 scores 0.806714535 and epoch 4 scores 0.807326460. Model35 processing
is complete for folds 0--2. Fold 3's interrupted seven-checkpoint run is
preserved in outputs/model36/fold3_interrupted_20260905_024428. A fresh run
started in detached screen session road_model36_restart on 2026-09-05 at
02:48 local time, followed automatically by fold 4. The interruption cause
was not independently established; the missing process and incomplete
checkpoint matrix established the need for recovery.
The clean fold-3 restart has now completed all eight epochs and saved
checkpoint_07-0.9110.ckpt plus best_0.9243.safetensors. The best-file proxy is
not used to change the frozen epoch-7 policy. The detached runner successfully
handed off to exact checkpoint evaluation on all 819 fold-3 rows. That audit is
complete: fixed epoch 7 scores 0.808359660 raw multi-score (weighted WER
3.474664233, weighted CER 5.154893026), independently recomputed from its
prediction CSV. It also happens to be the best of the eight audited epochs;
the policy has not changed. Model66 scores 0.815051187 on the same heldout rows.
The raw prediction SHA256 is
94627bc8f4bdd0c6db0faa4162775775c2e89172653461e321b0f1ff21538153.
Epoch-7 inference on all 4,095 labeled rows completed and its manifest was
written. The orchestrator then started fold-3 Model35 CPU processing and fold-4
GPU training concurrently, as designed. Fold-3 correction is complete at
0.843273606; fold 4 has started and saved its first two checkpoints. Fold-4
training/evaluation/correction and all-data deployment remain incomplete; this
is not a complete five-fold OOF result.

The fixed raw recognizer across folds 0--3 scores 0.807220529 on 3,276 rows,
while Model66 scores 0.817162199 on exactly those rows. This partial comparison
is descriptive, not complete five-fold OOF. It confirms that Model36 raw is not
currently the promotion candidate: the preregistered Model35 confusions-stage
correction must be evaluated across all folds before any submission decision.

Checkpoint stability diagnostic
-------------------------------
After fold 3 evaluation completed, epochs 0--7 were combined across confirmation
folds 1--3 only (2,457 rows). The preregistered zero-based epoch 7 is currently
best in aggregate at 0.805285633. Epochs 4 and 5 score 0.804330497 and
0.804253096. Per-fold optima differ, so no fold-specific checkpoint policy is
introduced. Full results are in
outputs/model36/confirmation_folds123_epoch_diagnostic.json. This post-hoc
diagnostic does not change or provide an independent estimate for the frozen
fold-0 policy, and fold 4 remains unseen by it.

Final deployment commands (not yet run)
--------------------------------------
First run the read-only full-five-fold gate and inspect the paired validation
evidence. This gate recomputes scores and source provenance, but is not a
statistical confidence interval or a fully nested estimate:

  .venv-kraken/bin/python model35/check_candidate.py

Only after the candidate passes review, train the final model:

  .venv-kraken/bin/python model36/train.py --all-data

Then export the fixed zero-based epoch 7. export_checkpoint.py refuses missing,
duplicate, or abort checkpoints, requires the complete 0--7 matrix and a final
best-weights completion marker, and hashes the selected checkpoint, training
manifest, config, and converted weights. It never selects by filename score.
The output directory must be new, and no training or upload is performed:

  .venv-kraken/bin/python model36/export_checkpoint.py

Generate both inference files from that exact exported model:

  .venv-kraken/bin/python model36/infer.py \
    --checkpoint outputs/model36/deployment_epoch07/epoch07.safetensors \
    --input data/processed/train_folds.csv \
    --output outputs/model36/deployment_epoch07/train_predictions.csv \
    --no-constrain-charset --batch-size 1

  .venv-kraken/bin/python model36/infer.py \
    --checkpoint outputs/model36/deployment_epoch07/epoch07.safetensors \
    --input data/raw/Test.csv \
    --output outputs/model36/deployment_epoch07/test_predictions.csv \
    --no-constrain-charset --batch-size 1

Build the CPU-corrected submission using the same fixed recipe as validation:

  .venv-kraken/bin/python model35/build_submission.py \
    --oof-report outputs/model35/allfold_epoch07_verified/report.json \
    --training-manifest outputs/model36/foldall/run_manifest.json \
    --training-predictions outputs/model36/deployment_epoch07/train_predictions.csv \
    --test-predictions outputs/model36/deployment_epoch07/test_predictions.csv \
    --conversion-receipt outputs/model36/deployment_epoch07/epoch07.conversion.json \
    --output-dir outputs/model35/deployment_epoch07

The final artifact will be outputs/model35/deployment_epoch07/submission.csv.
These paths describe the intended workflow, not existing deployment artifacts.
No command here submits to Zindi or changes the running fold experiments.

Read-only auxiliary-decoder audit, 2026-09-05
--------------------------------------------
A completed fold-1 checkpoint was inspected on CPU with CUDA disabled:
outputs/model36/fold1/checkpoint_07-0.8368.ckpt contains 84 nrtr_head.* state
tensors with 19,921,920 total elements (including buffers, not a parameter
count). Its output projection has shape [1625, 512]. The corresponding exported
checkpoint_07-0.8368.safetensors has 418 tensors and no NRTR keys.

The installed Kraken source confirms this is an auxiliary training head:
train/ppocr.py builds/reinitializes it after codec resizing, trains it with
teacher-forced cross-entropy alongside CTC, and restores it from Lightning
checkpoints. lib/ppocr/network.py uses the CTC network for ordinary forward
inference. lib/ppocr/nrtr.py exposes causal decoder logits conditioned on shared
backbone features. The auxiliary head is initialized from scratch when loading
the pretrained recognizer; it has not inherited a pretrained language decoder.
Discarding it from the standard recognition export is intentional behavior,
not an identified export bug.

Installed-source SHA256 snapshots:
  train/ppocr.py: a6252ea3428047e969b5f39fdfcf1e3e06289b57b8877dbf0e469a3b21364043
  lib/ppocr/nrtr.py: 4367a8839b8809648e431bdff8ce778a3589732a66f165b6734ada6d32a9f1d0
  lib/ppocr/network.py: 60508b7cbf3fa941210ae419ebad7c04f4327ad66b6264290ae88e5f01b28032

Possible experiment AFTER Models35/36 finish: recover the saved auxiliary head
and evaluate autoregressive decoding or scoring of CTC candidate transcripts
on development fold 0. Match image preprocessing/backbone features, codec,
valid-column masks, BOS/EOS, and maximum decoding length. Freeze any candidate
selection rule before confirmation folds. This can reuse existing checkpoints
without retraining, but usefulness is unproven and it may hallucinate or perform
worse than CTC because it was trained only as auxiliary guidance. No NRTR
inference, new model, or score claim was produced by this audit. Keep the full
Lightning checkpoints; standard exports alone cannot recover this decoder.

Five-fold completion and deployment decision, 2026-09-05
--------------------------------------------------------
Fold 4 completed all eight epochs. Its fixed epoch-7 checkpoint is
checkpoint_07-0.9271.ckpt. Exact held-out evaluation scores it at 0.799320195
raw; the training filename score 0.9271 is only Kraken's proxy and must not be
reported as the challenge score. Model35's frozen confusions stage raises the
same fold to 0.837792473.

The complete fixed raw recognizer OOF is 0.805638484. The complete corrected
Model35 OOF is 0.841721630 and passes the strict per-fold/global gate against
Model66's 0.817071245. Confirmation folds 1--4 improve by 0.024039678 over
Model66, with a 5,000-draw paired interval [0.020817211, 0.027261848]. The
promotion decision is therefore to train one all-data model for exactly eight
epochs and export zero-based epoch 7. Per-fold proxy optima and the exploratory
Model69 selector do not alter this endpoint.

The completed fold pipeline log is outputs/model70/restart_fold3_pipeline.log
and ends with PIPELINE COMPLETE. At that checkpoint the final all-data
artifacts were still pending; their later completion is recorded below. No
submission has been uploaded.

Final inference is routed through src/road_htr/model36_inference.py, which
atomically adds the exact prediction CSV SHA256 to each completed manifest.
This strengthening is Model36-specific: src/road_htr/kraken_cli.py was restored
byte-for-byte to its historical pinned SHA256
50519a5a354b6c76b41822b93a6729b7399db30a30d14e6a77530b809d726d41,
avoiding dependency-receipt breakage in older Qwen models. The signer SHA256 is
2a63cecfa2b7bd2c19e097847ea158136215d5d55fef1daac777cf004ce15774.
Five signer tests plus the prior export, inference, and submission tests pass.

All-data deployment completion, 2026-09-05
------------------------------------------
The fixed eight-epoch all-data run completed and exported zero-based epoch 7
to outputs/model36/deployment_epoch07/epoch07.safetensors. The exported-model
SHA256 is d4107382f391479882003690138539176eb8e702d23bf1243ccff7637f15ff55.
Signed inference completed for all 4,095 training rows and 1,374 test rows:

  train_predictions.csv
    6ea5582fd03dcd35f38f7a8a0a499ccb8e011d3864069f59b7b1edd39c62c61e
  test_predictions.csv
    00407d9d37d5115153b3e7d52469b2be806538eb5bba7f7fb60a61bd52a9dfd2

These raw predictions feed Model35's fitted correction pipeline. Model36 alone
is not the recommended submission; the final guarded Model35 CSV is documented
in model35/readme.txt.
