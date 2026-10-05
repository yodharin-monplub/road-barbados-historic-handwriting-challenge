Model122: height-normalized full fine-tuning of LightOnOCR-2-1B
==============================================================
Context (2026-09-18): the Linux workstation that held every trained adapter and
every saved prediction is gone. Only code, readmes and public base weights
survived, so the whole recognizer stack is being rebuilt from scratch on a
Windows laptop (RTX 4050 6GB) plus a rented vast.ai GPU. The last known public
score is Model112 = 0.911129378; nothing in outputs/ exists to reproduce it.

Two changes versus the old Model73/92/114 recipe, both chosen before any new
score was seen:
1. FULL fine-tuning (every vision, projector and decoder weight) instead of a
   rank-64 LoRA. Model117 showed lower LoRA rank hurt, and Model73's loss rose
   after 400 steps, so adapter capacity - not training length - was the limit.
2. HEIGHT-NORMALIZED input. The Pixtral processor only ever downscales to
   longest_edge 1540, so a 1119x65 line arrived about 5 patches tall and a
   line of aspect ratio 17 could never exceed ~90px. Every image is now resized
   to a fixed text height (target_height, default 128px) with the aspect ratio
   preserved and longest_edge raised to 2800, roughly 570 image tokens instead
   of 258. Resolution was previously varied only for the Qwen2-VL family
   (Models 40/43/48/50), never for LightOnOCR.
   This uses only the pixels of the supplied crop, applied identically to every
   row; image geometry is never used to select between predictions, which
   Zindi's computer-vision rules prohibit.

Metric note: Zindi divides weighted WER by 12 and weighted CER by 55, so one
word edit costs about 4.6x one character edit. Whole-word correctness -
including case and attached punctuation - is what the score rewards.

Protocol
- Frozen before training: 8 epochs, cosine schedule with 5% warmup, lr 2e-5
  (decoder) and 1e-5 (vision tower/projector), effective batch 16, bf16
  autocast over fp32 master weights, grad clip 1.0, Model92's mild
  augmentation plus +/-10% height jitter, seed 20260918.
- Scoring uses the exact competition metric on the held-out fold, at epochs
  2/4/6/8. The deployment checkpoint is the FIXED final epoch, so per-fold
  checkpoint cherry-picking cannot leak; intermediate scores are diagnostics
  for choosing the epoch count once, on fold 0.
- config_full_h128.json is the intervention; config_full_native.json is the
  matched control at native resolution (target_height 0, longest_edge 1540).
  Same seed, schedule and data, so the difference isolates resolution.
- Fold 0 is a development fold with a long selection history in this project;
  its result is NOT unbiased CV. A recipe advances only if it then holds on
  folds 1-4, which produce the complete out-of-fold estimate.
- No external data, no test labels, no manual edits, no image metadata.
  data/processed/test_ids_derived.csv is the 1374 image IDs absent from
  Train.csv, used only because OneDrive cannot yet hydrate Test.csv; the real
  SampleSubmission.csv is still required before any submission is written.

Files
  train.py                  self-contained trainer/scorer (full or LoRA)
  config_full_h128.json     intervention: full FT, 128px text height
  config_full_native.json   matched control: full FT, native resolution
  config_lora_local.json    small LoRA config used to smoke-test on the laptop
  vast_setup.sh             one-time GPU host setup (torch 2.14 cu130 + base)
  vast_unpack.sh            unpack uploaded code/images on the host
  vast_queue.sh             sequential job runner, skips finished folds

Run
  python model122/train.py --config model122/config_full_h128.json --fold 0
  python model122/train.py --config model122/config_full_h128.json --fold all --predict-test

Fold-0 results (2026-09-18, 819 held-out rows, exact competition metric,
standalone OCR with no ensemble and no text repair)
------------------------------------------------------------------------
  recipe                                     epochs  score
  full fine-tune, 128px height-normalized      3     0.870201  (4-epoch run)
  full fine-tune, native resolution            4     0.899716
  LoRA rank 64, native resolution              2     0.896569
  full fine-tune native, 8-epoch schedule      3     0.895066  (stopped)
  production repeat of full native + beam 4    4     0.901168

THE RESOLUTION HYPOTHESIS WAS WRONG. Upscaling line crops to a 128px text
height cost -0.029 against the matched native-resolution control under an
otherwise identical recipe. LightOnOCR is pretrained at its own document scale
and does not transfer to enlarged line crops; longest_edge stays 1540 and
target_height stays 0 in every later configuration. Recording this so the
experiment is not repeated.

What did hold: full fine-tuning beats the rank-64 LoRA of the old Model73/92
recipe by about +0.003 at the same resolution and the same data, and beam-4
decoding adds a further +0.0013 over greedy (0.899821 -> 0.901168). For
reference the old best standalone model on this fold was 0.8954, so the
rebuilt single model is slightly ahead of what was lost.

The 8-epoch schedule was stopped at epoch 4: training loss had fallen to 0.026
with fold-0 scores flat-to-down (0.8856 / 0.8951 / 0.8909 at epochs 2/3/4), so
4 epochs is fixed for every fold. The mid-schedule scores of different-length
cosine runs are not comparable to each other - only end-of-schedule scores are.

Production: config_prod.json (identical recipe, 4 epochs, one eval) runs all
five folds with --predict-test, and config_prod_beam.json adds beam-4 decoding
on fold 0 so the beam-versus-greedy delta is measured once, not tuned per fold.

Complete five-fold result (2026-09-18)
-------------------------------------
Per-fold: 0.899821 / 0.899198 / 0.895009 / 0.896928 / 0.892084
Complete OOF over all 4,095 clean rows: 0.896608
  weighted WER 1.821211, weighted CER 3.025931
This is a single standalone model with no ensemble and no text repair. The old
best standalone was about 0.8954 on fold 0 and the old four-source ensemble
scored 0.9054 OOF, so this recovers the single-model level that was lost with
the old machine.

Submission: submissions/model122_fold_vote_maj3.csv, a strict majority vote of
the five fold models on the test set (anchor fold 0 beam-4, 419 of 1374 lines
changed, one all-blank row recovered from the only fold that read it).
SHA-256 2de048caf7a9e6efff3255d38732fd073c66d347212170a794e87bf3732c1b97.
User-reported public score: 0.907524369, against 0.911129378 for the old
(now unreproducible) four-source ensemble. A vote of five same-recipe fold
models cannot be validated out-of-fold, so it is a deployment choice, not a
measured CV result; the leaderboard gap to the old ensemble is consistent with
this model family needing the other architectures back.

Status: complete; superseded by the multi-member ensemble work (model124).
