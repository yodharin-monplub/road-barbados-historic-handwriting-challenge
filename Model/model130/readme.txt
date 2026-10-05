Model130: PP-OCRv6/Kraken CTC member restored for architecture diversity
========================================================================
Why: submission #2 (Model124, four LightOnOCR/Qwen members) scored CV 0.905544
and public 0.909996, while the lost four-source ensemble had the same CV
(0.9054) but a better public score (0.911129). The difference is what we no
longer had: a CTC recognizer. Model112 had already recorded that the vote gains
come from architecture diversity, and Model124's own ablation agreed - dropping
the weakest, most different member cost more than dropping a strong sibling.

This rebuilds that member with the organizer-approved checkpoint:
  PP-OCRv6 medium historical HTR, Zenodo 10.5281/zenodo.21788410
  SHA-256 15313b51ace64cbfa81f8f6ef25ad64f04e5a6fb7f7823e67b107527bc081ac9
    (verified byte-for-byte against MODEL_LICENSES.md after download)
  Apache-2.0, explicitly approved by Zindi on 2026-08-28 (discussions/34481)
  kraken 7.1.1, torch 2.14.0+cu130, rented RTX 3090

Training is the unchanged Model22 recipe (model22/config.yaml): compact codec
resize "new", 96px line height, batch 1, accumulation 8, AdamW 5e-5, cosine over
10 epochs, NFD normalization, augmentation on, seed 20260903.

Checkpoint policy: FIXED zero-based epoch 7 for every fold, converted with
"ketos convert". This follows Model36's documented policy rather than Kraken's
own best-checkpoint filename proxy, which is chosen on the held-out fold and
would leak. (Kraken still writes its own "best" file; it is not used.)

Fold-0 measurements
  Kraken epoch 7 standalone                      0.809760
  Kraken epoch 4 (Kraken's own pick) standalone  0.805632
  4 LightOn/Qwen members, round-1 anchor         0.909429
  + Kraken as a fifth member                     0.912483  (+0.003054)
  6 members, round-2 anchor, min_votes 2         0.911892
A member scoring 0.81 alone adds +0.003 to a 0.909 ensemble: its errors are
genuinely decorrelated from the LightOnOCR family, which is the entire point.

Run (on a Kraken host):
  bash model130/kraken_setup.sh      # env + approved weights, hash-checked
  bash model130/run_folds.sh 0 1 2 3 4

Complete five-fold results (2026-09-18)
--------------------------------------
Kraken member alone, all 4,095 rows: 0.805866 (fold 0 0.809760, fold 1 0.802347)

Ensembles on the complete OOF, anchored on model129_pseudo2:
  5 members without Kraken (pseudo2, pseudo1, prod, lora, qwen)   0.906780
  6 members with Kraken                                           0.909114
  5 members with Kraken, dropping the redundant round-1 pseudo    0.909376  <- selected
  6 members with Kraken, min_votes 3                              0.907091
Kraken is by far the weakest member and still worth +0.0026 on the complete
CV. Dropping the round-1 pseudo member (whose errors largely duplicate the
round-2 member's) gains a further +0.0003.

Four rules were compared on the same complete OOF and the best taken, so the
selected number carries a little optimism; the spread is 0.0023.

Deployed as submissions/model130_ensemble5_kraken.csv
  SHA-256 6a44b025638c0055f58692c0559c7a8abe7174d59df1ceffd0f7abefd4bb2ad5
  1374 rows, 164 lines changed versus the anchor's own fold consensus.

Interruption note: the vast.ai account hit zero credit mid-run (spent by the
user's unrelated jobs) and the host stopped after fold 1. Folds 0-1 survived on
disk; fold 2 had been restarted from scratch after an earlier mishap where two
ketos runs wrote into one fold directory and produced duplicate "-v1"
checkpoints. run_folds.sh now refuses to start while another ketos is live and
clears a fold directory before retraining, so the fixed epoch-7 checkpoint is
never ambiguous.

Status: complete and deployed.
