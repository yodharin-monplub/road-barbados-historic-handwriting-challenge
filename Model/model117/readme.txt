Model117: lower-rank LightOnOCR adaptation
========================================
Hypothesis: reduce adapter capacity to limit overfitting on the small dataset.
Model73's validation loss worsened beyond400 steps; additional training alone
is therefore not the first next experiment. Model116 shuffled-word augmentation
lost0.002566 standalone vs its matched control and did not pass its screen.

Model117 starts fresh from the pinned LightOnOCR base. Relative to Model114,
only LoRA rank changes64->16 and alpha128->32 (same alpha/rank scaling of2).
Seed20260914;3276 original training rows,819 heldout fold0 rows; no synthetic
or external data. Same full-image augmentation, bf16, batch1/accumulation16,
400 steps, learning rate0.0001,20 warmup steps, cosine schedule. Fixed final
checkpoint400; no selection over intermediate validation scores.

Primary comparison: standalone vs matched rank64 Model114, and replacement of
only the Model92 voter in Model112's frozen family-aware word ensemble. Keep
Model73/107/Kraken and the original Model104 anchor unchanged. No extra votes.
Advance screen: ensemble gain>=0.0005 vs Model112 and standalone gain>=0.001
vs Model114. Passing is permission to consider full-fold confirmation, not
proof of a leaderboard improvement or an automatic submission.
All reused folds have development history; do not call this untouched CV.

Commands from workspace root with .venv-lighton/bin/python:
  model117/experiment.py prepare
  model117/experiment.py check
  scripts/run_model117.py
Uses the verified Model114 evaluation implementation through a process-local
protocol-path override. Original files and protocols are not changed. New
configuration, supervisor, evaluation entrypoint and baseline artifacts are
hash-pinned before training. Runtime versions must match Model114.
Local CUDA only; supervisor waits for GPU availability without interrupting
other jobs and checks running work every600 seconds. No cloud costs.
Artifacts: outputs/model117/status.json, train.log, fold0_oof_step400.csv,
report.json and replacement_ensemble_fold0.csv. No score exists at creation.

Completed pilot (2026-09-11)
----------------------------
All400 training steps and819 heldout predictions completed without blanks.
Standalone0.8772442261 vs matched Model1140.8917072302: -0.0144630040.
Replacement ensemble0.9045402309 vs Model1120.9054038514: -0.0008636206.
The predefined screen failed. Do not expand rank16 under this fixed400-step
recipe. This does not establish that all low-rank schedules would fail.
No submission produced; local training service completed and released CUDA.
