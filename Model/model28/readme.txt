MODEL 28 — leakage-safe cross-fitted postprocessing refit

Purpose
-------
Evaluate Models11-15-style text repair on any complete Model22 OOF prediction
CSV without choosing transformations on the fold being scored. This is an
evaluation/refit tool, not a claimed result: no Model22 OOF scores are recorded
here until genuine predictions are supplied and the program is run.

Protocol
--------
For each held-out fold f, Model28 starts from that fold's supplied OCR text and
fits the complete sequential pipeline using only rows whose fold is not f.
This exclusion covers:

* label vocabulary, unigram and bigram counts;
* every threshold/grid choice;
* inputs used to fit later stages; and
* recurring OCR-to-truth mappings.

The selected other-fold pipeline is then applied once to f. Held-out scores do
not control selection or later stages. Concatenating the five held-out outputs
therefore gives an honest postprocessing OOF evaluation, conditional on the
input OCR predictions themselves being genuine OOF predictions.

The script can verify IDs and fold handling, but it cannot prove checkpoint
provenance. Never give it Train predictions from an all-data model or from a
checkpoint that trained on the row being scored.

Sequential stages
-----------------
  raw                 supplied Model22 OOF text
  context             Model11 contextual one/two-edit lexical repair
  boundaries          Model12 exact split/merge repair
  confusions          Model13 valid-word contextual confusion repair
  approximate_joins   Model14 noisy joined-word repair
  supervised_errors   Model15 recurring OCR-error mappings

The transformation primitives and fixed search grids come from the existing
Models11-15 implementations. Every grid also receives an explicit disabled
candidate. Ties prefer fewer operations and then the declared grid order, so a
stage is not forced to alter its fitting rows when no candidate improves them.

Run
---
First combine one prediction file from each held-out fold into a complete CSV
in ``train_folds.csv`` order. Then run, using a new output directory:

  python model28/refit_oof.py \
    --truth data/processed/train_folds.csv \
    --predictions outputs/model22/oof_selected.csv \
    --output-dir outputs/model28/model22_selected

Model28 refuses a nonempty output directory and refuses any destination under
``outputs/model22``. It reads Model22 artifacts but never changes checkpoints
or active training outputs.

Artifacts
---------
``report.json`` contains the exact challenge metrics globally and per fold for
raw input and every sequential stage, together with each stage's delta and the
complete other-fold fitting audit. ``predictions/`` contains ordered OOF CSVs,
``changes/`` contains changed-row JSONL logs, and ``folds/`` contains selected
settings plus the supervised mapping learned without that fold.

The per-fold fitting reports deliberately include training-set search scores
for audit, but only the concatenated held-out scores in ``report.json`` are OOF
evidence. Inspect per-fold deltas before promotion; a small global gain that is
driven by one fold is not robust.

Deployment boundary
-------------------
Model28 does not silently turn its evaluation into a test submission. After a
stage family shows stable held-out gains, refit that approved recipe on all
honest OOF pairs for an all-data test model, or rebuild fold f's emitted recipe
from folds other than f before processing that fold model's test prediction.
Do not report a full-data refit's Train score as OOF.

No leaderboard feedback, external corpus, image metadata, or test label is
used. The score remains an estimate, not a leaderboard or prize guarantee.
