Model46 — frozen Model35 language-repair transfer, conditional fold 0 only

What it does
  Given one complete fold-0 OCR/VLM prediction CSV, Model46 rebuilds the
  Model11 contextual, Model12 boundary, and Model13 confusion states from
  ground-truth labels in folds 1--4 only. It then applies the already frozen
  Model35 fold-0 sequence once: raw -> context -> boundaries -> confusions.
  It stops there. It does not run approximate joins or supervised errors.

No candidate-side fitting or tuning
  The candidate supplies only its 819 fold-0 prediction strings. It is never
  used to fit a language state, choose a threshold, or choose a terminal stage.
  Every state constructor receives exactly the same truth-only fold-1--4 label
  sequence. The recipe, terminal policy, Model35 report, truth table and all
  relevant implementation sources are SHA-256 pinned in config.json.

Scope
  Scores are exact competition scores on fold 0 only and are explicitly
  conditional evidence for this transfer. They are not a full OOF score and
  never compare/select candidate models. An optional lineage receipt must bind
  the supplied candidate CSV by its resolved path or SHA-256.

CPU audit
  python model46/audit.py

Run once a Model39/43 fold-0 validation CSV exists
  python model46/transfer.py \
    --predictions outputs/model39/fold0/validation/epoch_001.csv \
    --output-dir outputs/model46/model39_fold0_transfer

The candidate must contain an explicit `prediction` column and exactly the 819
canonical fold-0 IDs (an optional fold column may contain only 0). Outputs are
new-only under outputs/model46: raw/context/boundaries/confusions CSVs, change
JSONL files, hashes, exact stage scores and the conditional report.
