MODEL 30 — METRIC-AWARE PP-OCRV6 SAMPLING EXPERIMENT

Purpose
-------
Model30 is an isolated, fold-safe training-manifest experiment built on the
same PP-OCRv6 checkpoint, image geometry, optimizer, and data exclusions as
Model22. It changes only how often each train-side line is presented. It does
not modify Model22's configuration, prepared data, checkpoints, or outputs.

Why weighting may help
----------------------
The competition error is

  0.5 * (weighted_WER / 12 + weighted_CER / 55),

where every word edit is weighted by sqrt(reference word count), and every
character edit by sqrt(reference character count). Kraken's PP-OCRv6 CTC term
uses PyTorch CTCLoss(blank=0, reduction="mean"). That reduction first divides
each sample's unnormalized CTC loss by its CTC target length. Therefore a
manifest frequency that approximates the challenge's per-line importance is

  ctc_length * 0.5 * (
      sqrt(word_count) / (12 * sum_train_sqrt_word_count)
    + sqrt(char_count) / (55 * sum_train_sqrt_char_count)
  ).

This is a deliberately conservative surrogate: CTC label errors do not map
exactly to word edits, and PP-OCRv6 also has an auxiliary loss. OOF evidence,
not the derivation alone, decides whether to promote it.

Fold-safety and validation policy
---------------------------------
For a fold run, held-out rows are removed before Model30 reads any Target
value. All normalization sums, Unicode CTC lengths, clipping, and integer
repetitions use training rows only. Changing a held-out label cannot change
the weighted training manifest. Validation receives the ordinary unweighted
Model22-equivalent manifest, with each eligible validation path listed once.

The default policy normalizes raw importances to mean 1, rescales and clips
them to [0.5, 2.0] while retaining mean 1, then creates a deterministic 2x
manifest by largest-remainder integer apportionment. Every eligible line is
retained at least once. A stable SHA-256 ordering avoids blocks of duplicate
examples. The `uniform` mode (also accepted as `identity`) is a true control:
it preserves the base manifest byte order with exactly one copy of every row.

Compute control
---------------
The wrapper scales epochs by unique_rows / weighted_manifest_rows. With the
default 2x repetition budget, Model22's 10 epochs become 5 weighted epochs.
For fold 0 this preserves 32,730 sample presentations and gives 4,095 estimated
optimizer steps versus Model22's 4,100 (batch size 1, accumulation 8), a ratio
of 0.998780. Cosine schedule epochs are scaled by the same rule.

Preparation and dry runs
------------------------
Use the Kraken virtual environment. These commands write only Model30
manifests and JSON receipts; they do not start training:

  .venv-kraken/bin/python model30/train.py --fold 0 --dry-run
  .venv-kraken/bin/python model30/train.py --fold 0 --mode uniform --dry-run

The metric run writes under `outputs/model30/metric_clipped/fold0`; the control
writes under `outputs/model30/uniform/fold0`. Each directory contains an atomic
`weighting_receipt.json` and `run_manifest.json` with source/pretrained hashes,
row-level lengths and weights, base/weighted manifest hashes, the exact Kraken
command, and estimated compute. Existing checkpoint or best-weight artifacts
make the wrapper refuse reuse of that output directory.

Training (not started)
----------------------
After inspecting the dry-run receipts, the intended fold-0 experiment is:

  .venv-kraken/bin/python model30/train.py --fold 0

Do not train all folds or make a submission unless exact fold-0 OOF improves
over the corresponding Model22 checkpoint under the competition scorer.

Status
------
Configuration, deterministic generator, wrapper, receipts, and tests created.
No Model30 training has been started.
