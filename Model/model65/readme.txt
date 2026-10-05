MODEL 65 — context insertion repair refitted after Model64
==========================================================

Model65 refits Model62's exact left/right-context insertion rule on Model64.
Every held-fold map excludes that fold; deployment uses all 4,095 honest OOF
pairs and no test labels.

Measured result
---------------
Model65 changes nine OOF rows and raises Model64 from 0.816746101 to
0.816914198 (+0.000168097), improving every fold. Its 5,000-sample incremental
interval is [+0.000071633, +0.000283833], positive in every sample. The
cumulative Model25-to-Model65 gain is +0.002766684 with interval
[+0.002228603, +0.003302236], positive in every sample.

The candidate changes two rows after Model64, validates against the sample,
and has SHA-256
0950d17c6197698d7ee3e59ecad424cd21b91d869444090f20cd7b8056904fa4.

Reproduce OOF:
  PYTHONPATH=src python model65/refit_context_insertion_repair.py oof \
    --predictions outputs/model64/oof/allfold_oof.csv \
    --output-dir outputs/model65/oof

Build the local candidate:
  PYTHONPATH=src python model65/refit_context_insertion_repair.py apply \
    --training-predictions outputs/model64/oof/allfold_oof.csv \
    --predictions outputs/model64/test_candidate/test.csv \
    --output-dir outputs/model65/test_candidate \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model65_gated_exact_context.csv

This is CPU-only and deterministic. No leaderboard score is claimed.
