MODEL 64 — exact-span repair refitted after Model63
===================================================

Model64 refits Model61's exact one/two-token recurring replacement rule on the
safer Model63 OOF vector. Each held-fold map still excludes that fold, while
the deployment map uses all 4,095 honest OOF prediction/label pairs.

It changes 65 OOF rows and raises Model63 from 0.816401608 to 0.816746101
(+0.000344493), improving every fold. Its 5,000-sample interval is
[+0.000108248, +0.000575080], positive in 99.84% of samples. It changes 26
rows on the corrected test lineage.

Reproduce OOF:
  PYTHONPATH=src python model64/refit_exact_span_repair.py oof \
    --predictions outputs/model63/oof/allfold_oof.csv \
    --output-dir outputs/model64/oof

Build local test stage:
  PYTHONPATH=src python model64/refit_exact_span_repair.py apply \
    --training-predictions outputs/model63/oof/allfold_oof.csv \
    --predictions outputs/model63/test_candidate/test.csv \
    --output-dir outputs/model64/test_candidate

This is CPU-only and deterministic. No leaderboard score is claimed.
