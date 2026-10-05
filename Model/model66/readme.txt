MODEL 66 — exact three-token-context substitution repair
========================================================

Purpose
-------
Model66 repairs a single token only when the same wrong token recurs under the
same exact left and right tokens. This captures errors that are unsafe as global
substitutions but reliable inside a repeated historical phrase.

Leakage control
---------------
Mappings come only from one-token difflib replacement blocks. An exact
left/current/right context needs support at least two and precision at least
80%. Every held-fold map excludes that fold; deployment uses all 4,095 honest
Model65 OOF pairs and no test labels.

Measured result
---------------
Model66 changes 22 OOF rows and raises Model65 from 0.816914198 to
0.817071245 (+0.000157047), with every fold improving. Its seeded 5,000-sample
incremental interval is [+0.000024860, +0.000286774], positive in 99.22% of
resamples. The cumulative Model25-to-Model66 gain is +0.002923731 with interval
[+0.002366053, +0.003478743], positive in every resample.

The validated candidate changes five rows after Model65 and has SHA-256
d156a1852bb578dfc9755107ecaba4d00a1f1fec49c6b62b284417917292d956.

Reproduce OOF:
  PYTHONPATH=src python model66/contextual_substitution_repair.py oof \
    --predictions outputs/model65/oof/allfold_oof.csv \
    --output-dir outputs/model66/oof

Build the local candidate:
  PYTHONPATH=src python model66/contextual_substitution_repair.py apply \
    --training-predictions outputs/model65/oof/allfold_oof.csv \
    --predictions outputs/model65/test_candidate/test.csv \
    --output-dir outputs/model66/test_candidate \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model66_contextual_substitutions.csv

This is CPU-only and deterministic. No leaderboard score is claimed.
