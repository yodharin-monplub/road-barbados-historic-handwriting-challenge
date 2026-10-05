MODEL 54 — fold-safe unanimous final-token canonicalization
===========================================================

Deployment correction (2026-09-04)
----------------------------------
The original `test_candidate` and named submission inherited the raw-source
lineage error documented in Model23 and must not be submitted. The corrected
stage is `outputs/model54/rebased_test/test.csv`, rooted at the verified
postprocessed Model19 vector; it changes 13 test rows. Model66 is the current
fully receipted descendant.

Purpose
-------
Model25 still contains recurring case and edge-punctuation variants in the last
token of a crop. Model54 groups training-label final tokens by their casefolded
ASCII alphabetic core after stripping only leading/trailing nonletters. A
prediction's final token is replaced only when one exact training spelling is
unanimous with at least 15 occurrences.

Leakage control
---------------
Every OOF row uses a map built from the other four outer folds. The test map
uses all clean training labels. The mapping receives no image, ID, OCR
confidence, test membership, or held-out target. Only the final non-whitespace
span can change; all preceding bytes and whitespace are preserved.

Measured result
---------------
The frozen rule changes 24/4,095 rows and raises complete five-fold OOF from
0.814147514 to 0.814251941 (+0.000104427). All five folds improve:

  fold 0  0.817504017  (+0.000053926)
  fold 1  0.814154708  (+0.000217790)
  fold 2  0.813026611  (+0.000042096)
  fold 3  0.812382009  (+0.000115689)
  fold 4  0.814188355  (+0.000092510)

The gain is small; Model54 is a safer incremental candidate, not evidence that
the larger Model50 experiment will succeed. No leaderboard score is claimed.
A seeded 5,000-sample paired row bootstrap gives a 95% percentile interval of
[+0.000014319, +0.000200304], with positive gain in 98.92% of resamples.

Reproduce OOF
-------------
  PYTHONPATH=src python model54/restore_final_tokens.py oof \
    --predictions outputs/model25/allfold_oof.csv \
    --output-dir outputs/model54/oof

Build local test candidate
--------------------------
  PYTHONPATH=src python model54/restore_final_tokens.py apply \
    --predictions outputs/model25/test_from_stage15.csv \
    --output-dir outputs/model54/rebased_test

Artifacts
---------
  outputs/model54/oof/report.json
  outputs/model54/oof/allfold_oof.csv
  outputs/model54/rebased_test/report.json
