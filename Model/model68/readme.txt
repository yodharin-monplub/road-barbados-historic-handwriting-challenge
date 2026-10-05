Model68 - exact full-line translation memory

Purpose
-------
When a complete, case-sensitive Model66 prediction exactly matches a prediction
seen on another training fold, replace it with that prediction's unanimous
reference transcription. No fuzzy matching or case normalization is allowed.

Leakage controls
----------------
Each held-out fold uses mappings fitted only on the other four folds. Test
mappings use only genuine Model66 OOF prediction/reference pairs. Ambiguous
source strings are rejected: the selected reference must account for 100% of
observations. A single exact training match is allowed because the entire line,
including spelling, case, punctuation, and spacing, must match.

Commands
--------
OOF:
  env PYTHONPATH=src .venv-kraken/bin/python model68/exact_line_memory.py oof \
    --predictions outputs/model66/oof/allfold_oof.csv \
    --output-dir outputs/model68/oof

Test:
  env PYTHONPATH=src .venv-kraken/bin/python model68/exact_line_memory.py apply \
    --training-predictions outputs/model66/oof/allfold_oof.csv \
    --predictions outputs/model66/test_candidate/test.csv \
    --output-dir outputs/model68/test_candidate \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model68_exact_line_memory.csv

Measured result
---------------
Five-fold OOF rises from 0.8170712445762045 to 0.8170888934942640
(+0.0000176489180594). Folds 0 and 1 improve; folds 2--4 are unchanged. Four
OOF rows match: two improve, one regresses, and one ties. A seeded 5,000-sample
paired bootstrap gives [-0.000019653, +0.000068494], positive in only 76.36%
of resamples. The stage is therefore not promoted. The all-data mapping finds
no exact test prediction, so the generated submission is byte-identical to Model66
(SHA256 d156a1852bb578dfc9755107ecaba4d00a1f1fec49c6b62b284417917292d956).
