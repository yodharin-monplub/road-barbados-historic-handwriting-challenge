Model 78: leakage-safe correction search for Model73
====================================================

Model78 applies the established sequential text-repair pipeline to the complete
signed Model73 OOF predictions. For outer fold f, every recipe, lexicon,
threshold, and supervised mapping is fitted only on folds other than f. The
heldout fold is transformed once and used only for scoring. Every stage grid
contains an explicit no-op candidate.

Run:

  PYTHONPATH=src python3 model78/refit_oof.py \
    --truth data/processed/train_folds.csv \
    --predictions outputs/model76/model73_step400_oof.csv \
    --output-dir outputs/model78/model73_step400

Promote only a terminal stage whose full five-fold gain is robust. This model
does not use test labels and never creates or uploads a Zindi submission.

Status
------

Completed and retired. Raw Model73 is 0.8894664158. The sequential stages score
0.8832303119 (context), 0.8829132044 (boundaries), 0.8828939017
(confusions), 0.8826428530 (approximate joins), and 0.8820800504
(supervised errors). Every stage is below raw, so none is eligible for test
deployment. The complete report remains at
`outputs/model78/model73_step400/report.json` for negative-result provenance.
