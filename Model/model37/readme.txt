MODEL 37 — TOKEN-FUSION / ROW-SELECTOR STACKING CHECK
=====================================================

Purpose
-------
Audit whether Model34's conservative token-fused transcript could safely be a
primary candidate inside Model33. The exploratory command completed, but the
subsequent composition audit rejected the nesting: Model34 and Model33 use
different deterministic partitions. A Model34 rule for one row can therefore
depend on labels of that row's Model33-held-out peers, leaking held-group
information into Model33 features. Own-row exclusion alone is insufficient.

Result (fold 0 conditional experiment)
--------------------------------------
  Model34 token fusion                 0.819147344
  Model22 epoch-7 raw                  0.813019600
  Model37 inner-cross-fitted selector  0.841910806

The exploratory program printed 0.841910806, 0.000234335 below Model33's direct
selection between Model25 and Model22 (0.842145141). **That number is not a
valid nested held-out score and must not be reported as OOF.** Model37 is
rejected on provenance before its apparent metric is considered. Strict reuse
would require regenerating Model34 separately after removing each Model33-held
group, or freezing Model34 on independent outer-fold data.

Reproduce
---------
  PYTHONPATH=src .venv-kraken/bin/python model33/evaluate.py \
    --truth data/processed/train_folds.csv \
    --candidate-a outputs/model34/fold0_crossfit.csv \
    --details-a outputs/model8/allfold_full_constrained.jsonl \
    --candidate-b outputs/model22/checkpoint_evaluation/fold0/predictions/checkpoint_07-0.9209.csv \
    --details-b outputs/model22/checkpoint_evaluation/fold0/predictions/checkpoint_07-0.9209.jsonl \
    --config model33/config.json --single-fold 0 \
    --output-dir outputs/model37/fold0_inner_crossfit

Artifact
--------
`outputs/model37/fold0_inner_crossfit/report.json` is retained only as a failed
composition-audit artifact. Its generic split flags do not authenticate the
nested lineage of candidate A; do not promote or submit it.
