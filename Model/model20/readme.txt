MODEL 20 — iterative recurring-error repair (controlled rejection)

Purpose
-------
Test whether a second inner-cross-fitted pass of Model 15 can discover residual
OCR-to-truth substitutions exposed by the first pass.

Leakage control
---------------
As in Model 15, every evaluated row is corrected by a mapping learned from the
other deterministic inner groups inside its label-isolated outer fold. Test
mappings would be learned only from supplied train labels and OOF predictions.

Result
------
On the Model 18 chain, a second pass moved five-fold OOF from 0.808761727 to
0.808815257 (+0.000053530), but regressed folds 0 and 3 and left fold 4
unchanged. This is too small and inconsistent to promote. The experiment is
retained as negative evidence; do not use it in a final submission unless it
later passes paired-bootstrap and untouched-model checks.

Reproduce
---------
  python model15/tune_supervised_errors.py \
    --truth data/processed/train_folds.csv \
    --predictions outputs/model18/allfold_stage15.csv \
    --output-config outputs/model20/allfold_second_pass_config.json \
    --output-predictions outputs/model20/allfold_oof.csv
