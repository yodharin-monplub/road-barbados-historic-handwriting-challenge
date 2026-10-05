Model98: guarded Model88/Qwen3-VL selector
==========================================

Model95 may be useful as a diverse OCR witness even if its raw score is below
Model88.  Model98 tests that hypothesis with the existing dependency-free,
metric-weighted logistic selector and its already-frozen Model88 configuration.

On development fold 0, every reported row is excluded from both the selector
fit and its target-language model through five-way inner cross-fitting.  Model98
must gain at least 0.001 before any all-fold work is authorized.  A passing fold
0 result still requires untouched-fold confirmation.

Run after Model95 finishes:

  python3 model98/evaluate_fold.py \
    --direct outputs/model95/fold0/direct/predictions.csv

No submission is created automatically.
