MODEL 53 — train-attested aligned-token VLM/OCR fusion
======================================================

Purpose
-------
Model52 showed that fluent whole-line VLM selection is unsafe. Model53 always
starts from Model25 and allows the Model50/Model46 candidate to propose only an
isolated one-for-one word replacement. It can never copy a VLM insertion,
deletion, multiword rewrite, or whole line.

Frozen policy
-------------
The policy was frozen before Model50 produced any held-fold prediction. Tokens
are aligned with a deterministic sequence matcher. A proposed VLM token is
accepted only when all of these conditions hold:

* exactly one fallback token aligns to exactly one VLM token;
* at least the immediately preceding or following token agrees exactly;
* leading and trailing punctuation agree;
* alphabetic cores differ by at most one edit for short words or two edits for
  words of at least six characters;
* the VLM word occurs at least five times in fold-excluded training labels;
* if the fallback word is also attested, the VLM word is at least four times as
  frequent; and
* fold-excluded unigram/bigram context score improves by at least 4.0 with
  context weight 4.0.

The decision API receives only the two strings and the fold-excluded token
counts. IDs, image geometry, test membership, and held-out targets are absent.
Original fallback whitespace is byte-preserved outside an accepted token span.

Status — rejected by negative control
-------------------------------------
Nine focused tests and the CPU dry audit pass. The frozen policy was then run
once against the rejected Model39 epoch-1 output after Model46. It changed only
20 tokens in 19/819 rows, but still reduced the Model25 fold-0 slice from
0.817450092 to 0.816875829 (-0.000574262). Model53 is therefore rejected as a
general VLM safety layer and must not be submitted or silently retuned. The
report is outputs/model53/model39_negative_control/report.json.

Commands
--------
  PYTHONPATH=src python -m unittest tests.test_model53_vlm_ocr_token_fusion -v
  PYTHONPATH=src python model53/audit.py

Historical interface after Model50 epoch 1 and Model46 transfer:

  PYTHONPATH=src python model53/evaluate.py \
    --fallback outputs/model25/allfold_oof.csv \
    --vlm outputs/model46/model50_epoch001_transfer/predictions/03_confusions_fold0.csv \
    --fold 0 \
    --output-dir outputs/model53/model50_epoch001

Model53 is CPU-only. A fold-0 result is conditional evidence, not full OOF.
The negative result is retained to prevent repeating unsafe VLM-copy logic.
