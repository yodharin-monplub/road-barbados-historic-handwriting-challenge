Model96: manuscript-order/context audit
=======================================

Purpose
-------
Test the hypothesis that IDs or supplied row order identify adjacent manuscript
lines.  If that were true, lines could be joined into documents, corrected with
context, and split back to their original IDs.

This model is analysis-only.  It must not create a submission until a candidate
page/order key produces a repeatable out-of-fold gain.  Grammar modernization is
also forbidden because the target preserves seventeenth-century spelling,
capitalization, abbreviations, and transcription marks.

Run
---

  python3 model96/analyze_order.py

Outputs
-------

  outputs/model96/order_report.json
  outputs/model96/train_csv_order_indexed.txt
  outputs/model96/train_csv_order_plain.txt
  outputs/model96/test_csv_order_indexed.txt
  outputs/model96/test_csv_order_plain.txt
  outputs/model96/archive_order_indexed.txt

The indexed versions preserve the exact row-to-ID mapping.  The plain versions
are convenient for reading but must never be split by sentence or token count.

Promotion rule
--------------
Do not ask an LLM to correct the concatenated test file unless manuscript
adjacency is independently established.  Any later corrector must return one
record per input ID, preserve order and line count exactly, and beat Model88 on
held-out folds before it can be applied to test.
