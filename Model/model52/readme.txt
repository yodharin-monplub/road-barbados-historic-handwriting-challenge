MODEL 52 — frozen guarded Model50/Model25 selector
==================================================

Purpose
-------
Model50 is a heterogeneous vision-language recognizer, but generative OCR can
occasionally emit fluent hallucinations. Model52 makes Model25 the safe
fallback and accepts a Model50-derived line only when a fixed structural guard
passes and a fold-excluded character language model gives the VLM line a clear,
distance-adjusted advantage.

Inputs and frozen order
-----------------------
The intended VLM input is Model50 after the already frozen Model46 terminal
repair (context -> boundaries -> confusions). The fallback is Model25. The
policy in config.json was fixed before any Model50 held-fold predictions
existed. It is not tuned after seeing Model50's fold-0 score.

For each line, the selector rejects the VLM candidate when it is blank, contains
a chat-role marker, repeats a four-word n-gram, has an extreme character/token
length ratio relative to Model25, or exceeds 0.60 normalized character edit
distance from Model25. A remaining VLM candidate must beat Model25 under the
order-5 fold-excluded label character LM by at least:

  0.04 + 0.08 * normalized_character_distance

Every decision receives only the two prediction strings and the frozen LM. It
does not receive the ID, image geometry, test membership, or held-out target.
Held-out labels are used only after all decisions are frozen to compute the
challenge score.

Current status — rejected by negative control
---------------------------------------------
Implementation, 11 focused tests, and the CPU dry audit pass. The frozen policy
was then run once against the known rejected Model39 epoch-1 output after the
same Model46 terminal repair. It accepted 213/819 VLM lines and degraded the
Model25 fold-0 slice from 0.817450092 to 0.760894139. The Model39 input itself
scored 0.357686046.

This demonstrates that train-label fluency plus coarse structural guards cannot
reliably distinguish a correct transcription from a fluent hallucination.
Model52 is rejected and must not be used for a submission or silently retuned.
Its report is outputs/model52/model39_negative_control/report.json. A stricter
successor must use local agreement rather than whole-line replacement.

Audit
-----
  PYTHONPATH=src python model52/audit.py

Historical held-fold command
----------------------------
The following interface remains reproducible, but the policy is rejected:

  PYTHONPATH=src python model52/evaluate.py \
    --fallback outputs/model25/allfold_oof.csv \
    --vlm outputs/model46/model50_epoch001_transfer/predictions/03_confusions_fold0.csv \
    --fold 0 \
    --output-dir outputs/model52/model50_epoch001

Deployment
----------
Only after a promoted all-data Model50 model, its test inference, Model46-style
all-data repair, and Model25 test candidate exist:

  PYTHONPATH=src python model52/apply.py \
    --fallback outputs/model25/test.csv \
    --vlm <model50_repaired_test.csv> \
    --output-dir outputs/model52/test_candidate \
    --sample data/raw/SampleSubmission.csv \
    --submission submissions/model52_guarded_hybrid.csv

Scope
-----
One held fold is conditional evidence, not complete OOF. Model52 is CPU-only,
uses no external corpus or test labels, performs no manual row edits, and never
routes on prohibited image metadata. The negative result is retained so the
same unsafe selection idea is not repeated.
