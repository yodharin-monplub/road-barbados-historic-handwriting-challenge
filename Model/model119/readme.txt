Model119: three-beam LightOnOCR decoding
======================================
Inference-only experiment; no training. Model114's pinned fold0 step400 adapter,
same images and settings, except num_beams1->3. Keep repetition penalty1.05,
max_new_tokens64, length_penalty1, deterministic decoding, one returned string.
This tests sequence search, not text correction or grammar modernization.
Model118's neutral repetition penalty gained only0.000117 standalone and did
not pass its screen; it is not silently combined with this beam-search test.

Compare all819 original heldout fold0 images against Model114. Also replace
only Model92's voter in Model112's fixed family-word ensemble, keeping all
other voters and the original Model104 anchor unchanged. Report comparisons
to both Model112 and the matched Model114 replacement ensemble.
Advance screen: standalone gain>=0.001 vs Model114 AND ensemble gain>=0.0005
vs Model112. These reused folds provide development evidence, not unbiased CV.
Passing requires other-fold confirmation before submission work.

Run .venv-lighton/bin/python model119/experiment.py prepare, then
.venv-lighton/bin/python scripts/run_model119.py. Supervisor first decodes3
heldout images as a beam-compatibility smoke test, then decodes/scorers all819.
No adaptive setting changes after looking at smoke outputs, no extra training,
external data, cloud compute, test predictions or automatic submission.
Shared historical code and protocols remain untouched; new artifacts are
separate and configuration/source/checkpoint hashes are pinned before decoding.
Progress: outputs/model119/status.json; result: outputs/model119/report.json.
Safe GPU waiting and600-second background monitoring are retained.

Completed pilot (2026-09-11)
----------------------------
819 heldout predictions,180 changed lines,0 blanks. Three-beam score0.8953541410
vs matched greedy0.8917072302: +0.0036469108. This is meaningful standalone
development evidence for sequence search on this checkpoint.
Replacement ensemble0.9054975334 vs Model1120.9054038514: only+0.0000936820,
below the required ensemble gain and below Model114's greedy replacement.
The preregistered combined screen therefore FAILED. No submission prepared.
Do not reinterpret the standalone improvement as an ensemble/public gain.
Further multi-member decoding tests must be separately numbered and frozen.
