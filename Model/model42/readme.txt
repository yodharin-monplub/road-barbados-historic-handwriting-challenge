MODEL 42 — FROZEN MODEL35-CONFUSIONS VS MODEL25 NESTED SELECTOR AUDIT
=====================================================================

Purpose
-------
Model35's fold-0 Model13/confusions output is the current strongest available
same-fold OCR postprocess (0.846941606). Model25 is weaker by itself on fold 0
(0.817450092), but may repair a small, complementary subset. Model42 tests
that claim conservatively, with Model35 as candidate A and Model25 as
candidate B.

Protocol
--------
The Model35 confusions CSV is accepted only when its provenance-verified
report binds it as the held-out fold-0 output. The Model25 OOF CSV and the
Model33 feature implementation are SHA-pinned. Model42 does not use Model33
as a candidate: its fold-0 cross-fitted strings are transductive with respect
to a second selector assessment and would make the partition lineage harder to
interpret. It also deliberately does not consume Model22 confidence values.

For each of two predeclared ID-hash salts and each of five inner partitions:

* the held partition is absent from selector labels and metric importance;
* features are target-independent pair geometry only: disagreement, length,
  prefix/suffix, and exact/casefold equality; no target language model or
  confidence threshold is used;
* every probability is frozen before its held partition is scored.

The predeclared primary is a deliberately small 9-feature pair-geometry
logistic selector: only candidate disagreement/length/prefix/suffix/equality.
There is no post-score model or threshold
search. The primary can be called conditionally promising only if every salt improves
by at least 0.001 and all twenty held partitions improve on frozen Model35.
This is still fold-0 conditional evidence, not a deployable test policy or a
complete five-fold OOF result.

Run
---
  PYTHONPATH=src python3 model42/evaluate.py \
    --output-dir outputs/model42/fold0_rerun

The evaluator is CPU-only, uses no OCR model, network, or GPU, refuses a
nonempty output directory, and writes only below outputs/model42.

Tests
-----
  PYTHONPATH=src python3 -m unittest model42.test_model42 -v

Status
------
The declared two-salt panel was run on CPU. Both salts selected zero Model25
rows, exactly reproducing frozen Model35/confusions at 0.846941606. The
predeclared 0.001 robust-gain rule therefore rejects the selector; Model42
does not create a submission or a new terminal policy.

Authoritative report and artifacts:

  outputs/model42/fold0_authoritative_rejection_v2/report.json
  outputs/model42/fold0_authoritative_rejection_v2/runs.csv
  outputs/model42/fold0_authoritative_rejection_v2/selections.csv
