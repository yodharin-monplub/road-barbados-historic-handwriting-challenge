Model125: pseudo-label self-training on the test images
=======================================================
Zindi's organizers explicitly permit automated pseudo-labeling for this
challenge (discussions/34459, already cited by Model7). This adapts the
recognizer to the test distribution without any manual labeling.

Method: for each fold k, take the 1,374 test-image transcripts produced by the
Model122 fold-k model, treat them as labels, and train a fresh model from the
same pinned base on 3,276 real rows + those pseudo rows (4,650 total, 4 epochs,
recipe otherwise identical to model122/config_prod.json). Fold k is then scored
exactly as before.

Why this stays honest: fold k's pseudo-labels come from the fold-k model, which
never saw fold k. No held-out label enters training, directly or through a
label-derived transcript, so the fold-k score remains a fair held-out estimate.
The pseudo-labels are wrong some of the time - that is the point of the test,
not a defect - and nothing is hand-corrected.

Gate, fixed before the run: fold 0 must beat the matched Model122 fold-0 score
of 0.899821 greedy by at least +0.002 to justify spending the remaining GPU
budget on the other four folds. Fold 0 has prior development history in this
project, so a pass is permission to confirm, not proof of generalization.

Run:
  python model125/prepare_pseudo.py
  python model122/train.py --config model125/config_pseudo.json --fold 0

Fold-0 gate result (2026-09-18): PASSED
---------------------------------------
  Model122 baseline, fold 0, greedy : 0.899821  (wWER 1.7697, wCER 2.9084)
  Model125 pseudo-label, fold 0     : 0.902607  (wWER 1.7138, wCER 2.8584)
  gain +0.002786, against the pre-registered +0.002 requirement
Training used 4,649 rows: 3,276 real plus 1,373 pseudo-labeled test lines (one
test image produced no text from this model and was dropped rather than
labeled with an empty string).

Both weighted WER and weighted CER improved, so the gain is not a single-metric
artifact. All five folds are now running with --predict-test so the member can
join the Model124 ensemble; the complete out-of-fold number is what decides
whether it replaces Model122 as the primary member.

Caveat kept in view: the pseudo-labels come from a model whose own fold-0 score
is 0.8998, so roughly one word in fifty-five is being taught wrong. The gain
says the adaptation outweighs that noise on this fold, not that the labels are
correct.

Status: five-fold run in progress.
