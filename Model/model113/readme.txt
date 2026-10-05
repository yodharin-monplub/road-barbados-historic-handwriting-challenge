Model113: family-aware character ensemble (Model112's fixed secondary recipe)
============================================================================

Replay the already-evaluated diagnostic_family_char recipe without retuning.
Use the SAME Model104/105 anchor as in validation, not Model112's word output.
This is an alternative to Model112, not an unvalidated word-then-character chain.

Sources: Model73/77, Model92/105, Model107, and corrected Kraken. At each aligned
character position, require Kraken plus at least two LightOn models to agree.
Preserve case, punctuation, historical spellings, whitespace and word boundaries
unless the fixed character vote authorizes an internal change. Empty whole-word
results retain the original word. A missing/blank source makes the entire test
line keep the original anchor; it is not counted as an additional fallback vote.

Previously measured development OOF:
- Original Model104 ensemble: 0.9014774424
- Model112 family word:        0.9036181592
- This family character rule: 0.9039049159
The character rule improves over the original anchor on all five folds, but is
only +0.0002867567 above the word rule overall and may not improve the leaderboard.
The reused upstream ensemble is not fully nested; this is development evidence.
Do not describe it as an independent untouched validation result.

Run: PYTHONPATH=src .venv-lighton/bin/python model113/prepare_submission.py
No training, GPU, cloud, external data, or automated Zindi upload.
Preserve Model112, whose user-reported public score is 0.911129378.
Outputs: outputs/model113/submission_report.json, test_changes.csv and the
separate submissions/model113_family_character_ensemble.csv.

Completed preparation
--------------------
Saved OOF score reproduced: 0.903904915886025 (+0.000286756682316 versus
Model112 word voting). Eight fusion regression tests passed. All four source
CSVs match the previously verified Model112 inputs; all LightOn inference
receipts still bind the same all-data models and official Test.csv.

Submission validated: 1374 rows, 103 changed lines versus the original anchor,
65 changed lines versus Model112. One blank-source line retains the original
anchor. Serialized predictions were checked for exact equality to fused text.
SHA-256 b689cb1f2ac54233db3569870fdea77f9013ff74eaa19312284b0e0963c1954e
User submitted as S28NrUEn and reported public score 0.910877117.
This is 0.000252261 below Model112 (0.911129378), which remains the best.
The small character-fusion development gain did not transfer to public scoring.
