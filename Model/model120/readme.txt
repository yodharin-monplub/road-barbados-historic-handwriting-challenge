Model120: paired beam decoding of original ensemble members
==========================================================
Model119 beam3 improved Model114 standalone by0.003647 but its single-voter
ensemble gain was only0.000094. This separately frozen experiment tests both
original Model73 and Model107 voters together, using their exact fold0 adapters.
Model92's fold0 adapter is not available locally; keep its verified greedy
prediction file unchanged. Never substitute an all-data model for a fold model.
Kraken predictions, original Model104 anchor and Model112 word-voting rule also
remain unchanged. No extra votes, learned selector or character-fusion search.

Run four819-image inference passes sequentially: Model73 greedy/beam3 and
Model107 greedy/beam3. Greedy replays on the current local runtime are paired
controls for old cloud/local numerical differences. Repetition penalty remains
1.05, max_new_tokens64, one returned string. No new training or external data.

Primary recipe changes both73 and107 to beam3 simultaneously. Compare to the
original Model112 and to the same two members re-decoded greedily locally.
Frozen screen: primary ensemble gain>=0.0005 vs BOTH baselines. Do not pick
the best subset after scoring. Report standalone beam-vs-greedy deltas and
old-vs-local greedy disagreements as diagnostics. Confirmation on other folds
is required before submission. Fold0 has prior development history.

Sources/adapter trees are verified against their original OOF manifests;
canonical fold0 coverage, input hash, prediction hash, base hashes, generation
settings and local adapter hashes are checked. New protocol pins runtime,
old source hashes, new code/configs and adapter trees before any inference.
All old artifacts and Model112 best public submission are preserved.

Commands with .venv-lighton/bin/python from workspace root:
  model120/experiment.py prepare
  model120/experiment.py check
  scripts/run_model120.py
Supervisor waits for GPU availability, runs the4 passes then scores, and
checks every600s without restarting other jobs. Local CUDA only; no cloud.
Progress: outputs/model120/status.json; result: outputs/model120/report.json.
No test inference or automatic submission. No score exists at creation.

Completed2026-09-11: original ensemble0.9054038514; local greedy0.9054511171;
both-beam ensemble0.9059250094. Gains+0.0005211579 and+0.0004738922 respectively.
The combined predefined screen FAILED (paired-control gain below0.0005).
Standalone73 local greedy0.8910611183 -> beam0.8956011280;
standalone107 local greedy0.8954071892 -> beam0.9011239789.
Greedy replay differed from original cloud on48 and49 lines, illustrating why
the paired control matters. All4 passes had819 rows and0 blank predictions.
Model121 separately documents a research continuation on folds1-4 motivated
by this borderline result; it does not retroactively mark Model120 as passed.
