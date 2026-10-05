Model138: GRPO against the competition metric - READY, NEEDS GPU BUDGET
=======================================================================
Why this and nothing else: Model135's oracle shows the ceiling of combining the
current members is 0.9285, and we are at 0.9102 CV / 0.9131 public, so fusion is
nearly exhausted. Model137's forum review found that external data is forbidden,
that the label-quality rumour is worth one row, and that the post-processing
ideas other participants tried are the same ones we measured as worthless. The
single untried technique with evidence behind it is GRPO: a participant
reporting "slightly above 0.91" uses "qwen + lora + grpo and beam search"
(discussion 34350).

The idea: supervised training maximizes token likelihood, but the challenge
scores weighted word and character edits, where a word edit costs about 4.6
character edits. GRPO optimizes that objective directly. For each training line
the policy samples a group of transcriptions, each is scored with the EXACT
per-row competition cost (row_cost mirrors the metric: word edits/12 plus
character edits/55, each weighted by the square root of the reference length),
the within-group spread becomes advantages, and probability moves toward the
better samples. Groups where every sample scores identically carry no signal
and are skipped.

Protocol (frozen before any run):
- Start from the Model134 supervised checkpoint for the same fold, retrained
  once with save_model true (config_sft_save.json) because the earlier runs did
  not keep their weights.
- Train a fresh rank-16 LoRA on the attention projections only, lr 2e-6,
  group size 6, temperature 0.9, top-p 0.95, effective batch 8 groups,
  300 steps, no augmentation (the reward must reflect the real image).
- Evaluate the held-out fold every 100 steps with the exact metric, and record
  the supervised starting score first so the comparison is like-for-like.
- Gate: fold 0 must improve on its own supervised starting point by at least
  +0.002 before any other fold is trained. Fold 0 has development history, so a
  pass is permission to confirm, not proof.
- The log-probability mask covers generated tokens up to and including the first
  EOS, so padding after a finished sample never enters the objective.

Cost estimate: the supervised re-run is about 50 minutes per fold, and GRPO
sampling six continuations per line makes each optimizer step roughly six
generations plus one forward/backward - expect 1.5-2 hours for 300 steps on a
24GB GPU. Fold-0 gate about 2-3 USD; all five folds plus test inference another
6-8 USD. Account credit was 1.26 USD when this was written, so the user has to
decide before it runs.

Run:
  python model122/train.py --config model138/config_sft_save.json --fold 0
  python model138/grpo.py --config model138/config_grpo.json --fold 0

RESULT (2026-09-19): GATE FAILED - GRPO does not help at this scale
-------------------------------------------------------------------
Ran on a rented RTX 3090, user-authorized 4 USD budget; actual spend ~1 USD.

  supervised start (same code path, fold 0)   0.901960
  GRPO step 25                                0.902076   +0.000117
  GRPO step 50                                0.901427   -0.000533
  GRPO step 75                                0.900793   -0.001167
Required +0.002; the curve peaks within noise at step 25 and then declines
monotonically, so the run was stopped at step 75 rather than spending the rest
of the budget confirming a trend that was already clear.

Why it does not work here, measured rather than guessed:
- 83% of sampled groups carry NO learning signal: all six samples score
  identically, so the within-group advantage is zero and the group is skipped.
  At temperature 0.9 it was 93%; raising to 1.15 only moved it to 83%. A
  supervised OCR model on a line image is nearly deterministic - it is not the
  open-ended generation setting GRPO was designed for. The gradient therefore
  comes from a small, unrepresentative slice of the data.
- With 8 signal-bearing groups per optimizer step, 75 steps saw roughly 600
  useful lines. The forum participant reporting GRPO success presumably ran
  orders of magnitude longer; that is a different budget, not a different idea.

Two real bugs were found and fixed on the way; both silently produced a loss of
exactly zero, which is worth recording:
1. Samples generated under torch.inference_mode() cannot re-enter autograd.
   Generation now runs under torch.no_grad().
2. The EOS mask counted the PROMPT's own <|im_end|> tokens - the chat template
   contains two - so the cumulative "after EOS" test zeroed every generated
   position. The mask now starts searching at the first generated token. Before
   the fix the gradient norm was exactly 0.000e+00 and the adapter never moved,
   while the script appeared to train normally for hours.

Side observation with wider consequences: re-running the IDENTICAL supervised
config and seed on different hardware (torch 2.9/cu128 on a 3090 versus
torch 2.14/cu130 earlier) scored 0.902701 against 0.905961 - a 0.0033 spread.
That is larger than several recent "improvements", so single-fold differences
below about 0.003 should not be trusted without repeats.

Status: implemented, debugged, measured, rejected. Reward function unit-checked
(identical text 0.0, one substituted word 0.2459, one missing word 0.4837).
Nothing from this experiment enters a submission.
