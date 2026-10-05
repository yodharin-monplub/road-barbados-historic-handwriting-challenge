Model139: choosing the two entries for private judging
=======================================================
Zindi scores two selected submissions on the private leaderboard, and after
Model138 measured a run-to-run spread of about 0.003, differences of a few
ten-thousandths can no longer be taken at face value. This runs a paired row
bootstrap between the two live candidates.

compare_entries.py precomputes each row's weighted WER and CER terms once and
then resamples rows, because the metric is a ratio of sums - the same answer as
re-scoring every resample, but seconds instead of hours. The reconstruction is
asserted against src/road_htr/metrics.py on the real data before bootstrapping,
which caught two details worth knowing: the scorer strips chat tokens from BOTH
sides first, and an empty reference still carries unit weight.

Result, complete out-of-fold, 4,095 rows, 3,000 resamples, seed 20260919:

  submission #4  (model129_pseudo2 anchor + prod + lora + qwen + kraken)  0.909376
  submission #5  (model134_pseudo3 anchor, same four voters)              0.910200
  observed gain  +0.000824
  95% interval   [-0.000057, +0.001651]
  positive in    96.5% of resamples
  rows differing 794 of 4,095

Reading: #5 is probably the better model, but the interval touches zero, so the
two are close to equivalent on cross-validation. The public leaderboard is the
sensible tiebreaker - #4 already scored 0.913145.

Recommendation for the two private entries:
  1. Submission #4 - the highest public score actually observed.
  2. Submission #5 - the highest cross-validation score.
They share four of five members, so this is a hedge between two measurements of
the same family rather than true diversification. If a genuinely different
second entry is wanted, submission #2 (four members, no Kraken, public 0.909996)
is the most decorrelated option on record, at a measured cost of about 0.003
public score.

Run:
  python model139/compare_entries.py --a <oof_a.csv> --b <oof_b.csv> --resamples 3000
