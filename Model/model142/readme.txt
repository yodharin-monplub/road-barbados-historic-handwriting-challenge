Model142: quality-weighted word vote - no gain
==============================================
Model124 gives every member one vote, so Kraken (0.8059 out-of-fold) counts as
much as the Qwen2.5-VL-7B member (0.9024). That is deliberate - diversity has
outperformed strength throughout - but it deserved a measurement rather than an
assumption.

Two weighting schemes, both fixed before the result was seen:
  oof    weight proportional to (member OOF score - 0.80)
  logit  weight = log(s / (1 - s)) on the same score
An alternative replaces the anchor word when its weighted support exceeds the
anchor's own by at least a margin.

Complete five-fold OOF, all 4,095 rows, anchor model134_pseudo3:
  DEPLOYED Model124 rule (one vote each, min_votes 2)   0.913774
  logit weighting, margin 1.0                           0.913905   +0.00013
  logit weighting, margin 1.5 / 2.0                     0.913116 / 0.912661
  equal weights, margin 1.0 / 1.5 / 2.0                 0.913354 / 0.913140 / 0.913140
  oof weighting, margin 1.0 / 1.5 / 2.0                 0.912095 / 0.909003 / 0.907469

Conclusion: no. The best weighted variant is +0.00013, far inside the noise band,
and weighting by measured accuracy (the "oof" scheme) is actively worse - it
suppresses exactly the weak-but-decorrelated members that made the ensemble work
in the first place. One member, one vote stays.

This is the fifth fusion idea to return nothing (see also Model126 text repair,
Model127 character voting, Model135 family-aware voting, Model139's entry
comparison). Combining the current members is exhausted; only new members move
the score.

Run: python Model/model142/weighted_vote.py --scheme logit --margin 1.0
