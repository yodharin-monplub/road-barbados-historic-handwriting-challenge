Model135: architecture-aware ("family") word vote - no material gain
====================================================================
Model112 recorded that its gains came from agreement across architectures, and
Model124's plain rule counts every member equally, so four LightOnOCR-family
members can outvote the anchor between themselves while the single CTC member
is easily ignored. Model135 tags each member with its family (lighton, qwen,
ctc) and requires an alternative to be backed by at least `min_families`
distinct families before it replaces an anchor word.

Complete five-fold OOF, anchor model134_pseudo3:
  sources prod+lora+qwen+kraken, families>=1, votes>=2   0.910200  (= Model124's plain rule)
  sources prod+lora+qwen+kraken, families>=2, votes>=2   0.910236  (+0.000036)
  families>=2, votes>=3                                  0.907485
  + model129_pseudo2 as a fifth source, families>=2      0.909551
  + model129 and model125, families>=2                   0.908970
  only prod+qwen+kraken, families>=2                     0.908497

Conclusion: the family constraint changes almost nothing here (+0.00004, well
inside noise) because with only four sources a two-member agreement is nearly
always cross-family already - the rule only blocks the prod+lora pair, which
rarely agrees against the anchor on its own. It also does NOT make extra
same-family members safe to add: the five- and six-source variants are still
worse, so correlated members dilute the vote even when they cannot form a
majority by themselves.

Kept as a measured neutral result. The deployed submission keeps Model124's
simpler rule, since a 0.00004 difference does not justify a second code path
in the deployment.

Run: python model135/family_vote.py oof --min-families 2 --min-votes 2


Headroom check (oracle.py) - why voting cannot reach 0.94
---------------------------------------------------------
Per-row oracle over the five deployed members, complete OOF, 4,095 rows:
  best single member (model134_pseudo3)         0.902252
  deployed five-member vote                     0.910200
  ORACLE: pick the best member for every row    0.928476
  rows where at least one member is exactly right  1,396 of 4,095 (34.1%)
  oracle wins per member: pseudo3 2425, prod 614, lora 505, qwen 301, kraken 250

The oracle is a cheat - it reads the labels to choose - and it still lands at
0.9285, well below the 0.94 target. So no smarter combination rule over these
members can reach 0.94: the ceiling is set by the members themselves. The vote
has already captured about 44% of the distance from the best member to that
ceiling (0.902252 -> 0.910200 of a possible 0.902252 -> 0.928476).

Two consequences worth recording:
1. Further work on voting, repair or selection is close to exhausted; the
   measured gains (+0.00004 for family-awareness, ~0 for text repair, negative
   for character voting and for extra same-family members) agree.
2. Reaching 0.94 requires better recognizers, not better fusion. Two thirds of
   rows have no member that reads the line exactly right, which is a statement
   about the recognizers and the difficulty of the handwriting, not about the
   ensemble. Model136 prepares that next step (a larger VLM member) but needs
   GPU budget beyond what remains.
