Model132: second Kraken variant at 128px line height - REJECTED
==============================================================
Hypothesis: if one CTC member was worth +0.0026, a second CTC member at a
different input scale (128px lines instead of 96px, max_width raised to 4864 so
wide lines still fit) would add more decorrelated votes.

Fold-0 result:
  Kraken 96px standalone                     0.809760
  Kraken 128px standalone                    0.801171
  5-member vote (one Kraken)                 0.911511
  6-member vote (both Krakens)               0.910925   (-0.000586)

Rejected after fold 0; folds 1-4 were not trained. Two members of the SAME
architecture trained by the SAME recipe make correlated errors, and with
min_votes 2 they can outvote the anchor together, so the second one dilutes the
vote instead of strengthening it. Diversity means a different architecture, not
a different hyper-parameter of the same one - the Model112 "family" rule made
exactly this distinction by requiring agreement across families.

Cost: about 0.25 USD for the fold-0 gate, stopped immediately afterwards.
