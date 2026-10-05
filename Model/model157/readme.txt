Model157: where the deployed ensemble loses its score (error analysis, CPU only)
==============================================================================
Status: COMPLETE (2026-09-25), about 0.01 USD.

WHY
At rank 98 the deployed ensemble (public 0.919377) is 0.0154 behind first place
(0.934738). Every recent experiment gained at most ~0.001. Before spending on a new
direction, this measures whether the remaining loss is concentrated in some FIXABLE
class (multi-line crops read on the wrong line, truncation, dirty labels) or spread
across ordinary character-level misreadings. The competition discussion suggested
the former: multi-line crops whose label covers only the centre line (thread 34089),
and "very dirty" images (thread 34350).

METHOD
analyze_errors.py: the deployed twelve's complete out-of-fold predictions (4,095 rows,
each predicted by models that never saw it). Because the score is a ratio of summed
length-weighted edits, each row's share of the total loss is additive, so categories
can be ranked by loss share, with "max gain if perfect" as a hard upper bound.
split_tall.py: splits that loss by crop height x error severity (character edits /
target length). Outputs: analysis_output.txt, row_errors.json (per-row numbers).

RESULTS
Ensemble OOF 0.922499; 2,762 of 4,095 rows have at least one error.
Loss is diffuse: the worst 40 rows hold only 6.3% of it, the worst 10% hold 34%.

By crop height (median 65 px):
  tall (>= 1.8x median)  1,102 rows  40.8% of loss  1.52x the average per-row loss
  mid  (1.3 - 1.8x)        299 rows   8.2%          1.13x
  normal                 2,694 rows  51.0%          0.77x

By severity (character edits / target length), all heights:
  small  < 0.1           ~72% of loss   (1 or 2 misread letters, spacing, "^" forms)
  mid    0.1 - 0.4       ~26%
  GROSS  >= 0.4           1.6%          (wrong line or missing line)
Within tall crops: small 24.1%, mid 15.7%, gross only 1.0%.
Rows where the prediction is much longer or shorter than the label: almost none.

Other slices (overlapping): near misses of 1-2 edits 32% of loss; labels with the
superscript marker ^ 24%; labels with digits 7%; community-reported suspect IDs 2.4%.

CONCLUSION - no large fixable class exists
The wrong-line / missing-line failure the forum discussed is real (see the TOP 40 list:
e.g. JU7lRwk3jKkus24Z, VmrEALeZiP1Y6nF9) but rare: 1.6% of the loss, at most +0.0012
even if every such row were fixed perfectly. Tall crops are harder mainly because they
contain MORE ordinary misreadings, not because the model picks the wrong line.
The remaining loss is thousands of small character-level errors. No cleaning,
filtering or post-processing step can close a 0.015 gap; only a stronger reader can -
a larger or better-trained recogniser, and that is what the leaders must have.

This matches Model126 (text repair: nothing), Model148-155 (members and rule changes:
nothing) and turns the "dirty data" hint from the forum into a measured answer for
this ensemble.

Run: python Model/model157/analyze_errors.py && python Model/model157/split_tall.py
(needs the members' OOF files and Data/raw/images for crop sizes; Pillow)
