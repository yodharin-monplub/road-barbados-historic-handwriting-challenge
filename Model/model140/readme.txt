Model140: fourth pseudo-label round (labels from the six-member ensemble)
=========================================================================
Each round relabels the 1,374 test images with the best ensemble available and
retrains the main LightOnOCR member on 3,276 real rows plus those pseudo rows.
Fold k's labels come only from the fold-k models, none of which saw fold k, so
the held-out score stays honest. Organizer-approved (discussion 34459).

Round 4's labels come from the six-member vote that includes Qwen2.5-VL-7B
(complete OOF 0.913774), a clearly better labeler than round 3's five-member
vote (0.910200).

Member progression (complete five-fold OOF, all 4,095 rows)
  no pseudo-labels (model122)                0.896608
  round 1, single-model labels (model125)    0.899227   +0.002619
  round 2, 4-member vote labels (model129)   0.900386   +0.001159
  round 3, 5-member vote labels (model134)   0.902252   +0.001866
  round 4, 6-member vote labels (this)       0.902625   +0.000373

Expectation set before the run: a small gain. The round-4 labels differ from the
round-3 anchor's own predictions on only 53-78 lines per fold, against 126-158
for round 3, so the labeler and the member are converging and this well is
running dry.

PREPROCESSING NOTE - a real bug caught here
-------------------------------------------
The first attempt died with CUDA out of memory at 23.2GiB on a 24GB card. The
cause was the upload archive, not the recipe: to save upload time on a slow
home connection the images had been re-exported capped at 1540px on the LONGEST
edge only, dropping the 176px HEIGHT cap that every earlier member was trained
with. Tall crops (aspect ratio below about 8.75, roughly the bottom few percent)
then arrived far larger, producing many more vision patches, and Pixtral's dense
block attention mask grows with the square of the patch count.

Two lessons: the height cap was doing real work, and any change to the image
pipeline silently changes every member's inputs. The original archive
(height <= 176, longest edge <= 2800, JPEG q92) was restored, after which peak
memory returned to 20.1GiB, matching the earlier runs exactly. Preprocessing
must stay identical across members or their predictions are not comparable.

Run:
  python Model/model129/prepare_pseudo2.py --members model134_pseudo3 model122_prod \
      model122_prod_lora model123_qwen3vl4b_lora model130_kraken model136_qwen25vl7b \
      --output-dir Other/outputs/pseudo4
  python Model/model122/train.py --config Model/model140/config_pseudo4.json --fold N --predict-test

RESULT: the pseudo-label ladder is exhausted
--------------------------------------------
Member, complete OOF: 0.902625, only +0.000373 over round 3.
Per fold versus round 3: -0.000184 / +0.000908 / +0.000778 / -0.001870 / +0.004020
 - four folds move by less than 0.002 in either direction and one gains 0.004;
   this is the shape of noise, not of a real improvement.

Ensembles on the complete out-of-fold set (min_votes 2):
  six members, round-3 anchor (SUBMISSION #6, public 0.915407)   0.913774
  six members, round-4 anchor                                    0.913969
  seven members, both pseudo anchors                             0.914102
A 3,000-resample paired bootstrap of the seven-member candidate against the
submitted six-member ensemble gives +0.000328 with a 95% interval of
[-0.000524, +0.001183] and only 78% of resamples positive. Not significant.

DECISION: no new submission. Submission #6 stays the deployed entry. Spending a
slot to chase +0.0003 that the bootstrap cannot distinguish from zero would be
chasing noise, and the measured run-to-run spread for an identical recipe is
about 0.003 - ten times this difference.

The ladder in full: +0.002619, +0.001159, +0.001866, +0.000373. Each round
relabels with a better ensemble, but the labeler and the member have converged -
round 4 changed only 53-78 lines per fold, against 126-158 for round 3. Further
rounds are not worth GPU time; a new architecture is (the Qwen2.5-VL-7B member
was worth +0.0036 CV and +0.0017 public on its own).

Cost: about 1.50 USD for five folds on a 0.27 USD/h RTX 3090.

UPDATE (2026-09-21): the seven-member candidate WAS uploaded by the user as an
optional extra submission and scored 0.915619140 public - a new best, +0.000212
over submission #6. So a CV gain the bootstrap could not separate from zero
(+0.000328) still moved the leaderboard in the same direction. Lesson recorded:
"not significant" means "cannot be confirmed", not "will not help"; for cheap,
low-risk changes a real submission is a legitimate test.

Status: deployed as submission #7 (public 0.915619), the current best entry.
