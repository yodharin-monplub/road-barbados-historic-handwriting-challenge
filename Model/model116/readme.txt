Model116: shuffled short-word handwriting augmentation
====================================================
User idea: join random word crops into new 3- or 4-word images with known labels.
This differs from Model107, which reconstructed existing long transcripts.
No language model, external text, handwriting generator, or test images are used.

Pilot: 819 synthetic images +3276 original training lines, with819 original
fold0 lines held out. Crop donors and the crop-bank aligner exclude fold0.
Reuse the Model107 bank's24840 confidence-filtered crops, not uncertain cuts.
Verify donor source images, exact transcription spans, crop hashes and aligner
lineage. Automated confidence filtering cannot guarantee perfect word cuts.
Each synthetic line samples3 or4 crop instances uniformly without replacement,
requires at least2 source images, and joins their exact labels with single spaces.
Sampling seed20260916; preserve case, historical spelling and punctuation.
Normalize crop height to the donor median; use paper-colored gaps and reject
aspect ratios above36. No grammar correction or sentence reconstruction.
819 is a controlled pilot, not a claim to have used every possible combination.

Matched control: completed Model114, seed20260914, same base, rank64 LoRA,
bf16, full images, mild augmentation, batch1/accumulation16,400 optimizer steps,
20 warmup steps,0.0001 cosine learning rate, local Torch2.13 runtime.
Start from the pinned base, NOT the trained Model114 adapter. Evaluate/save at
step400 only. The intervention is adding the shuffled short-line examples.
80/20 is the dataset row ratio, not guaranteed per batch or per target token.
At a fixed step budget, synthetic rows replace some real-row training exposures.

Frozen development screen: standalone gain>=0.001 vs matched Model114 AND
family-word replacement ensemble gain>=0.0005 vs Model112. Replace only Model92's
voter, keep Model73/107/Kraken and the original Model104 anchor unchanged.
Also report gain vs Model114's same replacement ensemble. No extra LightOn votes.
These folds have prior selection history; results are development evidence,
not untouched CV. Full-fold confirmation is needed before any submission.

Code is isolated to Model116. A process-local validator hook enables the new
manifest format only in its training entrypoint; shared historical source files
and existing frozen protocols remain unchanged. New provenance is frozen before
training. No overwrite of old model outputs and no cloud rental.

Run with .venv-lighton/bin/python from the workspace root:
  model116/experiment.py prepare
  model116/test_model116.py
  scripts/run_model116.py
The supervisor runs train -> heldout inference -> scoring, checks every600s,
and waits safely for GPU availability. It does not launch further folds/test
predictions or upload a submission automatically.
Progress: outputs/model116/status.json; results: outputs/model116/report.json.

Pilot preparation completed
---------------------------
819 synthetic lines:386 three-word and433 four-word images, using2727 distinct
word crops from the24840-crop pool. Example exact labels include "and or sale"
and "ninth Leiveten^t persons houses"; no attempt is made to correct grammar.
The full dataset integration check passed (4095 training rows), all9 Model116
regression tests passed, and all4 historical seed-protocol tests passed.
Completed400 steps and819 heldout predictions on2026-09-10. Standalone score
0.8891412186 versus matched Model1140.8917072302 (gain-0.0025660116).
Family-word replacement ensemble0.9057154795 versus Model1120.9054038514
(gain+0.0003116280), also slightly below Model114's replacement0.9057409291.
The predefined screen failed. No submission prepared; cloud was not used.
