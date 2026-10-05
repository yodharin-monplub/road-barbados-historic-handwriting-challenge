Model136: Qwen2.5-VL-7B LoRA - PREPARED, NOT RUN (needs GPU budget)
===================================================================
Everything cheap has been tried. The measured pattern across five submissions
is that a genuinely different architecture is worth several times its CV gain
on the leaderboard (+0.0031 public for the Kraken member), while more variants
of an existing family are worth almost nothing, and the pseudo-label rounds
have flattened to +0.0008 on the ensemble. The remaining lever is a
substantially stronger recognizer.

Candidate: Qwen/Qwen2.5-VL-7B-Instruct, Apache-2.0, already vetted in
MODEL_LICENSES.md (revision cc594898137f460bfe9f0759e9844b3ce807cfb5). It is a
third VLM family - larger than the 4B Qwen3-VL member (which alone scores only
~0.838 yet earns its place in the vote) and unrelated to LightOnOCR's Pixtral
encoder.

Recipe (frozen): rank-32 LoRA on attention and MLP projections, lr 1e-4 cosine,
effective batch 16 (1 x 16), 3 epochs, bf16, native image scale with
min_pixels 256*28*28 and max_pixels 1280*28*28, greedy decoding, seed 20260918,
trained on real rows plus the round-3 pseudo-labels, exactly as Model134.
No height normalization - Model122 measured that upscaling costs -0.029.

Cost estimate: about 2 hours per fold on a 24GB GPU, so roughly 10-12 GPU-hours
for the five folds plus test inference. At 0.35-0.55 USD/h that is 4-7 USD,
which exceeds what remains of the authorized budget (account credit was
1.26 USD when this was written, shared with the user's other projects).

Gate before spending on all folds: train fold 0 only, then check the standalone
score and, more importantly, whether adding it to the Model134 five-member vote
improves the fold-0 ensemble. Model132 showed a member can be individually
reasonable and still make the vote worse, so the ensemble delta decides, not
the standalone number.

Run when funded:
  bash model122/vast_setup.sh          # adjust the torch index-url to the host's driver
  hf download Qwen/Qwen2.5-VL-7B-Instruct --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 \
      --local-dir checkpoints/model136/qwen2.5-vl-7b-instruct
  python model122/train.py --config model136/config_qwen25.json --fold 0 --predict-test

FOLD-0 GATE: PASSED (2026-09-20)
--------------------------------
Trained on a rented RTX 3090 (24GB), 3 epochs, rank-32 LoRA, real rows plus the
round-3 pseudo-labels, exactly the frozen recipe above. 873 optimizer steps,
about 4 hours, 18.5GiB peak.

  Qwen2.5-VL-7B standalone, fold 0            0.905972
    for comparison: model134_pseudo3 0.905961, Qwen3-VL-4B 0.840700,
    Kraken 0.809760, model122_prod 0.899821
  five-member vote (the submission #5 recipe)  0.912252
  six-member vote with the 7B added            0.915979   +0.003727
  same six with the 7B as anchor               0.915792

It is as strong as the best LightOnOCR member while belonging to a third
architecture family, which is exactly the combination that has paid off in this
competition: the Kraken CTC member was worth +0.0031 public at 0.81 standalone,
and this is stronger still.

Note on the oracle ceiling reported in model135 (0.9285): that was a property of
the five members it was computed over, not a law. Adding a strong decorrelated
member raises the ceiling itself, which is why this gate matters more than any
further tuning of the existing vote.

Folds 1-4 are training to produce the complete out-of-fold estimate and the
per-fold test vectors; only then will the six-member rule be validated on all
4,095 rows and deployed. Fold 0 has development history and cannot carry a
submission on its own.

COMPLETE FIVE-FOLD RESULT (2026-09-20)
--------------------------------------
member alone, all 4,095 rows: 0.902391 - the best single model in the project
  folds: 0.905972 / 0.906241 / 0.899182 / 0.904064 / 0.896492
  (for comparison model134_pseudo3, the previous best, scores 0.902252)

Ensembles on the complete out-of-fold set, anchor model134_pseudo3, min_votes 2:
  five members, no 7B (submission #5)          0.910200
  SIX members with the 7B        (SELECTED)    0.913774   +0.003574
  six members, 7B as the anchor instead        0.913067
  six members, min_votes 3                     0.911151
  five members, dropping the LoRA sibling      0.914327
  seven members (adding the round-2 pseudo)    0.913231

The drop-the-LoRA-sibling variant measured highest, but a 3,000-resample paired
row bootstrap put that gain at +0.000553 with a 95% interval of
[-0.000338, +0.001432] and only 89% of resamples positive - not significant. All
six members were kept, since a larger committee should be the safer choice on
the private split when the measured difference is noise.

Deployed as Other/submissions/model136_ensemble6_qwen7b.csv
  SHA-256 00ba19df641bb66589c94dcf352298ba131f80853e6925e36f812b747f6b6954
  1374 rows, 93 lines changed versus the anchor own fold consensus.

Cost: about 3.50 USD for roughly 24 GPU-hours on a 0.148 USD/h RTX 3090.

Status: complete and deployed as submission #6.
