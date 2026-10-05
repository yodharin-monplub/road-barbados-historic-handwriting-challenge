Model149: the Kraken/PP-OCRv6 CTC member trained WITH round-5 pseudo-labels
===========================================================================
Model130 is the only member that had never seen pseudo-labels, and the most
decorrelated one, so it was the last untried combination.

Kraken reads one labelled CSV, so the pseudo rows are added per fold:
Data/processed/train_folds_pseudo5_fold{fold}.csv = the 4,095 real rows plus the
1,373 round-5 pseudo rows, the pseudo rows tagged with a fold OTHER than the
validation fold. Fold k's held-out 819 rows are untouched, so the score stays
comparable with Model130's. config.yaml carries {fold} in train_csv/prepared_dir/
output_dir and train.py renders it; everything else is the Model22 recipe
(fixed zero-based epoch 7, seed 20260903).

FIVE PRACTICAL TRAPS, all worth recording
-----------------------------------------
1. Kraken configs and Model22/130's scripts predate the Model/ + Other/ split.
   road_htr.kraken_cli resolves relative paths against Other/, so the host needs
   Other/data -> ../Data and Other/model149 -> ../Model/model149 symlinks, and
   command-line paths (--checkpoint/--input/--output) must be ABSOLUTE.
2. Model22/infer.py adds <parent>/src to sys.path, which no longer exists.
   Model149/infer.py resolves the root by AGENTS.md instead. Without it every
   fold's inference dies AFTER training, and the runner still reports success.
3. kraken_setup.sh downloads the checkpoint under checkpoints/model8/registry but
   looks for it under Other/checkpoints/...; the copy is done here, with the
   published sha256 verified.
4. This Kraken build has no kraken.__version__, so the setup's banner line aborts
   the script under set -e.
5. Kraken installs torch 2.14+cu130: a host with an older driver (CUDA 12.8)
   fails at model load. Rent with cuda_max_good >= 13.0.
Also: the CTC preflight rejected one pseudo row (267xSV848jfudVh1, transformed
width 3923 > max 3584); it is in exclude_ids.txt. The preflight is worth having -
Kraken would have silently replaced that line.

RESULT - deployed, for a reason that is not the obvious one
-----------------------------------------------------------
complete 5-fold OOF: 0.804717   (Model130 without pseudo-labels: 0.805866, -0.001149)
The pseudo-labels did NOT make this member better.

ENSEMBLE (complete OOF, anchor model134_pseudo3, min_votes 2)
  twelve members (previous candidate)     0.921278
  THIRTEEN (+model149)                    0.921716  (+0.000438)  <- NEW BEST CV
  thirteen, min_votes 3                   0.921464
  twelve, model149 REPLACING model130     0.921401  (+0.000123)
bootstrap thirteen vs twelve: +0.000438, 95% [+0.000015, +0.000839], 97.9% positive

A member that is no stronger, and here slightly weaker, still helps because it is
wrong in different places: same architecture, different training data. Compare
Model148, where a GOT-OCR2 twin of equal strength HURT the ensemble (-0.00075).
The two results together sharpen the rule: what an ensemble gains from a new
member is decorrelation, not quality, and retraining the most decorrelated
architecture on different data buys more of it than retraining a correlated one.

candidate: Other/submissions/model149_ensemble13_candidate.csv
  21 of 1374 test lines differ from the twelve-member candidate
  sha256 1af9ef97eb335a4b156cf76b3ad84366dc81dd4c63533335048c3a8c7988babb
public: not yet submitted

cost: about 2.8 USD for five folds (about 13 min/epoch, epoch 7 reached in ~1.8 h).
Run: bash Model/model130/kraken_setup.sh, then Model/model149/train.py --fold N
     and Model/model149/infer.py (see the traps above).
