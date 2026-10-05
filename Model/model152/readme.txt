Model152: systematic member selection (no training, CPU only, about 0.10 USD)
=============================================================================
Every member set so far was chosen by hand: add the new member, try one or two
obvious drops, keep what scored best on the complete OOF. That is selection on
the same data the score is read from. This run does it systematically AND checks
whether the procedure survives on data it never saw.

Method (Model/model124/member_select.py)
  PART 1 backward elimination on the complete OOF: repeatedly drop the member
    whose removal helps most, until nothing helps. Optimistic by construction.
  PART 2 the same elimination using folds 0-3 ONLY, then score the resulting
    subset on fold 4, which took no part in that selection. This is the honest
    estimate of what pruning is worth.

PART 1 (all folds)   15 -> 11 members, 0.921492 -> 0.922412
  dropped in order: model151_kraken_pseudo3, model122_prod_lora,
                    model141_gotocr2, model150_kraken_pseudo4
PART 2 (folds 0-3)   15 -> 12 members, 0.922080 -> 0.922983
  dropped in order: model151_kraken_pseudo3, model122_prod, model141_gotocr2
  HELD-OUT fold 4:  deployed 14 = 0.919794
                    part-2 subset = 0.920565   (+0.000771)
                    all 15        = 0.919143
The pruning survives on a fold that never took part in choosing it, so this is a
real effect and not the usual selection noise.

Candidates on the complete OOF, bootstrapped against the deployed fourteen:
  A deployed 14                 0.922014   -
  B 14 minus GOT-OCR2 (13)      0.922098   +0.000085, 65.8% positive
  C part-1 winner (11)          0.922412   +0.000399, 89.6% positive
  D part-2 winner (12)          0.922499   +0.000485, 95.3% positive   <- DEPLOYED

D is deployed because it is the only candidate chosen without using fold 4 and
then confirmed on fold 4; C scores similarly but was selected on the very data it
is scored against. Relative to the deployed fourteen, D drops model122_prod and
model141_gotocr2.

Both selection runs independently dropped model141_gotocr2. GOT-OCR2 earned its
place when the ensemble had eight members (+0.0007); with Qwen3-VL-8B, the
pseudo-label variants and three Kraken variants present, its errors are now
covered by others. Ensemble membership is not monotone - a member that helped
once can become redundant later, and nothing re-tests that unless you ask.

candidate: Other/submissions/model152_ensemble12_pruned.csv
  55 of 1374 test lines differ from the deployed fourteen
  sha256 7cdaada34149b920f58755f707fff2bb736740cbd780022c5d504c94389a7389
public: 0.919377358  <- NEW BEST (fourteen-member 0.918264019, +0.001113)
A 0.10 USD CPU-only run bought more public score than the last three GPU rounds
(about 9 USD) combined. Re-testing old decisions is cheaper than adding members.

Members (12): model134_pseudo3 (anchor), model140_pseudo4, model122_prod_lora,
model130_kraken, model136_qwen25vl7b, model143_qwen3vl4b_vision,
model144_qwen25vl7b_vision, model145_qwen3vl4b_vision_pseudo4,
model146_qwen3vl8b_vision, model147_qwen3vl8b_pseudo5, model149_kraken_pseudo5,
model150_kraken_pseudo4.

Run: python Model/model124/member_select.py   (needs the members' OOF files only)
NOTE: keep that file out of any directory that is on sys.path when another script
runs - a module named select.py shadows the standard library's and breaks numpy.
