Model155: choosing the ensemble anchor (CPU only, about 0.03 USD)
=================================================================
Status: COMPLETE (2026-09-24). NEGATIVE - the current anchor stays.

QUESTION
vote_all (Model/model124/vote.py) starts from the ANCHOR member's prediction and only
replaces a line when at least min_votes other members agree on the same alternative.
So the anchor is a real hyper-parameter, and Model141 showed that swapping it changes
test lines. It had never been chosen on purpose: model134_pseudo3 has been the anchor
since Model134 only because it was listed first.

METHOD (Model/model124/anchor_select.py)
Members: the deployed Model152 pruned twelve.
  PART 1 every member as anchor, complete OOF (optimistic - chosen on the same data).
  PART 2 rank anchors on folds 0-3, then score the winner on held-out fold 4, which
         took no part in the ranking. Same honest check as Model152 and Model154.
Anchors are scored in parallel processes; 24 cores took about two minutes.

PART 1 - complete OOF
  model140_pseudo4                     0.922629
  model134_pseudo3                     0.922499  <- current
  model122_prod_lora                   0.922478
  model147_qwen3vl8b_pseudo5           0.922355
  model146_qwen3vl8b_vision            0.922239
  model143_qwen3vl4b_vision            0.922089
  model144_qwen25vl7b_vision           0.921866
  model145_qwen3vl4b_vision_pseudo4    0.921791
  model136_qwen25vl7b                  0.921700
  model130_kraken                      0.921006
  model150_kraken_pseudo4              0.920797
  model149_kraken_pseudo5              0.920760

PART 2 - honest check
  best anchor on folds 0-3: model140_pseudo4 (0.923357)
  held-out fold 4, current anchor (model134_pseudo3): 0.920565
  held-out fold 4, model140_pseudo4:                  0.919724   (-0.000841)

CONCLUSION
The +0.00013 that model140 shows on the complete OOF is selection noise: chosen
without fold 4, it loses 0.0008 on fold 4. The current anchor stays. The spread
across all twelve anchors is narrow (0.9208 to 0.9226), and the ordering makes sense:
the strong LightOnOCR/Qwen members make good anchors, the three Kraken CTC variants
(0.80 standalone) the worst - an anchor is overruled only by agreement, so a weak
anchor leaves more errors standing.

Taken with Model152 (member pruning, +0.0011 public) this also says where the
combination rule's slack was: in WHICH members vote, not in who starts the vote.

Run: python Model/model124/anchor_select.py   (needs only the members' OOF files)
