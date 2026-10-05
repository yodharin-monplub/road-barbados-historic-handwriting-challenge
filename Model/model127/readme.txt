Model127: character-level vote inside aligned words - REJECTED
==============================================================
Model113's old recipe rebuilt: identical member alignment to Model124's word
vote, but when the members disagree on a word the decision is made per character
position inside that word instead of on the whole token. Word boundaries are
preserved - a word is never deleted, split or merged; only the tie-break
changes.

Complete five-fold OOF, four members, anchor model125_pseudo:
  word vote (Model124)        0.905544
  character vote (this)       0.903308   -0.002236

Rejected. Historically the character rule was marginally ahead of the word rule
(Model113 measured +0.000287 on the old, much weaker ensemble) and still lost on
the public leaderboard, 0.910877 against 0.911129. With today's stronger members
it is clearly behind: when good recognizers disagree about a word they usually
disagree about the whole word, so letting characters vote independently mixes
two readings into a spelling that neither model proposed.

Run:
  python model127/char_vote.py oof --members model125_pseudo model122_prod \
      model122_prod_lora model123_qwen3vl4b_lora --min-votes 2
