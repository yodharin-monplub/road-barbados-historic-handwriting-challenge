Model124: anchored word-vote ensemble over four rebuilt members
===============================================================
Replaces the lost Model112 ensemble, whose inputs no longer exist. Same idea as
Model112 - vote across architectures, not across seeds of one model - but every
member is retrained and the voting rule is validated on complete out-of-fold
predictions rather than assumed.

Members (each trained per fold, so each row is predicted only by models that
never saw it):
  model125_pseudo          pseudo-label self-trained LightOnOCR   OOF 0.899227
  model122_prod            LightOnOCR full fine-tune              OOF 0.896608
  model122_prod_lora       LightOnOCR rank-64 LoRA                OOF 0.892456
  model123_qwen3vl4b_lora  Qwen3-VL-4B LoRA (other architecture)  OOF ~0.838

Rule: align every member to the anchor (model125_pseudo) with a word-level
Levenshtein alignment; replace an anchor word only when at least 2 of the other
3 members agree on the same alternative and fewer agree with the anchor.
Insertions and deletions are voted the same way. vote.py holds the alignment;
its behavior on substitution, insertion and deletion cases is unit-checked.

Complete out-of-fold results over all 4,095 clean rows
------------------------------------------------------
  anchor alone (model125_pseudo)                  0.899227
  4 members, min_votes 2  (SELECTED)              0.905544
  4 members, min_votes 3                          0.901841
  3 members without the LoRA member               0.903053
  3 members without the full fine-tune            0.903271
  3 members without Qwen (measured earlier)       0.902839
Four rules were compared on the same complete OOF and the best was taken, so
0.905544 carries a little selection optimism; the spread between rules is small
(0.0037) and every rule beats the best single member. For reference the old
four-source ensemble scored 0.9054 OOF and 0.911129 public.

The weakest member matters. Dropping Qwen (OOF 0.838, the only non-LightOnOCR
family) costs more than dropping either strong LightOnOCR sibling, which is the
same lesson Model112 recorded: decorrelated errors beat individually stronger
but correlated ones. On fold 0 an even weaker upscaled model (0.8685) was also
a net-positive fourth voter.

Deployment: each member first collapses its five fold models into one test
vector by strict majority (3 of 4), then the members vote with the rule above,
so the deployed rule is the validated one. Empty lines fall back to the anchor
and then to the first member that read the image.

  submissions/model124_ensemble4_maj.csv
  1374 rows, validated against SampleSubmission.csv, no empty predictions
  238 lines changed versus the anchor's own fold consensus
  SHA-256 b90a54cace4fb945bf209b54c0ba374db50bc51abcc3253f3631e5c737f092a1

Run:
  python model124/ensemble.py oof  --members model125_pseudo model122_prod \
      model122_prod_lora model123_qwen3vl4b_lora --min-votes 2
  python model124/ensemble.py test --members ... --min-votes 2 --fold-min-votes 3
  python model124/build_submission.py --anchor outputs/model124/test_ensemble.csv \
      --output submissions/model124_ensemble4_maj.csv

Status: submitted for scoring; public result pending.
