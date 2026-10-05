Model137: competition-forum review and a data-driven label audit
================================================================
Prompted by the user on 2026-09-19: search the forum for clues, look for
external data, read the discussions.

1. EXTERNAL DATA IS FORBIDDEN
-----------------------------
Organizer ruling, discussion 34734 ("External data and pretrained models",
meganomaly/Zindi):
  "Competition data only may be used for training or fine-tuning. Additional
   external datasets may not be used."
  "Publicly available pretrained base models are allowed, provided their licence
   permits the challenge host to use, modify, reproduce and deploy the resulting
   solution." Research-only / non-commercial weights are not permitted.
Discussion 34612 asks specifically about MNIST, EMNIST and READ/ICFHR and has
no organizer reply, but 34734 already answers it: no.
=> No external handwriting corpus may be added. Our pipeline already complies;
   this closes the "find external data" idea rather than opening it.
Pseudo-labeling stays allowed (34459) provided it is strictly algorithmic with
no human-in-the-loop transcription, which is how Models 125/129/134 work.

2. WHAT OTHER COMPETITORS REPORT
--------------------------------
- discussion 34350: a participant reports "slightly above 0.91" using
  "qwen + lora + grpo and beam search", and suspects some training labels are
  unreliable. GRPO - reinforcement learning against a reward - is the one
  technique in this challenge that we have not tried; it optimizes the scoring
  metric directly instead of token cross-entropy. Our best public score is
  0.913145, so we are at or slightly ahead of that report.
- discussion 34861 lists five post-processing ideas that a participant measured
  as worthless (recasing from a lexicon, markup fixes, snapping rare words to
  the training vocabulary, over-transcription guards, per-batch resolution
  normalization). Every one matches what we measured independently: our text
  repair (Model126) gained ~0, character voting (Model127) lost 0.002, and
  upscaling (Model122) lost 0.029.

3. DATA QUALITY - AUDITED, MOSTLY A DEAD END
--------------------------------------------
label_audit.py scores every clean training row by the label-error signature:
the five members AGREE with each other but all disagree with the supplied
transcription. Over 4,095 rows, median member-to-member similarity is 0.942 and
median member-to-label similarity is 0.951, and exactly ONE row shows the
strong signature (JU7lRwk3jKkus24Z, consensus 0.896 versus label 0.341 - the
models read a completely different sentence, so image and transcript look
misaligned). It was on the community's 21-ID list and the earlier audit had
retained it. One row cannot matter.

expansion_analysis.py chased the one systematic pattern the audit surfaced:
some labels write an abbreviation AND its expansion ("Execut:^rs Executors",
"und^:r under", "S:^d Said"), which the recognizers never reproduce. Measured:
4 of 4,095 rows, accounting for 4 of the anchor's 7,009 word edits - 0.06% of
the word error. Negligible; not worth a rule.

Conclusion: the "unreliable data" the forum mentions is real but tiny. The
remaining error is genuine difficulty, consistent with Model135's oracle
ceiling of 0.9285 and with only 34% of lines having any member exactly right.

4. WHAT THIS LEAVES
-------------------
The only untried lever with real evidence behind it is GRPO-style training
against the competition metric (see model138). Everything else - external data,
label cleaning, more post-processing - is either forbidden, measured to be
worthless, or too small to matter.

Run:
  python model137/label_audit.py --top 20
  python model137/expansion_analysis.py
