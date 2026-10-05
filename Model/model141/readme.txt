Model141: GOT-OCR2 member - trained, measured, NOT deployed
============================================================
A fourth architecture family, chosen because adding a new family is the only
thing that has reliably paid in this competition (Kraken +0.0031 public, the
Qwen2.5-VL-7B +0.0017) while variants of existing members have not.

Base: stepfun-ai/GOT-OCR-2.0-hf, Apache-2.0, ungated, ~580M. An OCR-specialist
with a SAM-style vision encoder and a Qwen2-0.5B decoder - structurally unlike
LightOnOCR (Pixtral+Qwen3), the Qwen VLMs, and the PP-OCRv6 CTC model. Licence
checked through the HF API before any download; Qwen3-VL-8B and InternVL3-8B
(both apache-2.0) and Kosmos-2.5 (mit) were the alternatives, all far more
expensive per fold.

Recipe: rank-32 LoRA, lr 1e-4 cosine, effective batch 16 (2 x 8), 4 epochs,
bf16, real rows plus the round-4 pseudo-labels, greedy decoding, seed 20260918.
Its processor has no chat template, so model122/train.py gained a "gotocr2"
family branch: the processor emits the full prompt (fixed 1024x1024 image ->
exactly 286 tokens), the target is concatenated after it, and the prompt span is
masked out of the loss.

TWO REAL BUGS, both worth recording
-----------------------------------
1. Generation never stopped. GOT-OCR2's configured EOS is <|endoftext|>, but its
   chat format - and therefore every training target - ends with <|im_end|>. The
   model was never scored against a stop token it actually emits, so each line
   ran to max_new_tokens and the weighted CER blew up to 30.8. Fold 0 scored
   0.434. Passing the right eos_token_id at generation lifted it to 0.640 with
   no retraining change.
2. The vision encoder was frozen. GOT-OCR2's encoder uses qkv / proj / lin1 /
   lin2, none of which match the q_proj/k_proj/v_proj/o_proj/gate/up/down list
   every other member uses, so PEFT silently adapted only the text decoder -
   the part that least needs to learn 17th-century handwriting. Adding those
   four names took fold 0 from 0.640 to 0.813555, above Kraken's 0.809760.
   A LoRA target list is model-specific; matching names is not optional.

Fold-0 results (819 held-out rows, exact competition metric)
  GOT-OCR2 standalone, frozen encoder            0.640066
  GOT-OCR2 standalone, encoder adapted           0.813555
  six-member vote (submission #6 recipe)         0.915979
  seven members, adding GOT-OCR2                 0.916329   +0.000350
  seven members, min_votes 3                     0.914788
  GOT-OCR2 REPLACING Kraken                      0.913659   -0.002320
  drop the LoRA sibling, keep both OCR models    0.916241

DECISION: not deployed. The member is genuinely good (0.8136, better than
Kraken standalone) and it does help, but by +0.00035 on a single fold - inside
the noise band that a 3,000-resample bootstrap has repeatedly shown for
differences this size, and an order of magnitude below the 0.003 run-to-run
spread. Completing folds 1-4 would cost about 1.30 USD of the roughly 1.80 USD
left, for an expected public gain of perhaps 0.0002-0.0004. The same standard
rejected Model132 (-0.0006) and Model140 (+0.00033); applying it here too.

It is also informative that swapping GOT-OCR2 in for Kraken loses 0.0023: the
two OCR specialists are not interchangeable, and Kraken's CTC errors remain the
more valuable kind of different.

Everything needed to finish it is in place - config, the family branch, the
fold-0 model and predictions - so folds 1-4 are one command if the budget or
the judgement changes.

Cost: about 1.10 USD across three fold-0 attempts (two of them diagnosing the
bugs above).

UPDATE (2026-09-21): folds 1-4 WERE run after all (queue14, same host), and
the complete result reverses the fold-0 decision. Folds 1-4 scored 0.814321 /
0.812775 / 0.817287 / 0.814685. On the complete OOF with the submission #7
recipe (anchor model134_pseudo3, min_votes 2, fold-min-votes 3 for test):
  seven members (submission #7)     0.914150
  eight members (+GOT-OCR2)         0.914857   +0.000707
Twice the fold-0 estimate, and twice the +0.00033 that still moved the public
score for Model140. Candidate: Other/submissions/model141_ensemble8_candidate.csv.

Two reproduction notes for whoever builds the next ensemble:
 - Submission #7's anchor is model134_pseudo3, NOT model140_pseudo4. Member
   order matters; with model140 first the test file differs on 12 lines.
   (Both orders give ~0.9141 OOF, which is why this went unnoticed.)
 - The rebuild of #7 from the files in Other/outputs matches it byte for byte
   with the model134-first order.

Run:
  bash Model/model141/setup_host.sh
  python Model/model122/train.py --config Model/model141/config_gotocr2.json --fold N --predict-test
