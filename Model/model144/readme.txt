Model144: Qwen2.5-VL-7B member with vision attention adapted
============================================================
Status: fold 0 only (2026-09-22), folds 1-4 NOT run - out of GPU credit.
  fold 0 standalone 0.904500 (Model136: 0.905972) - no standalone gain; its
  vision MLPs were already adapted, so only attention was new here.
  fold-0 ensemble: 9 members (with Model143) 0.920203 -> 10 members 0.920818
  (+0.0006); replacing Model136 instead: 0.920722. Marginal on one fold -
  the same size GOT-OCR2 showed on fold 0 before completing at +0.0007.
  Completing it: about 4 h per fold, ~2.6 USD for folds 1-4 plus test.
  Fold-0 test predictions were not written (host stopped to save credit).

Model136's LoRA targets reached the vision MLPs (gate/up/down_proj share
names with the decoder) but not the vision attention (attn.qkv, attn.proj).
This config adds "qkv" and "proj". The merger (mlp.0 / mlp.2) stays frozen:
its layer names are bare indices that a name list cannot target safely.

Otherwise identical to Model136, except the pseudo-labels move from round 3
to round 4 (Model140's labels), matching the current ensemble's anchor.
Note that this makes the fold-0 comparison with Model136 confounded by the
label change; Model140 measured that change alone at about +0.0004.

Gate: fold 0 standalone vs Model136 fold 0 (0.905972), then the fold-0 vote
with it replacing Model136.

Estimated cost: fold 0 about 1-1.2 USD; all five folds plus test 3.5-6 USD.

Run:
  python Model/model122/train.py --config Model/model144/config_vision.json --fold 0 --predict-test
