Model143: Qwen3-VL-4B member with the vision encoder actually adapted
=====================================================================
Status: COMPLETE (2026-09-22). All five folds, complete OOF 0.893923 standalone
(Model123: ~0.838). Nine-member ensemble 0.917124 complete OOF, +0.002267 over
the eight-member ensemble, bootstrap 100% positive. See score.txt.
Folds ran in parallel on four rented RTX 3090s, about 11 s/step (6 s on a
faster host), roughly 2.5 h per fold including test prediction.

Why
---
Model141 found that a LoRA target list copied between architectures silently
misses encoders whose layers use other names. Checked against the published
weight indexes on Hugging Face (2026-09-21):

  Qwen3-VL-4B vision blocks:  attn.qkv, attn.proj, mlp.linear_fc1, mlp.linear_fc2
  Qwen3-VL-4B merger:         linear_fc1, linear_fc2
  Qwen2.5-VL-7B vision:       attn.qkv, attn.proj, mlp.gate_proj/up_proj/down_proj

Model123 used q/k/v/o_proj + gate/up/down_proj, so its ENTIRE vision encoder
and merger were frozen - only the text decoder learned. (Model136's Qwen2.5-VL
got its vision MLPs adapted by name coincidence but not its attention.) That
is a plausible cause of the gap between Qwen3-VL-4B (fold scores 0.834-0.844)
and Qwen2.5-VL-7B (0.90). For GOT-OCR2 the same fix was worth +0.17 on fold 0.

Recipe: identical to Model123 except the LoRA targets add qkv, proj,
linear_fc1, linear_fc2 (PEFT matches whole module names, so "proj" hits only
the vision attention output). No pseudo-labels, so the fold-0 gate isolates
the fix. train.py now prints "trainable vision tensors: N" - it must be > 0.

Gate: fold 0 standalone vs Model123 fold 0; then fold-0 vote with it replacing
Model123 in the seven-member ensemble. Proceed to folds 1-4 only if the
ensemble gains.

TWO ENVIRONMENT TRAPS hit on the first launch (2026-09-21)
---------------------------------------------------------
1. A bare "proj" target also matches visual.patch_embed.proj - a Conv3d -
   so the list form put LoRA on the patch embedding (210 vision tensors
   instead of 208). The config now uses a regex anchored on attn.proj.
2. Far worse: with torch 2.9 + cuDNN, the bf16 Conv3d patch embedding is
   pathologically slow. Even the ORIGINAL Model123 recipe could not reach
   step 5 in five minutes on this host (faulthandler stack: conv.py
   _conv_forward inside modeling_qwen3_vl.py). The patch embedding's kernel
   equals its input patch, so train.py's new "conv3d_as_linear" flag runs it
   as the identical matmul (max abs diff 8.8e-4 in fp32/TF32 against the real
   Conv3d). Result: 11 s/step with the vision encoder trained.
   Any Qwen-VL member on this environment needs "conv3d_as_linear": true.
Also: code.tgz must include AGENTS.md and Other/src - train.py walks up to
AGENTS.md to find the project root and spins forever at "/" without it.

Estimated cost: fold 0 about 0.4-0.6 USD on an RTX 3090; all five folds plus
test about 2-3 USD. A follow-up with the same fix for Model136 (Qwen2.5-VL-7B,
adding qkv and proj) would cost about 3.5-6 USD.

Run:
  python Model/model122/train.py --config Model/model143/config_vision.json --fold 0 --predict-test
