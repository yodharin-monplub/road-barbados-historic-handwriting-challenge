Model123: Qwen3-VL-4B-Instruct LoRA challenger (architecture diversity)
======================================================================
Purpose: a second, genuinely different recognizer family to ensemble with the
rebuilt LightOnOCR models. Model112 showed the old gains came from voting
across architectures (Kraken + LightOn), not from more LightOn variants; that
whole ensemble is gone with the old machine, and this project currently has
only one family again.

Base: Qwen/Qwen3-VL-4B-Instruct, Apache-2.0, already vetted in
MODEL_LICENSES.md (revision ebb281ec70b05090aa6165b016eac8ec08e71b17).
Qwen2-VL attempts (Models 27/39/43/50) all failed or were stopped on the 6GB
laptop; this runs on a rented 32GB GPU instead, at the model's own dynamic
resolution rather than the degraded low-resolution settings those used.

Recipe (frozen before any score): shared model122/train.py with family
"qwen3vl", rank-32 LoRA on attention and MLP projections, lr 1e-4 cosine,
effective batch 16 (2 x 8), 3 epochs, bf16, native image scale with
min_pixels 256*28*28 and max_pixels 1280*28*28, greedy decoding, seed 20260918.
Model122's finding that upscaling line crops hurts (-0.026 on fold 0) is
carried over: no height normalization.

Gate: fold 0 standalone score, compared against the Model122 full fine-tune
(0.899716). It does NOT need to win outright - a slightly weaker model with
uncorrelated errors is still valuable in the word vote - but it must reach at
least about 0.88 to be worth ensembling, and the ensemble gain itself must be
measured on out-of-fold predictions before any submission.

Fold 0 has prior selection history in this project; this is development
evidence, not unbiased CV.

Run (on the rented host):
  python model122/train.py --config model123/config_lora.json --fold 0

Status: not yet run; no score.
