MODEL 73 — FULL-LINE LIGHTONOCR-2-1B BASE LORA
================================================

Purpose
-------
Model73 is the next high-VRAM architecture challenger after Model71 fold-0
validation. It fine-tunes the openly available Apache-2.0
LightOnOCR-2-1B-base checkpoint with LoRA on the provided training folds. It
is an OCR-specific vision-language model and is independent of the Kraken
PP-OCRv6 family, so useful errors may be complementary even if its standalone
score does not win.

Why this differs from Model2
----------------------------
Model2 was drafted before the image geometry experiments completed and uses a
0.84 center crop plus a long instruction. The established Kraken experiments
found the full supplied line crop stronger than the center view. Model73
therefore freezes full-image input and the model's native empty OCR prompt. It
also caps deterministic greedy generation at 64 new tokens; the audited labels
fit well below that limit, while the cap protects against the repetition loops
seen in zero-shot Model1.

Pinned base
-----------
Repository: lightonai/LightOnOCR-2-1B-base
Revision: c1bc8eb6625be7b34178ff53481b242f7b8ffcb8
License: Apache-2.0

The immutable revision must be downloaded before training and the resulting
snapshot is verified against the 15-file SHA-256 ledger in config.yaml before
either training or inference can load it. Both paths run with local-files-only
enabled. The challenge rules permit pretrained models
that are openly available to everyone; no external dataset, paid inference
API, test label, or manual transcription is used.

Fold-0 protocol
---------------
The first run trains only folds 1--4 and validates on canonical fold 0. It uses
bf16 LoRA rank 64, batch eight, 2-step accumulation (effective batch 16), five fixed epochs, seed
20260903, cosine learning rate 1e-4, and full line images. Adapter checkpoints
are selected by validation loss inside the training fold split. Exact
competition scoring of generated fold-0 text is required before any further
folds or test inference.

Commands
--------
  PYTHONPATH=src .venv-lighton/bin/python model73/train.py --fold 0

  PYTHONPATH=src .venv-lighton/bin/python model73/infer.py \
    --adapter outputs/model73/fold0/best_adapter \
    --input data/processed/train_folds.csv --fold 0 \
    --output outputs/model73/fold0_oof.csv

Status
------
The isolated environment and pinned checkpoint passed import, hash-ledger,
CUDA, and one-row real-image generation smoke tests. Fold-0 validation loss
was best at checkpoint step 400 (0.4674833); steps 500, 600, and 700 worsened,
so the overfitting tail was stopped and step 400 was frozen before confirmation
folds were decoded. Fixed checkpoint 400 scored 0.8906962580, 0.8954270287,
0.8887612944, 0.8835489034, and 0.8888979614 on folds 0--4. The exact global
score over all 4,095 canonical OOF rows is 0.8894664157724265, with no blanks.
All five complete checkpoint trees and signed inference artifacts were copied
from RunPod and independently hash-verified locally. Nothing was submitted
automatically.

The first isolated environment resolution exposed an impossible constraint in
the older shared VLM manifest: Transformers 5.16.1 requires safetensors 0.8 or
newer while that manifest capped it below 0.8. Model73 now owns a complete
pinned requirements file with safetensors constrained to [0.8, 0.9), leaving
the historical model environments unchanged. The environment also pins
torchvision 0.29.0, the official matching companion for torch 2.14.0, so the
Pod's system torchvision 0.23.0 cannot be imported by Transformers against the
newer isolated torch runtime.

The environment must be created without `--system-site-packages`. RunPod's
image carries torchaudio 2.8 system-wide, which is binary-incompatible with
the isolated Torch 2.14 runtime and is unused by this image-only model.

The first successful optimizer-step smoke used batch one with 16-step
accumulation and consumed only about 5.3 GiB of the 46 GiB A6000. Before any
meaningful training, batch four with 4-step accumulation was verified at about
10.9 GiB and 5.2 seconds per optimizer step. The final throughput setting is
batch eight with 2-step accumulation, preserving the effective batch of 16 and
all other settings while retaining ample A6000 memory headroom.
