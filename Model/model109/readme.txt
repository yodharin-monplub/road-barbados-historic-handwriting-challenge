Model109: competition-adapted DiffusionPen handwriting generator
===============================================================

User requested this separate experiment on 2026-09-07. Model108 remains the
unchanged pretrained/style-reference baseline; Model109 fine-tunes the generator
using only automatic word crops and exact labels from eligible training lines.
Rules confirmation is the user's explicit statement recorded in Model108.

Initialize from the same pinned and hash-verified Model108 pretrained assets.
Train rank-8 attention LoRA adapters and the style-conditioning projection for
200 optimizer steps, microbatch 1, accumulation 4, learning rate 1e-4. Freeze the
text encoder, VAE, style extractor, and other generator weights. These small
adapters are a conservative local-GPU pilot, not full generator training from
scratch. Every original target word has five other same-line style crops.

Exclude fold 0 completely from all generator source data. Additionally exclude
every Model108 pilot target/style source line from generator fitting, so the
same 200-word pilot tests previously unseen source lines for this generator.
No external dataset, test image, manual annotation or modernized spelling.

Commands:
  .venv-lighton/bin/python model109/train.py --prepare-only
  .venv-lighton/bin/python model109/train.py
  .venv-lighton/bin/python model108/generate.py \
    --adapter outputs/model109/fold0/adapter.pt --output outputs/model109/fold0/pilot

Compare exact-word OCR acceptance with Model108 on identical requests/seeds.
Two distinct fold-excluded OCR checkers must agree with requested text exactly.
Generator loss, visual similarity and pilot OCR accuracy are NOT competition CV.
Only a subsequent real/synthetic OCR training ablation can justify a submission.
No Model109 competition CV or submission exists yet.

Preparation result: 18,685 eligible word crops from 2,113 source lines.
A one-step CUDA smoke test completed with finite loss and peak allocated CUDA
memory 1,110,093,312 bytes. The real 200-step run and subsequent paired quality
audits run under road-model109-pilot.service on the local RTX 4050. No extra
cloud rental is required. Keep this PC awake. Progress: outputs/model109/status.json
and outputs/model109/pilot.log. Training loss is not an OCR accuracy score.

Completed pilot result (2026-09-07)
----------------------------------
The 200-step generator fine-tune completed, and 200 matched-seed pilot images
were generated using source lines excluded from generator fitting.
Kraken matched 0/200 requested words exactly (147 blank readings); LightOnOCR
matched 7/200. Joint acceptance was 0/200, compared with Model108's 20/200.
This configuration degraded text fidelity despite lower denoising loss.
It is rejected for synthetic OCR training; no OCR CV or submission is claimed.

The failure is an experimental result, not an interrupted generator run.
Both originally interrupted OCR checks have now completed after recovering the
missing base checkpoint and installing PEFT locally. Training/generation was
not repeated. Adapter and failed images are retained for reproducibility.
Comparison: outputs/model109/comparison.json and
outputs/model109/fold0/pilot/word_comparison.csv.
