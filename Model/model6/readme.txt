MODEL 6 — Qwen2.5-VL-7B-Instruct QLoRA

Purpose
-------
A larger, mature VLM for diversity against Qwen3-VL and OCR-specialized models.
The 7B checkpoint is Apache-2.0 and pinned to
cc594898137f460bfe9f0759e9844b3ce807cfb5. This is distinct from the disallowed
Qwen2.5-VL-3B checkpoint, which uses the Qwen Research licence.

Hardware
--------
Use one 32–48 GB GPU for the supplied 4-bit QLoRA configuration. A 24 GB GPU may
work after reducing visual token count or enabling FlashAttention, but validate
any resolution change. For final reproducibility, record the CUDA/Torch build
and actual wall time from run_manifest.json.

Commands
--------
python model6/train.py --fold 0

python model6/infer.py \
  --adapter outputs/model6/fold0/best_adapter \
  --input data/processed/train_folds.csv \
  --fold 0 \
  --output outputs/model6/oof_fold0.csv

Follow the same five-fold and all-data procedure as model5. Test the six beam
candidates through model7's validation-tuned consensus. Do not use manual
test-specific corrections.

Possible second stage
---------------------
If SFT is competitive, model7 can add an automated edit-distance reward stage
or high-confidence pseudo-label retraining. Keep plain SFT as an independent
fallback because reinforcement tuning can overfit the small validation set.

Status
------
The audited CSVs and folds are ready. GPU execution is pending; this candidate
has not yet been trained or scored.
