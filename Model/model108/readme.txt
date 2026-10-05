Model108: sequential local handwriting-generator pilot (rules gate)
==================================================================

Run only after Model107's evaluation. The selected candidate is DiffusionPen,
a handwriting-specific diffusion generator, not a text-only chat LLM.
Project: https://github.com/koninik/DiffusionPen
Weights: https://huggingface.co/konnik/DiffusionPen
The project and weight card state MIT; its Stable Diffusion VAE and other
dependencies require separate license review. Pin immutable revisions and hashes.

STATUS: user explicitly confirmed checking the rules and instructed proceeding
on 2026-09-07. This is recorded as USER confirmation, not an independently
retrieved organizer statement. Nothing was posted to Zindi automatically.

When permitted, use text only from the current training partition and style
references from five automatically aligned word crops of a single eligible
training line. Never use validation/test images or labels for generation.
Generate a 200-word pilot first. Unsupported characters/oversized words are
rejected without normalization. Independent fold-excluded Kraken and LightOnOCR
checks must both reproduce the requested text exactly; this is conservative
filtering, not proof of ground-truth correctness. Never relabel generated images
manually or replace requested text with a model's guessed transcript.

Download only allowlisted model weights and required code. The upstream HF repo
also contains saved_iam_data and generated IAM images: these are external data
and MUST NOT be downloaded or used. Do not run the upstream default loader,
which expects IAM data. A competition-only loader is required.

Do not combine with Model107 for its initial ablation. If the pilot is viable,
use 80% real / 20% generated rows with the same 400-step LightOnOCR recipe.
Use Model107's development and full-OOF promotion thresholds. A generator that
does not pass text-fidelity checks should be rejected, not scaled up.

The pretrained generator passed a two-word CUDA smoke test. Its complete
200-word baseline is queued in scripts/run_model109_pilot.py, paired with the
separate fine-tuned Model109 on identical requests and random seeds. Both use
the same frozen KrakEN/LightOn OCR checks. Model108 itself is not fine-tuned.
There is no competition validation score or submission for Model108 yet.

Completed pilot result (2026-09-07)
----------------------------------
All 200 images were generated and checked with both fold-excluded recognizers.
Kraken matched 25/200 requested words exactly; LightOnOCR matched 84/200.
Only 20/200 (10%) passed BOTH checks. These are word-fidelity measurements,
not competition CV. No guessed labels were substituted for requested text.
The missing local OCR checkpoint was copied back from the authorized Vast
instance and all 15 files passed pinned SHA-256 verification. PEFT 0.20.0 was
installed locally to load the adapter. No new cloud instance was rented.

Decision: do not scale this current generator into synthetic OCR training.
Most requested words did not survive the conservative fidelity filter, and
there is no evidence of an OCR improvement. No Model108 submission is prepared.
Detailed comparison: outputs/model108/fold0/pilot/word_comparison.csv.
Quality report: outputs/model108/fold0/pilot/quality_report.json.

Exact asset allowlist and hashes: checkpoints/model108/asset_manifest.json.
No stock upstream data loader is imported. Generator checkpoints are loaded
with weights_only=True. The complete pilot records requested text, source/style
IDs, image hashes and generator hashes; outputs are never manually relabelled.
