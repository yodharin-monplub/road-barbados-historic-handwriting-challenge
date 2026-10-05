Model156: FINAL SOLUTION - reproducibility package
==================================================
This is the entry point for anyone reproducing the final submission, including a
Zindi code review. It adds no model; it pins down exactly how the best submission
is made and how every part of it can be rebuilt.

FINAL SUBMISSION
  file:   Other/submissions/model152_ensemble12_pruned.csv
  sha256: 7cdaada34149b920f58755f707fff2bb736740cbd780022c5d504c94389a7389
  public leaderboard: 0.919377358
  complete 5-fold out-of-fold CV: 0.922499 (4,095 rows, exact competition metric)

REBUILD IT (no GPU, standard library only, about a minute)
  python Model/model156/rebuild_final.py
It runs the ensemble on the twelve members' saved predictions in Other/outputs,
writes Other/submissions/model156_final_rebuilt.csv, recomputes the CV, and prints
REPRODUCED only if both the file's sha256 and the CV match the values above.
VERIFIED 2026-09-25 on a fresh Linux host (Python 3.10.13, standard library only,
nothing else installed): CV 0.922499 and sha256 7cdaada3... byte-identical, in 28 s.

HOW THE ENSEMBLE WORKS (Model/model124: vote.py, ensemble.py, build_submission.py)
For each test line, start from the ANCHOR member's transcription (model134_pseudo3)
and replace a word only when at least 2 other members agree on the same alternative
(min_votes 2). Each member's five fold models are first merged into one test
prediction by a 3-of-5 word majority (fold-min-votes 3). The same rule, applied to
the held-out folds, gives the CV; so the deployed rule is exactly the validated one.
  - member set: chosen by backward elimination, validated on a held-out fold
    (Model152, +0.0011 public)
  - anchor: re-tested against all twelve alternatives (Model155) and kept

THE TWELVE MEMBERS
Every member is trained per fold (5 folds, Data/processed/train_folds.csv), so each
training row is predicted only by models that never saw it.
Training entry points: Model/model122/train.py (LightOnOCR and Qwen members,
Model122's shared trainer), Model/model22 + Model/model130 (Kraken, the original
member), Model/model149/train.py + infer.py (the Kraken pseudo-label variants).

 member (Other/outputs/...)         recipe (Model/...)          base model                        CV alone
 model134_pseudo3        (anchor)   model134/config_pseudo3.json LightOnOCR-2-1B-base              0.902252
 model140_pseudo4                   model140/config_pseudo4.json LightOnOCR-2-1B-base              0.902625
 model122_prod_lora                 model122/config_prod_lora.json LightOnOCR-2-1B-base            0.892456
 model130_kraken                    model22/config.yaml via model130/run_folds.sh  PP-OCRv6 medium 0.805866
 model136_qwen25vl7b                model136/config_qwen25.json  Qwen2.5-VL-7B-Instruct            0.902391
 model143_qwen3vl4b_vision          model143/config_vision.json  Qwen3-VL-4B-Instruct              0.893923
 model144_qwen25vl7b_vision         model144/config_vision.json  Qwen2.5-VL-7B-Instruct            0.902563
 model145_qwen3vl4b_vision_pseudo4  model145/config.json         Qwen3-VL-4B-Instruct              0.898561
 model146_qwen3vl8b_vision          model146/config.json         Qwen3-VL-8B-Instruct              0.899437
 model147_qwen3vl8b_pseudo5         model147/config.json         Qwen3-VL-8B-Instruct              0.904972
 model149_kraken_pseudo5            model149/config.yaml         PP-OCRv6 medium                   0.804717
 model150_kraken_pseudo4            model150/config.yaml         PP-OCRv6 medium                   0.806886
Each Model/<dir>/readme.txt holds that member's full recipe, run commands and traps.

LICENCES (Other/MODEL_LICENSES.md has pinned revisions)
  LightOnOCR-2-1B-base ............ Apache-2.0
  Qwen2.5-VL-7B-Instruct .......... Apache-2.0
  Qwen3-VL-4B / 8B-Instruct ....... Apache-2.0
  PP-OCRv6 medium historical HTR .. Apache-2.0, explicitly approved by Zindi (2026-08-28)
No member uses a non-commercial or research-only licence. (Zindi ruled CHURRO-3B
ineligible on exactly that ground; nothing like it is in the final ensemble.)
External data: none. Pseudo-labels come only from the competition's own test images,
labelled by this project's own fold models (organizer-approved, discussion 34459).

PSEUDO-LABELS AND RETRAINING ORDER
Five members train on pseudo-labelled test lines. Fold k's labels come only from
fold-k models, none of which saw fold k, so the CV stays honest. The label files are
saved (Other/outputs/pseudo3, pseudo4, pseudo5; each has a manifest.json naming the
voting members per fold), so retraining one member needs only its round's files.
Rebuilding the labels themselves from nothing follows the chain:
  round 1 Model125 -> round 2 Model129 (pseudo2) -> round 3 Model134 (pseudo3)
  -> round 4 Model140 (pseudo4) -> round 5 Model147 (pseudo5)
  pseudo3 voters: model129_pseudo2 model122_prod model122_prod_lora
                  model123_qwen3vl4b_lora model130_kraken
  pseudo4 voters: model134_pseudo3 model122_prod model122_prod_lora
                  model123_qwen3vl4b_lora model130_kraken model136_qwen25vl7b
  pseudo5 voters: the eleven-member ensemble of Model146/score.txt; model144 is absent
                  from fold 0 only, because its fold-0 test predictions were never written
  command (each round): python Model/model129/prepare_pseudo2.py --members ... \
                        --min-votes 2 --output-dir Other/outputs/pseudoN

HARDWARE
Members were trained on rented single GPUs (RTX 3090 / 4090 / 5090, 24-32 GB) on
vast.ai. Host setup: Model/model146/setup_host.sh (Qwen/LightOnOCR) and
Model/model130/kraken_setup.sh with the fixes in Model/model149/readme.txt (Kraken).
Library versions are pinned in those scripts (torch 2.9 cu128, transformers 5.16.1,
peft 0.20.0; kraken 7.1).

WHAT WAS TRIED AND NOT KEPT (so a reviewer need not repeat it)
Model148 GOT-OCR2 on round-5 labels, Model151 a fourth Kraken variant, Model153
LightOnOCR on round-5 labels, Model154 TrOCR-large-handwritten, Model155 a different
anchor: each was flat or negative once checked on a held-out fold. Their score.txt
files give the numbers.
