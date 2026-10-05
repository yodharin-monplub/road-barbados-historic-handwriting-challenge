# Reproducing the current best submission

## Layout (AGENTS.md, 2026-09-20)

```
AGENTS.md              the only file in the root
Data/  raw/            competition images and CSVs (a junction to the working
                       disk, so the 442MB image set exists exactly once)
       processed/      train_folds.csv (4,095 clean rows) and test_ids_derived.csv
Model/ model1 .. model139   one folder per experiment: source, readme.txt, score.txt
Other/ src/ scripts/ config/ tests/   shared code, helpers, pinned settings
       outputs/ checkpoints/          junctions to the working disk (weights and predictions)
       submissions/ .env email.txt    submission CSVs and credentials
```
Every script finds the project root by walking up to `AGENTS.md`, so paths keep
working wherever the tree is moved. Run commands from the root; the rented GPU
hosts unpack the same three folders.

Everything below uses only the supplied competition data and publicly available
Apache-2.0 base models, in line with the organizer's ruling in
[discussion 34734](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34734):
competition data only for training, and pretrained weights only where the
licence permits the host's downstream commercial use.

Best public score: **0.913145** (`Other/submissions/model130_ensemble5_kraken.csv`,
complete five-fold OOF 0.909376).
Latest candidate: `Other/submissions/model134_ensemble5_pseudo3.csv`, OOF 0.910200.

## 0. Environment

Two environments are needed because Kraken pins an older Python.

```bash
# recogniser / ensemble environment
uv venv Other/.venv --python 3.11
uv pip install --python Other/.venv/bin/python torch torchvision \
    --index-url https://download.pytorch.org/whl/cu130   # cu128 for driver 570.x
uv pip install --python Other/.venv/bin/python transformers==5.16.1 accelerate==1.14.0 \
    peft==0.20.0 safetensors==0.8.0 sentencepiece Pillow==12.3.0 PyYAML==6.0.3 numpy

# Kraken environment for the CTC member
uv venv Other/.venv-kraken --python 3.12
uv pip install --python Other/.venv-kraken/bin/python "kraken==7.1.*"
```
A C compiler must be present (`apt-get install build-essential`); torch's
inductor needs it at runtime.

## 1. Data and base weights

```bash
# Data/raw/{Train.csv,Test.csv,SampleSubmission.csv,images/} from Zindi
python Other/scripts/prepare_folds.py --train-csv Data/raw/Train.csv \
    --exclude-ids Other/config/known_bad_train_ids.txt \
    --output Data/processed/train_folds.csv      # 4,095 rows, 819 per fold

hf download lightonai/LightOnOCR-2-1B-base \
    --revision c1bc8eb6625be7b34178ff53481b242f7b8ffcb8 \
    --local-dir Other/checkpoints/model73/lightonocr-2-1b-base
hf download Qwen/Qwen3-VL-4B-Instruct \
    --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
    --local-dir Other/checkpoints/model123/qwen3-vl-4b-instruct
XDG_DATA_HOME=$PWD/Other/checkpoints/model8/registry Other/.venv-kraken/bin/kraken get 10.5281/zenodo.21788410
# verify: sha256 of medium.safetensors must be
# 15313b51ace64cbfa81f8f6ef25ad64f04e5a6fb7f7823e67b107527bc081ac9
```

## 2. The five members (each trained per fold, five folds each)

| member | what | command |
|---|---|---|
| model122_prod | LightOnOCR full fine-tune, native resolution, 4 epochs | `python Model/model122/train.py --config Model/model122/config_prod.json --fold N --predict-test` |
| model122_prod_lora | same base, rank-64 LoRA, 2 epochs | `python Model/model122/train.py --config Model/model122/config_prod_lora.json --fold N --predict-test` |
| model123_qwen3vl4b_lora | Qwen3-VL-4B rank-32 LoRA, 3 epochs | `python Model/model122/train.py --config Model/model123/config_lora.json --fold N --predict-test` |
| model130_kraken | PP-OCRv6 CTC, fixed epoch 7 | `bash Model/model130/run_folds.sh 0 1 2 3 4` (Kraken env) |
| model134_pseudo3 | full fine-tune + round-3 pseudo-labels | see step 3 |

Fold `N` runs 0..4. Every run writes `oof_ep*_greedy.csv` (that fold's held-out
predictions) and `test_beam1.csv` (all 1,374 test rows) under `Other/outputs/<name>/foldN/`.

## 3. Pseudo-label rounds

Each round relabels the test images with the best ensemble so far and retrains
the main member on real + pseudo rows. Fold k's labels always come from a vote
among only the fold-k models, which never saw fold k, so held-out scores stay
honest. Permitted by [discussion 34459](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34459)
because the process is strictly algorithmic.

```bash
python Model/model125/prepare_pseudo.py                              # round 1 labels
python Model/model122/train.py --config Model/model125/config_pseudo.json --fold N --predict-test

python Model/model129/prepare_pseudo2.py --members model125_pseudo model122_prod \
    model122_prod_lora model123_qwen3vl4b_lora --output-dir Other/outputs/pseudo2
python Model/model122/train.py --config Model/model129/config_pseudo2.json --fold N --predict-test

python Model/model129/prepare_pseudo2.py --members model129_pseudo2 model122_prod \
    model122_prod_lora model123_qwen3vl4b_lora model130_kraken --output-dir Other/outputs/pseudo3
python Model/model122/train.py --config Model/model134/config_pseudo3.json --fold N --predict-test
```

## 4. Ensemble and submission

```bash
# validate the rule on the complete out-of-fold set (all 4,095 rows)
python Model/model124/ensemble.py oof --members model134_pseudo3 model122_prod \
    model122_prod_lora model123_qwen3vl4b_lora model130_kraken --min-votes 2

# apply the same rule to the test set and write a validated submission
python Model/model124/ensemble.py test --members model134_pseudo3 model122_prod \
    model122_prod_lora model123_qwen3vl4b_lora model130_kraken \
    --min-votes 2 --fold-min-votes 3
python Model/model124/build_submission.py \
    --anchor Other/outputs/model124/test_ensemble.csv \
    --output Other/submissions/model134_ensemble5_pseudo3.csv
```

Each member first collapses its five fold models into one test vector by strict
majority (3 of 4 sources), then the members vote with the rule validated above.
`build_submission.py` checks the ID set and column order against
`SampleSubmission.csv` and refuses to write an empty prediction.

## 5. Checks

```bash
python Other/scripts/score_csv.py --truth Data/processed/train_folds.csv \
    --predictions Other/outputs/model124/oof_ensemble.csv
python Other/scripts/validate_submission.py --submission Other/submissions/<file>.csv \
    --sample Data/raw/SampleSubmission.csv
python -m unittest discover -s Other/tests -v
```

## What is deliberately NOT in the pipeline

- No external datasets of any kind (forbidden; see Model/model137/readme.txt).
- No manual label edits or test-specific fixes; the three excluded training rows
  are documented in `Other/config/suspect_row_audit.txt` and the CSVs are unchanged.
- No image metadata (size, aspect ratio, pixel count) is used to choose between
  predictions, which Zindi's computer-vision rules prohibit.
- No text-repair chain: refitted on these predictions it gains nothing
  (Model/model126), and character-level voting is worse than word voting (Model/model127).
