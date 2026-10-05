Model154: TrOCR-large-handwritten, fine-tuned per fold
======================================================
Status: COMPLETE (2026-09-24). Negative/inconclusive - not deployed; see score.txt.

WHY
---
Every ensemble gain of the recent run came from decorrelation, and every variant of
an existing architecture is now exhausted (Model151, Model153 and Model148 were all
flat or negative). TrOCR is a genuinely new family for this ensemble: a ViT image
encoder with a RoBERTa-style text decoder, pre-trained specifically on handwritten
text lines - structurally unlike the causal VLMs (LightOnOCR, Qwen), the OCR VLM
(GOT-OCR2) and the CTC model (Kraken). Model3 implemented it early in the project,
but it was never trained.

LICENCE
-------
Zindi excludes models without commercial-use rights (it ruled CHURRO-3B out on
that basis). The Hugging Face card for microsoft/trocr-large-handwritten carries no
licence tag, but Microsoft's trocr/README.md in github.com/microsoft/unilm says the
project is covered by the repository's root LICENSE, which is MIT. Recorded in
Other/MODEL_LICENSES.md; verify in the final audit.
Revision: e68501f437cd2587ae5d68ee457964cac824ddee.

RECIPE (config.yaml)
--------------------
Model3's: images letterboxed to 256x1024 with interpolated ViT position
embeddings (the crops have median aspect ratio 16.8 - squashing them to 384x384
destroys letter shapes), centre band 0.84, AdamW lr 2e-5 cosine, 5% warmup,
bf16, gradient checkpointing, effective batch 8 (4 x 2), 10 epochs, no
pseudo-labels, greedy decoding.

FOUR THINGS FIXED BEFORE IT WOULD TRAIN CORRECTLY
-------------------------------------------------
1. CHECKPOINT LEAK. Other/src/road_htr/trocr_cli.py kept the checkpoint with the
   lowest loss ON THE VALIDATION FOLD (load_best_model_at_end + early stopping).
   The held-out fold would then choose its own model and the OOF score would be
   optimistic. New opt-in key training.select_best: false trains a fixed number
   of epochs and keeps the final weights - the same policy as the Kraken members.
   Model3's default behaviour is unchanged.
2. TOKENIZER. The checkpoint ships only legacy vocab.json + merges.txt, which
   transformers 5 can no longer convert ("Couldn't instantiate the backend
   tokenizer"). make_tokenizer.py rebuilds the identical byte-level BPE with the
   `tokenizers` library and checks it: vocabulary 50,265, <s>=0 <pad>=1 </s>=2,
   exact round-trips on period spellings such as "y^t".
3. SPECIAL TOKENS. transformers 5 keeps decoder_start/pad/eos token ids only in
   generation_config, but the training forward pass shifts labels using the
   model.config copies and fails with AttributeError. trocr_cli now copies them
   across after loading (start 2, pad 1, eos 2).
4. warmup_ratio was removed from TrainingArguments in transformers 5; a float in
   [0, 1) passed as warmup_steps means the same ratio.
5. SAVING. With greedy decoding (num_beams 1) the generation config still set
   early_stopping=True; transformers 5 refuses to SAVE that combination, so the
   first fold-0 run trained all 4,100 steps and then lost its weights at
   save_model. early_stopping is now set only when num_beams > 1 (trainer and
   inference).
Fixes 1-4 were found by a one-step smoke test on the rented GPU before any fold
ran; fix 5 was not, because that test covered training and generation but not
saving. After fix 5 a one-off manual check saved, reloaded and generated from the
model before the relaunch; it is NOT automated in go154.sh. Any future
transformers upgrade should repeat a save/reload check before a full run.
The loader also reports encoder.pooler.* as newly initialised - harmless, TrOCR
does not use the pooler.

HOW TO RUN (reproducible without this session)
----------------------------------------------
On a Linux GPU host with /workspace/{code.tgz,images.tar,setup_host.sh}:
  bash Model/model154/go154.sh 0 1 2 3 4
go154.sh sets up the venv (Model/model146/setup_host.sh), downloads the checkpoint
to Other/checkpoints/model154/trocr-large-handwritten, builds its tokenizer.json,
links Other/data -> ../Data (trocr_cli resolves config paths against Other/), then
for each fold trains, predicts the held-out fold and the test set, and writes
  Other/outputs/model154_trocr/fold{k}/{oof_ep10_greedy.csv,test_beam1.csv,DONE}
It refuses to train unless all 5,472 images are present (see Model153's note on
truncated uploads).
Single commands:
  python Model/model154/train.py --fold N
  python Model/model154/infer.py --model <abs>/Other/outputs/model154/foldN/best_model \
      --input <abs>/Data/processed/train_folds.csv --fold N --output <abs>/out.csv

HARDWARE
--------
One RTX 5090 (32 GB), about 0.51 USD/h (AGENTS.md: Option A, one instance).
12.9 GB used at batch 4. About 1.45 steps/s: 4,100 steps, ~47 min per fold.
