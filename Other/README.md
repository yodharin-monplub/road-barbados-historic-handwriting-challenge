# R.O.A.D. Barbados Historic Handwriting Challenge

This repository is a reproducible experiment suite for the Zindi historical
handwriting challenge. Zindi first normalizes raw weighted WER by 12 and raw
weighted CER by 55, then averages their errors 50/50. The local scorer mirrors
that ranking objective and also reports both raw metrics.

## Directory layout (AGENTS.md, 2026-09-20)

```
AGENTS.md              the only file in the project root
Data/    raw/          competition images and CSVs (junction to the working disk,
                       so the 442MB image set exists exactly once)
         processed/    train_folds.csv (4,095 clean rows), test_ids_derived.csv
Model/   model1 .. model139   one folder per experiment: source, readme.txt, score.txt
Other/   src/ scripts/ config/ tests/   shared code, helpers, pinned settings
         outputs/ checkpoints/          junctions to the working disk
         submissions/ .env email.txt    submission CSVs and credentials
         README.md REPRODUCE.md MODEL_LICENSES.md
```

Every script locates the project root by walking up to `AGENTS.md`, so the tree
can be moved without editing paths. Commands are run from the root, and the
rented GPU hosts unpack the same three folders. `Other/REPRODUCE.md` is the
end-to-end recipe for the best submission.

## Current state (2026-09-18): rebuilt after losing the old machine

The Linux workstation that held every trained adapter, every saved prediction
and every submission is gone. Only code, readmes and public base weights
survived. Models 122-130 are the rebuild, trained on a Windows laptop
(RTX 4050 6GB) plus rented vast.ai GPUs; everything below this section is the
history of the lost pipeline and its predictions can no longer be reproduced.

| Model | What it is | Complete 5-fold OOF | Public |
|---|---|---:|---:|
| model122 | LightOnOCR-2-1B full fine-tune, native resolution, 4 epochs | 0.896608 | 0.907524 |
| model122_prod_lora | same base, rank-64 LoRA, 2 epochs | 0.892456 | - |
| model123 | Qwen3-VL-4B LoRA (second architecture) | ~0.838 | - |
| model125 | model122 + pseudo-labeled test images (round 1) | 0.899227 | - |
| model129 | pseudo-labels from the round-1 ensemble (round 2) | 0.900386 | - |
| model130 | PP-OCRv6/Kraken CTC member (organizer-approved weights) | 0.805866 | - |
| model134 | round-3 pseudo-labels (labels from the Kraken ensemble) | 0.902252 | - |
| model136 | Qwen2.5-VL-7B LoRA, third architecture family | **0.902391** | - |
| model124 | four-member anchored word vote | 0.905544 | 0.909996 |
| model129 | five-member vote, round-2 pseudo-labels | 0.906780 | 0.910016 |
| model130 vote | five members including the Kraken CTC model | 0.909376 | **0.913145** |
| model134 vote | same five members, round-3 anchor | 0.910200 | 0.913738 |
| model140 | round-4 pseudo-labels (six-member labeler) | 0.902625 | - |
| model136 vote | six members, adding Qwen2.5-VL-7B | 0.913774 | 0.915407 |
| **model140 vote** | **seven members, adding the round-4 pseudo member** | **0.914102** | **0.915619** |

Rejected, with measurements kept: model126 (text repair, exhausted), model127
(character-level vote, 0.903308 versus 0.905544), model132 (a second Kraken at
128px: -0.0006 on fold 0 - same architecture means correlated errors),
model128 (centre-crop member: ~14 hours per fold on the 6GB laptop, abandoned),
model135 (family-aware voting, +0.00004), model137 (forum review and label
audit: external data is forbidden, and only 1 of 4,095 rows looks mislabeled),
model138 (GRPO against the metric: 0.901960 -> 0.900793 over 75 steps, because
83% of sampled groups produce identical transcriptions and carry no signal),
model140 (fourth pseudo-label round: the member gains +0.00037 and the ensemble
+0.00033, a bootstrap interval of [-0.00052, +0.00118] - the ladder is spent),
model141 (GOT-OCR2, a fourth architecture: 0.813555 standalone but only +0.00035
on the fold-0 vote, so not worth the remaining budget - two real bugs found on
the way, a stop token the model never emits and a silently frozen vision
encoder, both written up because they cost a training run each),
model142 (quality-weighted voting: +0.00013 at best, and weighting by measured
accuracy is worse because it suppresses the weak decorrelated members).

Five fusion ideas have now returned nothing (126 text repair, 127 character
voting, 135 family-aware voting, 140 a fourth pseudo round, 142 weighted
voting). Combining the current members is exhausted; only new members move the
score, and the cheapest untried family now costs more per point than it returns.

Two facts that bound further work with a FIXED set of members:
- **Oracle ceiling 0.928476** over the five members of submission #5. Picking
  the best of them for every row - cheating by reading the labels - still falls
  short of 0.94, so no combination rule over *those* models reaches it
  (model135). This is a property of the members, not a law: model136's
  Qwen2.5-VL-7B raised the fold-0 vote by +0.0037 on its own, which is what
  adding a strong, decorrelated member does.
- **Run-to-run noise is about 0.003.** The identical supervised recipe and seed
  scored 0.905961 and 0.902701 on different GPUs and torch builds, which is
  larger than several recent single-fold "gains" (model138).

The leaderboard has rewarded architecture diversity far more than CV suggests
(+0.0031 public for the Kraken member's +0.0026 CV) and rewarded extra members
of the same family barely at all (+0.00002 public for +0.0012 CV).

Measured lessons from the rebuild (all on fold 0 unless stated):
- Upscaling the line crops to a fixed text height costs -0.029. LightOnOCR is
  pretrained at its own scale; `longest_edge` stays 1540.
- Full fine-tuning beats the old rank-64 LoRA recipe by about +0.003; beyond
  4 epochs it overfits. Beam-4 decoding adds +0.0013.
- Pseudo-labeling the test images (organizer-approved) is worth +0.0026 on the
  complete OOF, and a second round using ensemble labels adds +0.0012 more.
- Ensemble diversity beats member strength: Kraken at 0.81 adds +0.0031 to a
  0.909 fold-0 ensemble, more than any additional LightOnOCR sibling.
- The old text-repair chain (models 11-15, 54-66) is exhausted: refitted on the
  new predictions it changes 0-22 rows for no gain (model126).

Historical state (2026-09-04):

- The official CSV bundle and 442 MB image archive are present under
  `data/raw/`: 4,098 labeled rows, 1,374 test rows, and 5,472 valid RGB JPEGs.
- Every CSV ID resolves to exactly one image. There are no missing, corrupt,
  exact-duplicate, or cross-split duplicate images.
- Three objectively unusable/mismatched training rows were visually confirmed,
  documented in `config/suspect_row_audit.txt`, and excluded without changing
  the original data. The clean five-fold file has 4,095 rows (819 per fold).
- Dataset reports are in `reports/data_audit.json` and
  `reports/dataset_diagnostics.json`.
- Model 8's competition-approved PP-OCRv6 historical HTR weights and dedicated
  Kraken 7.1 environment are installed. Full, center, and enhanced fixed-view
  OCR is complete for all five OOF folds and test.
- Model 19's three-view correction chain scores 0.810909662 OOF (weighted WER
  3.272472, weighted CER 5.801107). Its validated submission was uploaded once
  and received the current public score, 0.814499013. No later candidate has
  been submitted.
- Model 66 remains the strongest promoted five-fold result at 0.817071245 OOF.
  Model68 measured 0.817088893 without a fold regression, but its tiny
  bootstrap interval crosses zero and it finds no test match, so it is not
  promoted and its test vector is byte-identical to Model66.
- Model 35 is the strongest conditional fold-0 result at 0.846941606 after the
  frozen context, boundary, and confusion stages. It is not a complete OOF
  score and is not yet a submission candidate; folds 1--4 are still required.
- Model 39 was rejected after its complete epoch-1 fold-0 validation scored
  0.354774323 raw and 0.357686046 after the frozen Model46 repair. It has no
  completion receipt and cannot support Models 44 or 49.
- Model 43 passed its real RTX 4050 smoke at 4.631 GiB peak allocated and
  4.857 GiB peak reserved. Its uncheckpointed 16-step timing pilot was stopped
  at roughly 80 seconds/update; it has no held-fold score.
- Model 50 was stopped after two complete epochs and 481/615 updates. Fold-0
  validation improved from 0.359866622 to 0.394881794, still decisively below
  the 0.819849390 Model66 fold-0 slice. Its frozen Model46 correction reached
  only 0.396277996, so further VLM training was not promoted.
- Model36 fold 1 is now actively training on the RTX 4050. It extends the fixed
  eight-epoch Kraken policy whose fold-0 terminal repair reached 0.846941606.
- Model 51 proves an exact 37-position trailing-logit loss against the stock
  loss/gradients and passed a 4.421/4.584 GiB smoke. Its two-step timing pilot
  ran at roughly 75 seconds/update and was stopped without a validation score.
- Model 52's frozen whole-line VLM/OCR guard was rejected by a one-shot negative
  control: it accepted 213 Model39 hallucinations and reduced the Model25
  fold-0 slice from 0.817450092 to 0.760894139. It must not be submitted.
- Model 53's token-local VLM fusion was much safer but still rejected: 20 token
  changes reduced the same negative-control slice to 0.816875829.
- Model 54 improves the complete Model25 OOF from 0.814147514 to 0.814251941
  with 24 final-token changes and improves every fold.
- Model 55 adds previous-token context and reaches 0.814343584 complete OOF.
- Model 56 adds next-token-conditioned initial spelling and reaches 0.814483608
  complete OOF; its validated local candidate adds two rows and is not uploaded.
- Model 57 adds unanimous medial three-token case context and reaches
  0.814617639 complete OOF; its validated candidate adds six rows and is not
  uploaded.
- Model 58 adds systematic mappings learned from other genuine OOF folds and
  reaches 0.816080816 complete OOF.
- A deployment-lineage audit found that the original Model23â€“60 named test
  candidates were rooted at raw Model19 text while their OOF chain was rooted
  after Models11â€“15. The public Model19 upload itself is verified correct. The
  old later candidates are marked do-not-submit and the corrected chain is
  receipted in `reports/deployment_lineage_audit.json`.
- Model 66 reaches 0.817071245 complete OOF. Its corrected-lineage candidate is
  the strongest CPU fallback and has not been uploaded.
- Model67 exact-context deletion was rejected at 0.817031001. Model68 exact
  full-line memory reaches 0.817088893 without regression, but its deployment
  vector is exactly Model66 because no test prediction matched its memory.
- Models 41, 45, and 47 remain blocked on the complete canonical Model32 raw
  OCR draft receipts. Model49 is implemented, but its required completed
  Model39 adapter does not exist because Model39 was stopped and rejected.

## Reproduce the data checks

The supplied files are already in place. These commands rebuild the audits and
fold assignments without editing the raw CSVs:

```bash
python scripts/audit_data.py --raw-dir data/raw --output reports/data_audit.json
python scripts/prepare_folds.py \
  --train-csv data/raw/Train.csv \
  --exclude-ids config/known_bad_train_ids.txt \
  --output data/processed/train_folds.csv
```

## Model ladder

Each numbered directory has its own `readme.txt` and is deliberately a distinct
source of signal:

1. `model1`: LightOnOCR-2-1B zero-shot/TTA baseline.
2. `model2`: LightOnOCR-2-1B-base LoRA domain adaptation.
3. `model3`: TrOCR large handwritten fine-tuning with aspect-preserving input.
4. `model4`: custom CRNN-CTC with legal LineMix augmentation and character LM.
5. `model5`: Qwen3-VL-4B QLoRA supervised fine-tuning.
6. `model6`: Qwen2.5-VL-7B QLoRA supervised fine-tuning.
7. `model7`: validation-tuned consensus, confidence filtering, and optional
   fully automated pseudo-label retraining.
8. `model8`: competition-approved PP-OCRv6 medium historical HTR, runnable on
   CPU for inference and fine-tunable through Kraken 7.1.
9. `model9`: confidence-gated full/center PP-OCR selector tuned only on OOF
   predictions and explicitly avoiding prohibited image metadata.
10. `model10`: fold-safe one-edit lexical repair learned only from supplied
    training transcriptions, with every automated change logged.
11. `model11`: fold-safe contextual one/two-edit repair using only unigram and
    bigram counts from supplied training transcriptions.
12. `model12`: fold-safe exact word split/merge repair learned only from
    supplied training transcriptions.
13. `model13`: fold-safe valid-word confusion repair gated by train-only
    context scores.
14. `model14`: fold-safe fuzzy joined-word repair against frequent train-only
    bigrams.
15. `model15`: inner-cross-fitted recurring OCR-to-truth error repair, with
    support, precision, length, and edit-distance gates.
16. `model16`: legal lower-tail character-confidence view selection; retained
    as a controlled comparison after matching, but not beating, Model9.
17. `model17`: word-aligned full/center fusion using only OCR text and
    per-character confidence.
18. `model18`: fold-excluded character-LM full/center selector; 0.749227 raw
    and 0.808762 after the correction chain.
19. `model19`: promoted three-view confidence/character-LM selector; 0.753782
    raw and 0.810910 after correction.
20. `model20`: rejected second recurring-error pass; its +0.000054 aggregate
    gain was not stable by fold.
21. `model21`: rejected supervised intrinsic-feature selector; 0.738740 OOF.
22. `model22`: memory-safe PP-OCRv6 domain fine-tuning, exact per-epoch
    competition scoring, and a full-data fit that retains all clean rows.
23. `model23`: deterministic `+` to `&` repair; 0.813091 OOF after Model19.
24. `model24`: fold-excluded historical-abbreviation caret restoration;
    0.813705 OOF.
25. `model25`: fold-excluded high-support unanimous casing restoration;
    0.814148 OOF.
26. `model26`: fixed equal-weight five-fold text medoid for deployment; no
    ordinary OOF claim because four of five models see each training row.
27. `model27`: narrow Qwen2-VL challenger, rejected after epoch 1 scored
    0.106889805 on fold 0.
28. `model28`: leakage-safe cross-fitted refit of the Models 11--15 repair
    chain for genuine complete OOF inputs; no new complete result yet.
29. `model29`: PP-OCR probability export and CTC prefix-beam/character-LM
    decoder; implemented but awaiting the required fixed-fold artifacts.
30. `model30`: metric-aware PP-OCR training-sampling experiment; manifests
    and controls are ready, but training has not started.
31. `model31`: leakage-safe ByT5 OCR corrector; the 0.877 GiB GPU smoke passed,
    but no fold training or score exists yet.
32. `model32`: OCR-guided Qwen challenger and receipt-v2 raw Model8 draft
    producer; canonical train/test draft bundles are not complete.
33. `model33`: Model25/Model22 selector; strict fold-0 inner cross-fitting
    scored 0.842145141, conditional on the screened Model22 checkpoint and not
    a complete five-fold OOF estimate.
34. `model34`: token-confidence fusion; 0.819147344 on fold 0, conditional and
    awaiting fixed-epoch confirmation on untouched folds.
35. `model35`: same-fold-model cross-fitted text repair; the frozen fold-0
    terminal scores 0.846941606, currently the strongest conditional result,
    but folds 1--4 are missing.
36. `model36`: fixed eight-epoch PP-OCR policy for folds 1--4; wrappers are
    ready, while those folds and the all-data model remain untrained.
37. `model37`: rejected selector/fusion stack whose partitions were not
    strictly nested; its exploratory apparent score is invalid as OOF.
38. `model38`: four-salt robustness audit supporting Model33's conditional
    fold-0 selector result; it does not remove checkpoint-screen optimism.
39. `model39`: broad low-resolution Qwen2-VL challenger, rejected at epoch 1:
    0.354774323 raw and 0.357686046 with frozen Model46 repair.
40. `model40`: 520--640-token full-vision Qwen2-VL challenger; CPU-audited,
    with no GPU smoke, fold score, or submission claim.
41. `model41`: Model39-style Qwen with a receipted raw Model8 draft; blocked on
    the incomplete canonical Model32 draft bundle.
42. `model42`: conservative Model35-versus-Model25 selector audit; selected no
    Model25 rows and was rejected for zero gain over 0.846941606.
43. `model43`: organizer-aligned rank-16 high-resolution Qwen2-VL; real smoke
    passed at 4.631/4.857 GiB allocated/reserved, but its uncheckpointed
    16-step timing pilot was stopped at about 80 seconds/update; no score.
44. `model44`: frozen greedy-versus-beam-3 whole-vector decoder panel; blocked
    because rejected Model39 has no completed adapter receipt.
45. `model45`: high-resolution full-vision Qwen plus raw OCR draft; CPU-audited
    and blocked on the canonical Model32 draft bundle.
46. `model46`: frozen Model35 fold-0 repair transfer for complete challenger
    vectors; it raised rejected Model39 from 0.354774323 to 0.357686046 only.
47. `model47`: Model43 high-resolution recipe plus raw OCR draft; CPU-audited
    and blocked on the canonical Model32 draft bundle.
48. `model48`: rank-16 mid-resolution (520--640 token) fallback; CPU-audited
    with no GPU smoke or score.
49. `model49`: frozen no-repeat-4 versus no-repeat-8 decoder comparison; ready
    in code but unusable because Model39 is incomplete and rejected.
50. `model50`: geometry-only adaptive 630/1,024-token high-resolution Qwen;
    standalone hard-case smoke passed at 4.614/4.838 GiB; fold-0 training is
    active, with no score yet.
51. `model51`: exact 37-position trailing-logit high-resolution Qwen; smoke
    passed at 4.421/4.584 GiB and a two-step timing pilot was stopped at about
    75 seconds/update, with no fold score.
52. `model52`: frozen whole-line Model50/Model25 language guard; rejected after
    its Model39 negative control fell to 0.760894139 versus 0.817450092 for the
    fallback, proving fluent-text selection is not safe enough.
53. `model53`: train-attested aligned-token VLM/OCR fusion; rejected after its
    frozen Model39 negative control changed 20 tokens and slipped to 0.816875829.
54. `model54`: unanimous fold-excluded final-token spelling/punctuation
    canonicalization; complete OOF 0.814251941, with all five folds improving.
55. `model55`: previous-token-conditioned final-token canonicalization;
    complete OOF 0.814343584, with all five folds improving again.
56. `model56`: next-token-conditioned initial-token canonicalization; complete
    OOF 0.814483608, with four folds improving and one unchanged.
57. `model57`: unanimous medial case restoration under both-neighbor context;
    complete OOF 0.814617639, with all five folds improving.
58. `model58`: recurring OCR-error repair learned from the other genuine OOF
    folds; complete OOF 0.816080816, with a strong positive bootstrap interval.
59. `model59`: conservative second outer-fold residual error pass; complete OOF
    0.816206285, with no fold regression and a positive bootstrap interval.
60. `model60`: outer-fold contextual spelling repair; complete OOF 0.816334380,
    superseded by Model63's safer frequency gate.
61. `model61`: exact recurring one/two-token replacement spans; complete OOF
    0.816661718 with a wholly positive incremental bootstrap interval.
62. `model62`: exact-context insertion repair; complete OOF 0.816829814, with
    every fold and every bootstrap resample improving.
63. `model63`: frequency-gated contextual repair replacing Model60; complete
    OOF 0.816401608, with all folds improving.
64. `model64`: exact-span repair refitted after Model63; complete OOF
    0.816746101 with a wholly positive incremental interval.
65. `model65`: context insertion repair refitted after Model64; strongest
    complete OOF 0.816914198 at that stage.
66. `model66`: exact three-token-context substitution repair; strongest
    complete deployable OOF 0.817071245 and current corrected-lineage CPU candidate.
67. `model67`: exact-context single-token deletion experiment; rejected after
    decreasing complete OOF to 0.817031001.
68. `model68`: exact case-sensitive full-line translation memory; exploratory
    OOF 0.817088893, rejected for an interval crossing zero and zero test changes.
69. `model69`: exploratory leave-one-outer-fold-out checkpoint selector; ready
    after all five checkpoint leaderboards exist. Selection is not fully nested
    because the other fold models trained on the held-out fold's labels.
70. `model70`: fail-closed monitored continuation for Model36 folds 2--4,
    exact evaluation, Model35 CPU postprocessing, and final local OOF assembly.

Do not spend leaderboard submissions merely to choose hyperparameters. Select
models by out-of-fold weighted WER/CER, then use the public leaderboard only as
a final sanity check. Keep at least one diverse ensemble among the two entries
selected for private judging.

## Rebuild the submitted Model19 baseline

The compact JSON files beside Models 11â€“15 and 19 freeze the measured all-fold
settings. Rebuild the selected test view and validated final candidate with:

```bash
python model19/apply_three_view_lm.py \
  --train data/processed/train_folds.csv \
  --test data/raw/Test.csv \
  --primary outputs/model8/test_full.jsonl \
  --secondary outputs/model8/test_center.jsonl \
  --tertiary outputs/model8/test_enhanced.jsonl \
  --config model19/allfold_config.json \
  --output outputs/model19/test.csv
python scripts/build_postprocessed_submission.py \
  --profile three-view \
  --config-set allfold \
  --predictions outputs/model19/test.csv \
  --work-dir outputs/pipeline_allfold \
  --supervised-config model15/allfold_three_view_config.json \
  --training-predictions outputs/model19/allfold_stage14.csv \
  --submission submissions/model19_three_view_allfold.csv
python scripts/validate_submission.py \
  --submission submissions/model19_three_view_allfold.csv \
  --sample data/raw/SampleSubmission.csv
```

## Measured five-fold results

| Candidate | Weighted WER | Weighted CER | Multi-score |
|---|---:|---:|---:|
| PP-OCR full view | 4.866463 | 7.613014 | 0.728022 |
| Model17 token fusion | 4.687593 | 7.206053 | 0.739174 |
| Model18 two-view character-LM selector | 4.506512 | 6.930229 | 0.749227 |
| Model19 three-view character-LM selector | 4.425458 | 6.800636 | 0.753782 |
| Full-view Models 11â€“15 chain | 3.510850 | 6.407536 | 0.795464 |
| Model18 Models 11â€“15 chain | 3.305314 | 5.886855 | 0.808762 |
| **Model19 Models 11â€“15 chain** | **3.272472** | **5.801107** | **0.810910** |
| Model23 ampersand repair | 3.229595 | 5.757638 | 0.813091 |
| Model24 caret restoration | 3.217720 | 5.744532 | 0.813705 |
| **Model25 casing restoration** | **3.208428** | **5.738480** | **0.814148** |
| **Model54 final-token canonicalization** | **3.206183** | **5.737282** | **0.814252** |
| **Model55 contextual final-token canonicalization** | **3.203744** | **5.738379** | **0.814344** |
| **Model56 contextual initial-token canonicalization** | **3.201185** | **5.734704** | **0.814484** |
| **Model57 contextual medial-case restoration** | **3.198543** | **5.732069** | **0.814618** |
| **Model58 outer-fold error repair** | **3.171151** | **5.696667** | **0.816081** |
| **Model59 residual outer-fold error repair** | **3.169027** | **5.692602** | **0.816206** |
| Model60 contextual spelling repair | 3.166584 | 5.689706 | 0.816334 |
| Model61 exact-span repair | 3.160677 | 5.680776 | 0.816662 |
| Model62 context insertion repair | 3.158226 | 5.673520 | 0.816830 |
| Model63 gated contextual repair | 3.165530 | 5.687145 | 0.816402 |
| Model64 refitted exact-span repair | 3.159338 | 5.677628 | 0.816746 |
| **Model65 refitted context insertion repair** | **3.156887** | **5.670372** | **0.816914** |
| **Model66 exact-context substitution repair** | **3.153958** | **5.666524** | **0.817071** |
| Model67 exact-context deletion (rejected) | 3.154465 | 5.668628 | 0.817031 |
| Model68 exact full-line memory (not promoted) | 3.153603 | 5.666207 | 0.817089 |

Model68's unrounded multi-score is 0.817088893, versus 0.817071245 for Model66,
but its incremental interval is [-0.000019653, +0.000068494] and one changed
row regresses. It has no test-side match, so Model66 remains the strongest
promoted deployment vector. The larger scores below are explicitly conditional fold-0
experiments and must not be compared as complete OOF evidence.

## Later conditional fold-0 results

| Candidate | Fold-0 multi-score | Status |
|---|---:|---|
| Model22 fixed epoch 7, raw | 0.813019600 | screened fold-0 checkpoint |
| Model25 baseline on fold 0 | 0.817450092 | slice of complete OOF |
| Model34 token fusion | 0.819147344 | inner-cross-fitted, conditional |
| Model33 row selector | 0.842145141 | strict inner cross-fit, conditional |
| **Model35 terminal confusion repair** | **0.846941606** | strongest conditional fold 0; not full OOF |
| Model42 selector audit | 0.846941606 | zero rows changed; rejected |
| Model39 epoch 1, raw | 0.354774323 | rejected |
| Model39 plus frozen Model46 repair | 0.357686046 | rejected |
| Model52 on Model39 negative control | 0.760894139 | rejected; below Model25 fold-0 fallback |
| Model53 on Model39 negative control | 0.816875829 | rejected; below Model25 fold-0 fallback |

Model19 final fold scores are 0.814765, 0.810079, 0.810315, 0.808457, and
0.810929. A seeded 2,000-sample paired row bootstrap of Model19 versus Model18
gives `+0.002148`, with percentile interval `[+0.001244, +0.002991]` and a
positive gain in every resample.

## Earlier fold-0 development results

All values below use the same 819 held-out clean rows and the local reproduction
of Zindi's normalized multi-metric.

| Candidate | Weighted WER | Weighted CER | Multi-score |
|---|---:|---:|---:|
| PP-OCR full, pretrained decoder | 4.827372 | 7.488736 | 0.730780 |
| PP-OCR full, allowed-character decoder | 4.817187 | 7.467912 | 0.731394 |
| PP-OCR center, allowed-character decoder | 4.902869 | 7.478365 | 0.727729 |
| Model9 confidence gate, allowed-character decoder | 4.642130 | 7.043966 | 0.742542 |
| Model17 token-confidence fusion, before text repair | 4.613859 | 7.017782 | 0.743958 |
| Model10 fold-safe lexical repair on Model9 | **3.951156** | **6.572657** | **0.775617** |
| Model11 contextual two-edit repair on Model9 | **3.669149** | **6.192340** | **0.790825** |
| Model12 exact boundary repair on Model11 | **3.531495** | **6.122760** | **0.797193** |
| Model13 valid-word confusion repair on Model12 | **3.482314** | **6.107284** | **0.799383** |
| Model14 approximate joined-word repair on Model13 | **3.346919** | **5.942944** | **0.806518** |
| Model15 cross-fitted recurring-error repair on Model14 | **3.335421** | **5.912943** | **0.807270** |
| Full-view Model14 chain (initial-submission path) | 3.483584 | 6.307098 | 0.797513 |

As a conditional sanity check, a seeded 1,000-sample paired row bootstrap put
the fold-0 Model9-to-Model14 gain at `+0.063976` with a percentile interval of
`[+0.058425, +0.069579]`; every individual stage also had a positive interval.
This does not replace untouched-fold confirmation because settings were chosen
on fold 0. Reproduce the interval with `scripts/bootstrap_stage_gains.py`.

These are validation results, not hidden-test or prize guarantees. The next
large gain should come from fold-trained domain adaptation and heterogeneous
VLM/CTC consensus.

## Rules encoded here

- Only challenge data is used. Pretrained weights must be public and permit
  downstream commercial use.
- No manual label edits or test-specific corrections.
- Image size, aspect ratio, pixel count, and other image metadata are never used
  to choose predictions; Zindi's general computer-vision rules prohibit this.
- Seeds, exact configurations, predictions, and excluded train IDs are logged.
- Empty predictions are rejected before submission.
- CHURRO-3B is intentionally absent because Zindi ruled its research licence
  ineligible for this challenge.

## Shared utilities

```bash
python scripts/score_csv.py --truth data/raw/Train.csv --predictions path/to/oof.csv
python scripts/validate_submission.py --submission submissions/candidate.csv --sample data/raw/SampleSubmission.csv
python -m unittest discover -s tests -v
```

Model weights, downloaded data, caches, and submissions are ignored by Git.

## Competition references

- [Official challenge page](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge)
- [Zindi general competition rules](https://zindi.world/rules)
- [Organizer approval of the PP-OCRv6 historical model](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34481)
- [Organizer clarification on automated pseudo-labeling](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34459)
- [Approved PP-OCRv6 historical weights](https://zenodo.org/records/21788410)

