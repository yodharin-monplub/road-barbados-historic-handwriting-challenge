Model112: diverse-source, architecture-aware transcript ensemble
================================================================
Use saved Model73, Model92 and Model107 LightOnOCR predictions plus Model35
corrected Kraken. No new GPU training, external data, or cloud rental.

Four fixed recipes, specified before results:
1. family_word (primary): change an aligned word only if Kraken and at least
   two of the three LightOn models agree exactly, including insertion/deletion.
2. majority_word: exact three-of-four word majority, regardless of architecture.
3. family_char: same architecture-aware vote, character-level inside aligned
   words. Retain original word boundaries; do not delete whole words.
4. majority_char: same character method with ordinary three-of-four voting.
Raw anchor is Model107. Repeat against Model104 as an explicitly less reliable
development diagnostic, because its upstream ensemble is not fully nested.
Agreement among three LightOn variants alone cannot change the family-rule
anchor; a duplicate prediction is not added as a fifth independent vote.

Do not fit weights or thresholds from any evaluated label. Report every recipe
and every fold, not only the best outcome. Source-diversity measurements and a
label-oracle upper bound are diagnostic only, never deployment predictions.
Run: PYTHONPATH=src .venv-lighton/bin/python model112/run.py

Provenance: verify full canonical IDs, fold-specific source receipt hashes,
Model92 assembly ledger and Kraken fold-derived correction ledger. Model92's
original fold0 inference receipt is absent locally; its existing assembly hash
is a weaker witness. Folds1-4 have original receipts. All these folds have prior
development history; do not describe them as a newly untouched holdout.

No automatic test inference or submission. Continue toward deployment only if
there is credible gain over the best ensemble, not merely a weaker raw model.
Results and pairwise-error audit: outputs/model112/report.json.

Completed results (2026-09-10)
------------------------------
                         Raw Model107 anchor    Best-ensemble anchor (diagnostic)
Baseline                 0.8924795985           0.9014774424
Family word              0.8960740744           0.9036181592
Ordinary majority word   0.8960740744           0.9031075143
Family character         0.8963005647           0.9039049159
Ordinary majority char   0.8963005647           0.9026688351

The raw-anchor family/majority outputs are identical because Model107 is itself
one of the voters; changing its text needs the other three witnesses. They are
not separate independent models. Architecture-aware voting matters when the
anchor is the previously assembled best ensemble.

Primary family-word gains over the best ensemble by fold:
0 +0.0032334299; 1 +0.0022413601; 2 +0.0017882703;
3 +0.0021182126; 4 +0.0013226236.
Folds1-4 combined gain +0.0018674692, conditional paired-row bootstrap 95%
[+0.0013345243,+0.0023955808]. These remain historical development estimates.
Character-family results are retained as a secondary experiment, not selected
as the submission simply because their aggregate score is higher.

Diversification audit: Model92/107 error-cost correlation is 0.8485. Kraken is
weaker alone (0.84172), but has lower correlation and beats Model107 on 909 rows.
The four-source label-oracle score 0.92093 is a non-deployable diagnostic, NOT
an ensemble CV score or a prediction of leaderboard performance.

After checking the local results, replayed the primary word recipe on saved
test predictions, using matching all-data deployment models 77/105/107 and
corrected Kraken. All source hashes/IDs and LightOn adapter assets verified.
One row with blank source output keeps the best baseline rather than adding a
duplicated fallback vote. Twenty fusion/scoring/confirmation tests passed.

Prepared experimental submission (not uploaded):
submissions/model112_family_word_ensemble.csv
1374 rows; 51 changed lines versus the best Model104/105 submission.
SHA-256 f7f543d06f41ad2dd019090f913f8cc317c1e4c5e088f2cb2e67d4eceac4a380
Details: outputs/model112/submission_report.json and test_changes.csv.
Public score unknown. Keep the previous best submission; no improvement is
guaranteed. No GPU/cloud was used, and no training or monitoring job remains.

Public result supplied by user (submission uAXYp4BY): 0.911129378.
This is +0.001429924 above Model104/105's public 0.909699454. Preserve this
submission while Model113 separately tests the previously evaluated character
recipe; do not apply the character recipe on top of this word-voted output.
