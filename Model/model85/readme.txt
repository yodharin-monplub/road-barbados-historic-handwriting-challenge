Model 85: all-data LightOnOCR TTA medoid
========================================

Model85 deploys the Model83 rule only after it improved two independent held-out
folds. The exact combined 1,638-row score was 0.8990778006 versus 0.8930610892
for the original full image view, an uplift of 0.0060167114. Fold-specific
uplifts were +0.0065807486 and +0.0054525000.

The deployment sources are frozen in this order: Model77 full, enhanced,
center, center-enhanced, and Model35 Kraken-confusion. Model77 is the promoted
Model73 recipe refit on all 4,095 usable labeled rows for 500 optimizer steps,
which preserves the selected fold model's 1.9536-example exposure. The fixed
five-source rule first takes an exact majority and otherwise selects the source
string with minimum equal-weight mixed word/character distance. It never
synthesizes an unobserved character sequence.

Build the candidate after copying and hash-verifying all RunPod artifacts:

  PYTHONPATH=src python3 model85/apply.py \
    --test data/raw/Test.csv --sample data/raw/SampleSubmission.csv \
    --adapter outputs/model77/all/best_adapter \
    --source full=outputs/model83/model77_test/full.csv \
    --source enhanced=outputs/model83/model77_test/enhanced.csv \
    --source center=outputs/model83/model77_test/center.csv \
    --source center_enhanced=outputs/model83/model77_test/center_enhanced.csv \
    --kraken outputs/model35/deployment_epoch07/confusions_test.csv \
    --validation-report outputs/model83/fold0/report.json \
    --validation-report outputs/model83/fold1/report.json \
    --output outputs/model85/medoid_test.csv \
    --agreement-output outputs/model85/agreement.jsonl \
    --submission submissions/model85_model77_tta_medoid.csv \
    --report outputs/model85/report.json

The builder fails closed on test hashes, output hashes, fold/variant metadata,
empty predictions, adapter-tree drift, source order, or missing positive
validation evidence. It writes a complete receipt. Nothing uploads or submits
to Zindi automatically.
