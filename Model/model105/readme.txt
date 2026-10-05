Model105: all-data augmented LightOnOCR deployment
==================================================

Model105 refits the promoted Model92 augmentation recipe on all 4,095 supplied
training labels.  Model92 fixed checkpoint 400 using canonical fold 0 and then
confirmed the recipe over all five heldout folds before this refit.

Each held-fold model saw 3,276 rows for 400 effective batches of 16, or 6,400
example exposures (1.9536 passes).  The all-data refit therefore uses 500
steps, or 8,000 / 4,095 = 1.9536 passes.  All other optimizer, LoRA, seed,
augmentation, base-revision, and generation settings are unchanged.

GPU commands:

  PYTHONPATH=src .venv-lighton/bin/python model105/train.py

  PYTHONPATH=src .venv-lighton/bin/python model105/infer.py \
    --adapter outputs/model105/all/best_adapter \
    --input data/raw/Test.csv \
    --output outputs/model105/test.csv

This creates a deployment witness only.  Model104 decides whether to use it;
nothing is submitted automatically.

Automatic monitoring
--------------------

scripts/monitor_model105.py supervises the authorized Vast.ai instance 50076619
from this PC every 600 seconds. The remote pipeline runs training followed by
test inference. The supervisor downloads all five Model92 test predictions and
the Model105 predictions, verifies hashes and exact test IDs, and preserves the
Model105 adapter and training manifest locally before stopping the instance.
It then applies the already fitted Model104 selector and validates
submissions/model104_model105_selector.csv. It does not upload to Zindi.
Raw empty predictions remain intact in the downloaded files. During submission
construction, any blank Model105 prediction falls back to Model102 on that ID;
the affected IDs are recorded in outputs/model104/deployment_report.json.

The local service is road-model105-supervisor.service. Status and logs are in
outputs/model105/supervisor_status.json and supervisor.log. The PC must stay
awake and online. A failure is reported as needs_attention; it never destroys
the instance. Stopped-instance storage charges remain until separately removed.

Completed deployment (2026-09-07)
-------------------------------
Training completed 500 steps on 4,095 rows in 826.1 seconds. All 1,374 test
predictions and the best_adapter checkpoint were downloaded and SHA-256 checked.
One blank raw transcript falls back to Model102 in the final selector input.
Vast instance 50076619 was stopped automatically after artifact verification.
The API reports actual_status=exited with cur_state/intended_status=stopped;
the supervisor's original check expected actual_status=stopped and incorrectly
reported a failure. That check was corrected and local submission construction
completed. Model105 is an all-data deployment refit and has no independent OOF.
