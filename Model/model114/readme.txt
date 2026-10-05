Model114: local LightOnOCR seed diversity pilot
=============================================
Seed: 20260914. Model115 uses 20260915.
Goal: improve the Model112 family-aware word ensemble, not just standalone OCR.
Start each adapter independently from the pinned Model92 base checkpoint.
No external data, synthetic data, dictionary normalization, or test-label use.

Training: canonical fold0 held out (819 rows); 3276 other real rows for training.
Model92 augmentation, full images, rank64/alpha128/dropout0.05 LoRA, bf16.
400 optimizer steps, learning rate 0.0001, cosine schedule, 20 warmup steps.
Local RTX4050 6GB: batch1, accumulation16, effective batch16, evaluation batch1.
Only evaluate/save at step400, so best_adapter is the final step400 adapter.
The smoke test has a separate 2-step adapter and never initializes a real run.

Not a pure seed-only replication of the previous cloud experiment: microbatch
size and worker count differ, and local Torch is 2.13.0+cu130 versus cloud2.14.
Model114 and Model115 use the same local runtime and recipe except random seed.

Frozen fold0 evaluation: compare standalone predictions to Model92; also replace
only the Model92 voter in the existing Model112 word-fusion recipe. Model73,
Model107, Kraken, the original Model104 anchor, and vote threshold stay fixed.
Do not give correlated LightOn models additional votes over Kraken.
Advance-to-confirmation screen: replacement ensemble gain >=0.0005 over Model112.
This is development evidence, not unbiased CV; these folds have selection history.
No submission or full-data training is automatically authorized by this screen.

Commands (from workspace root, .venv-lighton):
  python model114/pilot.py prepare
  python scripts/run_seed_pilots.py
The supervisor runs a real trainer smoke, then trains/infers/evaluates 114 and115
sequentially. It records progress every 600 seconds without repeated chat calls.
It stops on any failure and never rents, starts, stops, or destroys cloud GPUs.
Results: outputs/model114/report.json, outputs/model115/report.json.
Progress: outputs/model114/seed_pair_status.json and the per-stage logs.
Best existing submission is retained: Model112 public score 0.911129378.

Recovery after training (2026-09-10)
----------------------------------
Model114 completed all400 steps; validation loss0.47426903 (not competition CV).
The original supervisor's GPU-free check stopped before inference. No partial
predictions or Model115 training outputs were created. Checkpoint400 is retained.
Recovery: python scripts/resume_seed_pilots.py
This separately recorded operational recovery leaves the frozen protocol/code
unchanged. It verifies the completed adapter, resumes114 inference/evaluation,
then trains/infers/evaluates115. The GPU check now waits for memory release or
other jobs instead of failing immediately. It never terminates another process.
After a short teardown grace period, GPU waiting checks every600 seconds.
