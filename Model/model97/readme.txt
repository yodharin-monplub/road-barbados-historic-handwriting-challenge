Model97: blinded Codex-vision sample baseline
==============================================

This is a diagnostic, not a submission model.  Twenty-four fold-0 IDs were
sampled deterministically from the label-free Model95 image manifest without
consulting their targets.  Codex visually transcribed each original image.  The
predictions were saved before the training labels were loaded for scoring.

The sample is small and the result is only an indication of direct visual OCR
quality; it must not be compared as if it were the full 819-row fold score.

Result
------
Codex vision scored 0.7992230486 on the frozen 24-row sample.  Model88 scored
0.8885608994 on exactly the same rows, a difference of -0.0893378508.  This
rejects chat vision as a replacement OCR engine.  It does not measure Model95's
Qwen3-VL checkpoint, which is evaluated independently on all 819 fold-0 rows.

Artifacts:

  outputs/model97/codex_blind_predictions.csv
  outputs/model97/comparison.csv
  outputs/model97/report.json
