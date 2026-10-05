Model 83: held-fold LightOnOCR multi-view screen
=================================================

Purpose
-------
Model83 tests whether label-free image-view diversity improves the Model73
LightOnOCR recognizer. On held-out fold 0 it decodes the frozen checkpoint-400
adapter with three already-supported deterministic views: enhanced full image,
center band, and enhanced center band. It combines those with the original full
view and the independent corrected Model35 Kraken output using Model26's fixed
equal-weight mixed-distance text medoid.

No leaderboard prediction or label is used to select a view. The evaluator
requires exactly the ordered sources full, enhanced, center, center_enhanced,
and kraken; verifies all four LightOnOCR inference manifests, image-variant
labels, truth hash, fold, output hash, and identical complete adapter trees;
then reports exact challenge scores for every source and the medoid.

When predictions are moved from a GPU host back to this workspace, pass
`--adapter-dir` with the complete local checkpoint. The evaluator verifies every
hash declared by each inference manifest against that checkpoint and requires
the six inference-critical adapter/processor files to be identical across all
four views. This permits a manifest made from an inference-only checkpoint copy
to be validated against its complete training checkpoint without trusting paths
that only existed on the GPU host.

GPU screen
----------

  bash model83/run_fold0.sh

Promotion policy
----------------
Fold 0 is the development screen. Continue to fold 1 only if the medoid gains
at least 0.001 absolute multi-score over the original full view, or if one
alternate view gains at least 0.003 and can define a simpler frozen candidate.
Freeze the chosen source set after fold 0. Promote to all five folds only if the
combined fold-0/1 score improves the original full-view baseline and fold 1
does not degrade by more than 0.001. Test inference happens only after the
five-fold OOF result is complete.

This staged gate avoids spending several GPU-hours on weak transforms. Model83
never uploads or submits to Zindi automatically.
