Model118: remove the LightOnOCR repetition penalty
================================================
Hypothesis: the1.05 token repetition penalty can discourage legitimate repeated
words/subwords in legal text. Test neutral penalty1.0, not dictionary correction.
This is inference-only using Model114's already-trained fold0 step400 adapter.
No retraining, external data, synthetic examples or test predictions.
Everything except repetition_penalty remains identical to Model114 inference.
Score all819 original heldout images; do not inspect only favorable examples.
Compare standalone score against Model114. Also replace only Model92's voter
in Model112's family-word rule, with the same original Model104 anchor and the
same other3 voters. Compare with Model114's replacement ensemble as well.
Advance screen: standalone gain>=0.001 vs Model114 AND ensemble gain>=0.0005
vs Model112. These are development results on previously reused folds, not
unbiased validation. Any passing recipe still needs other-fold confirmation.
No submission is prepared or uploaded automatically.

Run .venv-lighton/bin/python model118/experiment.py prepare, then
.venv-lighton/bin/python scripts/run_model118.py.
The supervisor waits for the existing Model117 service to finish successfully
before touching CUDA, then runs inference and scoring with600-second checks.
It does not stop other GPU processes or edit Model117's active files.
Artifacts: outputs/model118/protocol.json, status.json, fold0_oof.csv, report.json.
Original source hashes, Model114 adapter tree, runtime and new code/config are
frozen before inference. The default1.05-penalty predictions remain preserved.

Completed (2026-09-11): all819 predictions,53 lines changed,0 blanks.
Standalone0.8918241921 vs Model1140.8917072302: +0.0001169620.
Ensemble0.9056760721 vs Model1120.9054038514: +0.0002722207,
but below Model114's replacement ensemble0.9057409291. Screen failed.
No submission prepared; do not change the production decoding penalty.
