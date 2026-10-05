import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"model111"))
import confirm


class ConfirmationTests(unittest.TestCase):
    def test_routes_distinct_fold_checkpoints(self):
        for judge in ("model73","model107"):
            paths=[confirm.fold_info(judge,f)["adapter"] for f in range(5)]
            self.assertEqual(len(set(paths)),5)
            for f,path in enumerate(paths): self.assertIn(f"fold{f}/",path)

    def test_invalid_fold_rejected(self):
        for fold in (-1,5,None):
            with self.assertRaises(ValueError): confirm.fold_info("model73",fold)

    def test_unknown_judge_rejected(self):
        with self.assertRaises(ValueError): confirm.fold_info("model105",1)

    def test_wrong_receipt_fold_rejected_before_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            p=root/"outputs/model73/fold1_oof_step400.manifest.json"
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps({"fold":0}))
            with patch.object(confirm,"ROOT",root):
                with self.assertRaisesRegex(ValueError,"wrong fold"): confirm.check_fold_judge("model73",1)

    def test_original_pilot_functions_reused(self):
        import score
        self.assertIs(confirm.likelihood,score.likelihood)
        self.assertIs(confirm.make_prefix,score.make_prefix)

    def test_fold_zero_not_prepared_again(self):
        with self.assertRaisesRegex(ValueError,"Preserve completed pilot"): confirm.prepare(0)


if __name__=="__main__": unittest.main()
