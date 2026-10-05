import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from road_htr import stackmix
from road_htr.vlm_data import build_vlm_dataset


class CTCAlignmentTests(unittest.TestCase):
    def test_repeated_letters_require_blank(self):
        p=np.full((3,2),0.001)
        p[[0,2],1]=0.999
        p[1,0]=0.999
        spans=stackmix.ctc_align(p,[1,1])
        self.assertEqual([s[:2] for s in spans],[(0,1),(2,3)])

    def test_impossible_repetition_rejected(self):
        with self.assertRaises(ValueError):
            stackmix.ctc_align(np.array([[0.1,0.9],[0.1,0.9]]),[1,1])

    def test_invalid_probabilities_rejected(self):
        with self.assertRaises(ValueError):
            stackmix.ctc_align(np.array([[0.7,0.9]]),[1])


class SyntheticIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.patcher=patch.object(stackmix,"ROOT",self.root)
        self.patcher.start()
        self.truth=self.root/"truth.csv"
        self.truth.write_text("ID,Target,fold\na,SYM Esq.,1\nb,other,0\n")
        self.eligible={"a":{"ID":"a","Target":"SYM Esq.","fold":"1"}}
        Image.new("RGB",(30,12),"white").save(self.root/"image.png")
        self.manifest={"method":"stackmix","fold":0,"truth_sha256":stackmix.sha256(self.truth),
            "rows":[{"id":"syn1","target":"SYM Esq.","target_source_id":"a",
                "image":"image.png","sha256":stackmix.sha256(self.root/"image.png"),
                "source_ids":["a"],"donors":[
                    {"source_id":"a","text":"SYM","text_span":[0,3]},
                    {"source_id":"a","text":"Esq.","text_span":[4,8]}]}]}
        self.bank=self.root/"bank.json"
        self.bank.write_text(json.dumps({"fold":0,"lineage":{"training_excluded_folds":[0]},
            "words":self.manifest["rows"][0]["donors"]}))
        self.manifest["bank_path"]="bank.json"
        self.manifest["bank_sha256"]=stackmix.sha256(self.bank)

    def tearDown(self):
        self.patcher.stop()
        self.tmp.cleanup()

    def validate(self):
        path=self.root/"manifest.json"
        path.write_text(json.dumps(self.manifest))
        return stackmix.validate_synthetic(path,self.truth,0,self.eligible,"Target")

    def test_valid_case_and_punctuation_preserved(self):
        self.assertEqual(self.validate()[0]["target"],"SYM Esq.")

    def test_all_data_manifest_cannot_be_used_for_cv(self):
        self.manifest["fold"]=None
        bank=json.loads(self.bank.read_text())
        bank.update(fold=None,lineage={"training_excluded_folds":[]})
        self.bank.write_text(json.dumps(bank))
        self.manifest["bank_sha256"]=stackmix.sha256(self.bank)
        path=self.root/"manifest.json"
        path.write_text(json.dumps(self.manifest))
        self.assertEqual(len(stackmix.validate_synthetic(path,self.truth,None,self.eligible,"Target")),1)
        with self.assertRaisesRegex(ValueError,"fold/truth mismatch"):
            self.validate()

    def test_heldout_source_rejected(self):
        self.manifest["rows"][0]["source_ids"].append("b")
        with self.assertRaises(ValueError): self.validate()

    def test_fold_mismatch_rejected(self):
        self.manifest["fold"]=1
        with self.assertRaises(ValueError): self.validate()

    def test_wrong_spelling_rejected(self):
        self.manifest["rows"][0]["target"]="SIM Esq."
        with self.assertRaises(ValueError): self.validate()

    def test_aligner_trained_on_holdout_rejected(self):
        bank=json.loads(self.bank.read_text())
        bank["lineage"]["training_excluded_folds"]=[]
        self.bank.write_text(json.dumps(bank))
        self.manifest["bank_sha256"]=stackmix.sha256(self.bank)
        with self.assertRaisesRegex(ValueError,"validation fold excluded"): self.validate()

    def test_changed_pixels_rejected(self):
        Image.new("RGB",(30,12),"black").save(self.root/"image.png")
        with self.assertRaises(ValueError): self.validate()

    def test_duplicate_id_rejected(self):
        self.manifest["rows"].append(copy.deepcopy(self.manifest["rows"][0]))
        with self.assertRaises(ValueError): self.validate()

    def test_unapproved_generator_rejected(self):
        self.manifest["method"]="diffusionpen"
        with self.assertRaises(ValueError): self.validate()

    def test_training_mix_and_validation_exclusion(self):
        self.truth.write_text("ID,Target,fold\na,SYM Esq.,1\nb,heldout,0\nc,SYM Esq.,1\nd,SYM Esq.,1\ne,SYM Esq.,1\n")
        for identifier in "abcde":
            Image.new("RGB",(30,12),"white").save(self.root/f"{identifier}.png")
        self.manifest["truth_sha256"]=stackmix.sha256(self.truth)
        path=self.root/"manifest.json"
        path.write_text(json.dumps(self.manifest))

        class FakeDataset:
            @classmethod
            def from_dict(cls,data):
                result=cls()
                result.data=data
                return result
            def cast_column(self,*args): return self
            def set_transform(self,fn): self.transform=fn

        module=types.SimpleNamespace(Dataset=FakeDataset,Image=lambda **kwargs:None)
        with patch.dict("sys.modules",{"datasets":module}):
            result=build_vlm_dataset(self.truth,self.root,fold=0,fold_mode="train",synthetic_manifest=path)
            self.assertEqual(result.data["id"],["a","c","d","e","syn1"])
            self.assertNotIn("b",result.data["id"])
            self.assertEqual(result.data["completion"][-1][0]["content"][0]["text"],"SYM Esq.")
            with self.assertRaisesRegex(ValueError,"never enter validation"):
                build_vlm_dataset(self.truth,self.root,fold=0,fold_mode="validation",synthetic_manifest=path)
            baseline=build_vlm_dataset(self.truth,self.root,fold=0,fold_mode="train")
            self.assertEqual(len(baseline.data["id"]),4)


if __name__=="__main__":
    unittest.main()
