from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"model112"))
from fusion import fuse,vote


class FusionTests(unittest.TestCase):
    def test_similar_family_cannot_overrule_alone(self):
        values=dict(model73="SYM",model92="SYM",model107="SYM",kraken="SIM")
        self.assertEqual(vote(values,"SIM",True),"SIM")
        self.assertEqual(vote(values,"SIM",False),"SYM")
    def test_two_lighton_plus_kraken(self):
        values=dict(model73="SYM",model92="SYM",model107="SIM",kraken="SYM")
        self.assertEqual(fuse("SIM Esq.",{n:t+" Esq." for n,t in values.items()}),"SYM Esq.")
    def test_tie_preserves_anchor(self):
        self.assertEqual(vote(dict(model73="a",model92="b",model107="a",kraken="b"),"x",True),"x")
    def test_word_insertion(self):
        values=dict(model73="hath the land",model92="hath the land",model107="hath land",kraken="hath the land")
        self.assertEqual(fuse("hath land",values),"hath the land")
    def test_word_deletion(self):
        values=dict(model73="hath land",model92="hath land",model107="hath the land",kraken="hath land")
        self.assertEqual(fuse("hath the land",values),"hath land")
    def test_char_preserves_original_spacing(self):
        values=dict(model73="SYM Esq.",model92="SYM Esq.",model107="SIM Esq.",kraken="SYM Esq.")
        self.assertEqual(fuse("SIM  Esq.",values,character=True),"SYM  Esq.")
    def test_nonempty_and_source_identity(self):
        with self.assertRaises(ValueError): fuse("a",{"model73":"a"})
    def test_no_agreement_no_normalization(self):
        text="SYM  Esq."
        self.assertEqual(fuse(text,{n:text for n in ("model73","model92","model107","kraken")}),text)


if __name__=="__main__": unittest.main()
