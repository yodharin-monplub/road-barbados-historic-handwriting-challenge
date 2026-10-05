from __future__ import annotations

import unittest

from model66.contextual_substitution_repair import fit_mapping, load_config, repair_contextual_substitutions


class Model66ContextualSubstitutionTests(unittest.TestCase):
    def test_config_is_frozen(self) -> None:
        config,_=load_config("model66/config.json"); self.assertEqual(config["minimum_precision"],0.8)

    def test_context_mapping_is_learned(self) -> None:
        config,_=load_config("model66/config.json")
        mapping=fit_mapping(["for dues good"]*2,["for divers good"]*2,config)
        self.assertEqual(mapping[("for","dues","good")]["target"],"divers")

    def test_same_token_outside_context_is_preserved(self) -> None:
        mapping={("for","dues","good"):{"target":"divers","support":2,"observations":2,"precision":1.0}}
        result,changes=repair_contextual_substitutions("other dues owed",mapping)
        self.assertEqual(result,"other dues owed"); self.assertEqual(changes,[])

    def test_contextual_replacement(self) -> None:
        mapping={("for","dues","good"):{"target":"divers","support":2,"observations":2,"precision":1.0}}
        result,changes=repair_contextual_substitutions("for dues good causes",mapping)
        self.assertEqual(result,"for divers good causes"); self.assertEqual(changes[0]["after"],"divers")


if __name__ == "__main__": unittest.main()
