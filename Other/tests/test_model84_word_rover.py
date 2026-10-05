import unittest

from road_htr.word_rover import consensus_line


class WordRoverTest(unittest.TestCase):
    def test_three_vote_substitution_wins(self) -> None:
        value, diagnostic = consensus_line(
            "the cat sat",
            ["the cat sat", "the dog sat", "the dog sat", "the dog sat", "the cot sat"],
        )
        self.assertEqual(value, "the dog sat")
        self.assertEqual(diagnostic["changes"][0]["kind"], "substitution")

    def test_ambiguous_substitution_keeps_medoid(self) -> None:
        value, diagnostic = consensus_line(
            "the cat sat",
            ["the cat sat", "the dog sat", "the dog sat", "the cot sat", "the cot sat"],
        )
        self.assertEqual(value, "the cat sat")
        self.assertEqual(diagnostic["changes"], [])

    def test_three_vote_deletion_wins(self) -> None:
        value, diagnostic = consensus_line(
            "the old cat",
            ["the old cat", "the old cat", "the cat", "the cat", "the cat"],
        )
        self.assertEqual(value, "the cat")
        self.assertEqual(diagnostic["changes"][0]["kind"], "deletion")

    def test_three_vote_insertion_wins(self) -> None:
        value, diagnostic = consensus_line(
            "the cat",
            ["the cat", "the cat", "the black cat", "the black cat", "the black cat"],
        )
        self.assertEqual(value, "the black cat")
        self.assertEqual(diagnostic["changes"][0]["kind"], "insertion")

    def test_requires_exactly_five_nonempty_sources(self) -> None:
        with self.assertRaises(ValueError):
            consensus_line("a", ["a"] * 4)
        with self.assertRaises(ValueError):
            consensus_line("a", ["a", "a", "a", "a", ""])


if __name__ == "__main__":
    unittest.main()
