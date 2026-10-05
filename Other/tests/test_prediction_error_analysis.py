import unittest

from scripts.analyze_predictions import add_score_contributions
from road_htr.metrics import competition_error


class PredictionErrorAnalysisTests(unittest.TestCase):
    def test_contributions_sum_to_exact_metric_with_role_and_space_handling(self):
        references = ["first  reference", "assistant: second reference here", "short"]
        predictions = ["first reference", "<|assistant|>second wrong here", "short"]
        rows = [{"reference": ref, "prediction": pred} for ref, pred in zip(references, predictions)]
        add_score_contributions(rows)
        score = competition_error(references, predictions)
        self.assertAlmostEqual(sum(row["score_loss_contribution"] for row in rows), score["mean_error"], places=14)
        self.assertEqual(rows[2]["score_loss_contribution"], 0)
        self.assertEqual(rows[0]["word_edits"], 0)
        self.assertEqual(rows[0]["char_edits"], 1)

    def test_short_high_cer_row_can_have_lower_competition_impact(self):
        rows = [{"reference": "a", "prediction": "x"},
                {"reference": "one two three four five six seven eight nine ten", "prediction": "one two three four five six"}]
        add_score_contributions(rows)
        self.assertGreater(rows[0]["char_error_rate"], rows[1]["char_error_rate"])
        self.assertLess(rows[0]["score_loss_contribution"], rows[1]["score_loss_contribution"])

    def test_empty_evaluation_rejected(self):
        with self.assertRaises(ValueError):
            add_score_contributions([])


if __name__ == "__main__":
    unittest.main()
