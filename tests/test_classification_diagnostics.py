"""Classification metric and probability-diagnostic checks."""
import unittest

import numpy as np

from src.evaluation.metrics import (classification, classification_decisions,
                                    classification_probability_summary)


class ClassificationDiagnosticsTests(unittest.TestCase):
    def test_threshold_and_confusion_matrix_use_class_one_as_positive(self):
        labels = np.array([0, 0, 1, 1])
        probability = np.array([0.1, 0.6, 0.4, 0.9])
        metrics = classification(labels, probability)
        np.testing.assert_array_equal(classification_decisions(probability), [0, 1, 0, 1])
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["precision"], .5)
        self.assertEqual(metrics["recall"], .5)

    def test_probability_summary_separates_true_classes(self):
        labels = np.array([0, 0, 1, 1])
        probability = np.array([0.1, 0.6, 0.4, 0.9])
        summary = classification_probability_summary(labels, probability).set_index("true_class")
        self.assertEqual(summary.loc[0, "rows"], 2)
        self.assertEqual(summary.loc[1, "rows"], 2)
        self.assertEqual(summary.loc[0, "decisions_at_0_5"], 1)
        self.assertEqual(summary.loc[1, "decisions_at_0_5"], 1)
        self.assertAlmostEqual(summary.loc[1, "median_probability"], .65)

    def test_invalid_probabilities_are_rejected(self):
        for probability in ([.1, np.nan], [-.1, .2], [.2, 1.1], [[.1, .2]]):
            with self.subTest(probability=probability):
                with self.assertRaises(ValueError):
                    classification_decisions(probability)


if __name__ == "__main__":
    unittest.main()
