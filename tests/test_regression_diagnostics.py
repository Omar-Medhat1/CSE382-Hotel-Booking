"""Tests for the regression error calculations and category groups."""
import unittest

import numpy as np
import pandas as pd

from scripts.regression_diagnostics import category_familiarity, summarize_residuals
from src.features.policy import CATEGORICAL


class RegressionDiagnosticsTests(unittest.TestCase):
    def _categorical(self, countries):
        result = pd.DataFrame({col: ["A"] * len(countries) for col in CATEGORICAL})
        result["Country"] = countries
        return result

    def test_residual_signs_and_metrics(self):
        summary = summarize_residuals(
            [0, 100, 200], [10, 70, 220], ["no_rare_or_unseen"] * 3)
        overall = summary.loc[summary.group == "overall"].iloc[0]
        # Actual minus predicted: -10, +30, -20; MAE = 20.
        self.assertEqual(overall["count"], 3)
        self.assertAlmostEqual(overall["mean_residual_actual_minus_predicted"], 0)
        self.assertAlmostEqual(overall["mae"], 20)
        self.assertAlmostEqual(overall["rmse"], np.sqrt((100 + 900 + 400) / 3))
        self.assertEqual(overall["negative_predictions"], 0)

    def test_zero_adr_and_negative_predictions_are_reported(self):
        result = summarize_residuals(
            [0, 80, 100], [-5, 90, 110], ["no_rare_or_unseen"] * 3)
        zero = result.loc[(result.group == "observed_adr_band") &
                          (result.segment == "ADR = 0")].iloc[0]
        self.assertEqual(zero["count"], 1)
        self.assertEqual(zero["negative_predictions"], 1)
        self.assertEqual(zero["mae"], 5)

    def test_training_categories_only_and_missing_vs_unseen(self):
        train = self._categorical(["EG"] * 11 + ["FR"] * 2)
        valid = self._categorical(["EG", "FR", "US", np.nan])
        labels = category_familiarity(train, valid)
        self.assertEqual(labels.tolist(), ["no_rare_or_unseen", "at_least_one_rare",
                                           "at_least_one_unseen", "no_rare_or_unseen"])

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            summarize_residuals([10], [10, 20], ["no_rare_or_unseen"])
        with self.assertRaises(ValueError):
            summarize_residuals([-1], [10], ["no_rare_or_unseen"])


if __name__ == "__main__":
    unittest.main()
