"""Meaningful integrity checks: eligibility, time purging, and preprocessing isolation."""
import unittest
import numpy as np
import pandas as pd

from src.data.splitting import partition
from src.features.policy import (CATEGORICAL, GROUP_COLUMNS, NUMERIC, FORBIDDEN,
                                 engineering, select)
from src.preprocessing.pipelines import preprocessing


def fixture():
    dates = ["2016-06-01", "2016-06-02", "2017-02-01", "2017-02-02", "2017-06-01",
             "2016-08-01", "2016-08-02", "2016-12-30", "2017-04-29", "2016-08-03"]
    records = []
    for i, date in enumerate(dates):
        arrival = pd.Timestamp(date)
        record = {col: "A" if col in CATEGORICAL else 0 for col in GROUP_COLUMNS}
        record.update(LeadTime=10, ArrivalDateYear=arrival.year, ArrivalDateMonth=arrival.month_name(),
                      ArrivalDateDayOfMonth=arrival.day, Adults=2, Children=0, Babies=0,
                      StaysInWeekNights=2, StaysInWeekendNights=0,
                      ArrivalDate=arrival, StatusDate=arrival + pd.Timedelta(days=2),
                      ADR=100, IsCanceled=i % 2)
        records.append(record)
    records[5]["StatusDate"] = pd.Timestamp("2016-07-30")  # known before snapshot
    records[6]["StatusDate"] = pd.Timestamp("2016-08-01")  # same day: ambiguous
    records[7]["StatusDate"] = pd.Timestamp("2017-01-02")  # training label too late
    records[8]["StatusDate"] = pd.Timestamp("2017-05-02")  # validation label too late
    records[9]["LeadTime"] = 0  # booking did not exist on prior day
    return pd.DataFrame(records)


class ProtocolTests(unittest.TestCase):
    def test_snapshot_eligibility_and_temporal_purge(self):
        labels, _, metadata = partition(fixture())
        self.assertEqual(labels.iloc[:5].tolist(), ["train", "train", "validation", "validation", "test"])
        self.assertEqual(labels.iloc[[5, 6, 9]].tolist(), ["ineligible_at_snapshot"] * 3)
        self.assertEqual(labels.iloc[7], "purged_train_late_label")
        self.assertEqual(labels.iloc[8], "purged_validation_late_label")
        self.assertTrue(all(value == 0 for value in metadata["predictor_group_overlap"].values()))

    def test_outcomes_never_enter_model_inputs(self):
        frame = fixture()
        for task in ("regression", "classification", "clustering"):
            self.assertFalse(set(select(frame, task).columns) & FORBIDDEN)

    def test_training_statistics_and_unknown_categories(self):
        frame = fixture().iloc[:3].copy()
        frame.loc[0, "Children"] = np.nan
        frame.loc[1, "Children"] = 0
        frame.loc[2, "Children"] = 2
        train = select(frame, "regression")
        processor = preprocessing("regression").fit(train)
        medians_before = processor.named_transformers_["numeric"].named_steps["imputer"].statistics_.copy()
        unseen = train.iloc[[0]].copy()
        unseen.loc[:, "Children"] = 999
        unseen.loc[:, "Country"] = "UNSEEN_COUNTRY"
        transformed = processor.transform(unseen)
        self.assertEqual(transformed.shape[0], 1)
        np.testing.assert_array_equal(medians_before, processor.named_transformers_["numeric"].named_steps["imputer"].statistics_)
        self.assertEqual(medians_before[NUMERIC.index("Children")], 1)

    def test_zero_night_feature_is_finite(self):
        frame = fixture().iloc[[0]].copy()
        frame.loc[:, ["StaysInWeekNights", "StaysInWeekendNights"]] = 0
        result = engineering(frame)
        self.assertEqual(result.WeekendShare.iloc[0], 0)
        self.assertTrue(np.isfinite(result.LogTotalNights.iloc[0]))


if __name__ == "__main__":
    unittest.main()
