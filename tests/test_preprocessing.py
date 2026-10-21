import unittest

import pandas as pd

from config.experiment import TARGET_COLUMN
from src.preprocessing import (
    add_cold_stress_target,
    compare_city_timelines,
    compute_pooled_cold_threshold,
    select_period,
)


class TestPreprocessing(unittest.TestCase):
    def test_threshold_uses_only_pooled_warmup_not_future_evaluation(self):
        city_a = pd.DataFrame(
            {
                "timestamp_utc": pd.to_datetime(
                    [
                        "2016-01-01 00:00Z",
                        "2017-12-31 23:00Z",
                        "2018-01-01 00:00Z",
                    ],
                    utc=True,
                ),
                "temperature_2m": [0.0, 10.0, -1000.0],
            }
        )
        city_b = pd.DataFrame(
            {
                "timestamp_utc": pd.to_datetime(
                    [
                        "2016-01-01 00:00Z",
                        "2017-12-31 23:00Z",
                        "2018-01-01 00:00Z",
                    ],
                    utc=True,
                ),
                "temperature_2m": [20.0, 30.0, -1000.0],
            }
        )

        # Pooled warm-up temperatures are [0, 10, 20, 30].
        # Pandas' linear Q0.20 interpolation gives 6.0.
        threshold = compute_pooled_cold_threshold([city_a, city_b])
        self.assertAlmostEqual(threshold, 6.0)

        # Extreme future values must not change that threshold.
        city_a.loc[2, "temperature_2m"] = 9999.0
        city_b.loc[2, "temperature_2m"] = 9999.0
        self.assertAlmostEqual(
            compute_pooled_cold_threshold([city_a, city_b]),
            6.0,
        )

    def test_target_is_one_at_or_below_threshold_and_input_is_not_mutated(self):
        df = pd.DataFrame({"temperature_2m": [8.6, 8.7, 8.8]})
        result = add_cold_stress_target(df, 8.7)

        self.assertNotIn(TARGET_COLUMN, df.columns)
        self.assertEqual(result[TARGET_COLUMN].tolist(), [1, 1, 0])

    def test_select_period_includes_entire_end_date(self):
        df = pd.DataFrame(
            {
                "timestamp_utc": pd.to_datetime(
                    [
                        "2018-01-01 00:00Z",
                        "2018-01-01 23:00Z",
                        "2018-01-02 00:00Z",
                    ],
                    utc=True,
                )
            }
        )
        selected = select_period(df, "2018-01-01", "2018-01-01")
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected["timestamp_utc"].dt.hour.tolist(), [0, 23])

    def test_city_timeline_comparison_requires_exact_same_order(self):
        a = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range(
                    "2020-01-01", periods=3, freq="h", tz="UTC"
                )
            }
        )
        b = a.copy()
        self.assertTrue(compare_city_timelines(a, b))

        b = b.iloc[::-1].reset_index(drop=True)
        self.assertFalse(compare_city_timelines(a, b))


if __name__ == "__main__":
    unittest.main()
