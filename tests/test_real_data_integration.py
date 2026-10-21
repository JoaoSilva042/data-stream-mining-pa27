import unittest

from config.experiment import TARGET_COLUMN
from src.evaluation import build_evaluation_summary, evaluate_model, warm_up_online
from src.features import StreamFeatureBuilder
from src.preprocessing import (
    add_cold_stress_target,
    compare_city_timelines,
    compute_pooled_cold_threshold,
    get_evaluation_stream,
    get_warmup,
    load_city_data,
    structural_quality_report,
    invalid_values_report,
    unexpected_non_numeric_report,
)


class LastLabelModel:
    """Minimal deterministic online model used only to validate pipeline mechanics."""

    def __init__(self):
        self.last = None
        self.learn_calls = 0
        self.predict_calls = 0

    def predict_one(self, x):
        self.predict_calls += 1
        return self.last

    def learn_one(self, x, y):
        self.learn_calls += 1
        self.last = int(y)
        return self


class TestRealPA27Data(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.porto_raw = load_city_data("Porto")
        cls.braganca_raw = load_city_data("Braganca")
        cls.threshold = compute_pooled_cold_threshold(
            [cls.porto_raw, cls.braganca_raw]
        )
        cls.porto = add_cold_stress_target(cls.porto_raw, cls.threshold)
        cls.braganca = add_cold_stress_target(cls.braganca_raw, cls.threshold)

    def test_fixed_datasets_have_expected_common_chronology(self):
        self.assertEqual(len(self.porto), 87672)
        self.assertEqual(len(self.braganca), 87672)
        self.assertTrue(compare_city_timelines(self.porto, self.braganca))
        self.assertTrue(self.porto["timestamp_utc"].is_monotonic_increasing)
        self.assertTrue(self.braganca["timestamp_utc"].is_monotonic_increasing)

    def test_real_streams_are_structurally_clean_before_modelling(self):
        for city, df in (("Porto", self.porto_raw), ("Braganca", self.braganca_raw)):
            with self.subTest(city=city):
                quality = structural_quality_report(df, city)
                self.assertEqual(quality["missing_values"], 0)
                self.assertEqual(quality["duplicate_timestamps"], 0)
                self.assertEqual(quality["missing_timestamps"], 0)
                self.assertEqual(quality["unexpected_timestamps"], 0)
                self.assertEqual(quality["infinite_values"], 0)
                self.assertTrue(quality["chronological_order"])
                self.assertEqual(quality["non_numeric_weather_columns"], [])
                self.assertEqual(int(invalid_values_report(df).sum()), 0)
                self.assertEqual(unexpected_non_numeric_report(df), {})

    def test_official_period_sizes_and_frozen_threshold(self):
        self.assertAlmostEqual(self.threshold, 8.7)
        self.assertEqual(len(get_warmup(self.porto)), 17544)
        self.assertEqual(len(get_evaluation_stream(self.porto)), 70128)

    def test_first_2018_b_features_match_independent_dataframe_calculation(self):
        warmup = get_warmup(self.porto)
        evaluation = get_evaluation_stream(self.porto)

        builder = StreamFeatureBuilder("Porto", "B")
        for row in warmup.itertuples(index=False):
            y = int(getattr(row, TARGET_COLUMN))
            builder.update_one(row, y)

        x = builder.transform_one()
        first_eval_index = evaluation.index[0]
        pos = self.porto.index.get_loc(first_eval_index)

        expected_lag1 = self.porto["temperature_2m"].iloc[pos - 1]
        expected_delta = (
            self.porto["temperature_2m"].iloc[pos - 1]
            - self.porto["temperature_2m"].iloc[pos - 2]
        )
        expected_mean = self.porto["temperature_2m"].iloc[pos - 6:pos].mean()
        expected_std = self.porto["temperature_2m"].iloc[pos - 6:pos].std(ddof=1)
        expected_cold_hours = self.porto[TARGET_COLUMN].iloc[pos - 24:pos].sum()

        self.assertAlmostEqual(x["temperature_2m_lag1"], expected_lag1)
        self.assertAlmostEqual(x["temperature_delta_1h"], expected_delta)
        self.assertAlmostEqual(x["temperature_mean_6h"], expected_mean)
        self.assertAlmostEqual(x["temperature_std_6h"], expected_std, places=9)
        self.assertAlmostEqual(x["cold_stress_hours_24h"], expected_cold_hours)

    def test_full_porto_b_pipeline_processes_every_evaluation_event_once(self):
        warmup = get_warmup(self.porto)
        evaluation = get_evaluation_stream(self.porto)

        builder = StreamFeatureBuilder("Porto", "B")
        model = LastLabelModel()

        model, builder = warm_up_online(
            model,
            warmup,
            builder,
            TARGET_COLUMN,
        )
        evaluator, history, pred_s, upd_s = evaluate_model(
            model,
            evaluation,
            builder,
            TARGET_COLUMN,
        )
        summary = build_evaluation_summary(evaluator, pred_s, upd_s)

        self.assertEqual(model.learn_calls, 87648)  # 87672 total - 24 history-only rows
        self.assertEqual(model.predict_calls, 70128)
        self.assertEqual(len(history), 70128)
        self.assertEqual(summary["observations_processed"], 70128)
        self.assertEqual(summary["events"], 70128)
        self.assertEqual(builder.observations_seen, 87672)
        self.assertEqual(builder.temperature_window.count, 6)
        self.assertEqual(builder.cold_stress_window.count, 24)


if __name__ == "__main__":
    unittest.main()
