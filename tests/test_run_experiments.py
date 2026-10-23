import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from config.experiment import BASE_FEATURES, MODEL_NAMES, TARGET_COLUMN

try:
    import src.run_experiments as runner
    RUNNER_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent
    runner = None
    RUNNER_IMPORT_ERROR = exc


@unittest.skipIf(
    RUNNER_IMPORT_ERROR is not None,
    f"Runner dependencies unavailable in this environment: {RUNNER_IMPORT_ERROR}",
)
class TestRunExperiments(unittest.TestCase):
    def test_run_all_orchestrates_exactly_20_city_representation_model_combinations(self):
        dummy = pd.DataFrame({"dummy": [1]})
        city_streams = {
            "Porto": {"warmup": dummy, "evaluation": dummy, "full": dummy},
            "Braganca": {"warmup": dummy, "evaluation": dummy, "full": dummy},
        }
        factories = {name: (lambda: object()) for name in MODEL_NAMES}

        calls = []

        def fake_single(**kwargs):
            calls.append(
                (
                    kwargs["city_name"],
                    kwargs["representation"],
                    kwargs["model_name"],
                )
            )
            summary = {
                "city": kwargs["city_name"],
                "representation": kwargs["representation"],
                "model": kwargs["model_name"],
                "macro_f1": 0.5,
                "recall_positive": 0.5,
            }
            return (
                summary,
                pd.DataFrame(),
                {},
                object(),
            )

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(runner, "prepare_experiment_data", return_value=(city_streams, 8.7)), \
             patch.object(runner, "make_model_factories", return_value=factories), \
             patch.object(runner, "run_single_experiment", side_effect=fake_single), \
             patch.object(runner, "SUMMARY_DIR", Path(tmp)):
            result = runner.run_all_experiments(save_histories=False)

        self.assertEqual(len(calls), 20)
        self.assertEqual(len(set(calls)), 20)
        self.assertEqual(len(result), 20)

    def test_single_experiment_uses_fresh_state_and_returns_consistent_history(self):
        def make_df(start, n, offset=0):
            rows = []
            start_ts = pd.Timestamp(start, tz="UTC")
            for i in range(n):
                k = i + offset
                row = {
                    feature: float(k + j / 100)
                    for j, feature in enumerate(BASE_FEATURES["Porto"])
                }
                row["temperature_2m"] = float(k)
                row["timestamp_utc"] = start_ts + pd.Timedelta(hours=i)
                row[TARGET_COLUMN] = k % 2
                rows.append(row)
            return pd.DataFrame(rows)

        warmup = make_df("2020-01-01", 30, offset=0)
        evaluation = make_df("2020-01-02 06:00", 3, offset=30)

        created_models = []

        class DummyModel:
            def __init__(self):
                self.last = 0
                self.learn_calls = 0

            def predict_one(self, x):
                return self.last

            def learn_one(self, x, y):
                self.learn_calls += 1
                self.last = int(y)
                return self

        def factory():
            model = DummyModel()
            created_models.append(model)
            return model

        (
            summary,
            history,
            warmup_state,
            model,
        ) = runner.run_single_experiment(
            city_name="Porto",
            representation="B",
            model_name="dummy",
            warmup_data=warmup,
            evaluation_data=evaluation,
            model_factories={"dummy": factory},
            threshold=8.7,
        )

        self.assertEqual(len(created_models), 1)
        self.assertIs(model, created_models[0])
        self.assertEqual(len(history), 3)
        self.assertEqual(summary["observations_processed"], 3)
        self.assertEqual(summary["events"], 3)
        self.assertEqual(summary["warmup_rows"], 30)
        self.assertEqual(summary["evaluation_rows"], 3)
        self.assertEqual(
            summary[
                "warmup_model_updates"
            ],
            6,
        )

        self.assertTrue(
            summary[
                "warmup_feature_builder_ready"
            ]
        )

        self.assertEqual(
            warmup_state[
                "warmup_rows"
            ],
            30,
        )

        self.assertTrue(
            warmup_state[
                "feature_builder"
            ][
                "ready"
            ]
        )
        self.assertEqual(history["city"].unique().tolist(), ["Porto"])
        self.assertEqual(history["representation"].unique().tolist(), ["B"])
        self.assertEqual(history["model"].unique().tolist(), ["dummy"])


if __name__ == "__main__":
    unittest.main()
