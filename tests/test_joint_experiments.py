
import unittest
import pandas as pd

from config.experiment import BASE_FEATURES, TARGET_COLUMN
from src.run_joint_experiments import (
    aligned_rows,
    run_joint_experiment,
)


def make_data(start, n, temperature, target):
    rows = []

    for i in range(n):
        row = {
            feature: float(i)
            for feature in BASE_FEATURES["Porto"]
        }

        row["temperature_2m"] = temperature + i
        row["timestamp_utc"] = (
            pd.Timestamp(start, tz="UTC")
            + pd.Timedelta(hours=i)
        )
        row[TARGET_COLUMN] = target
        rows.append(row)

    return pd.DataFrame(rows)


class TestJointExperiments(unittest.TestCase):

    def setUp(self):
        self.warmup = {
            "Porto": make_data(
                "2020-01-01", 25, 10, 0
            ),
            "Braganca": make_data(
                "2020-01-01", 25, -20, 1
            ),
        }

        self.evaluation = {
            "Porto": make_data(
                "2020-01-02 01:00", 2, 35, 0
            ),
            "Braganca": make_data(
                "2020-01-02 01:00", 2, 5, 1
            ),
        }

    def test_shared_model_and_prediction_order(self):

        created = []

        class DummyModel:
            def __init__(self):
                self.updates = 0
                self.prediction_states = []

            def predict_one(self, x):
                self.prediction_states.append(self.updates)
                return 0

            def learn_one(self, x, y):
                self.updates += 1
                return self

        def factory():
            model = DummyModel()
            created.append(model)
            return model

        summaries, histories, state, model = (
            run_joint_experiment(
                representation="B",
                model_name="dummy",
                warmup_data=self.warmup,
                evaluation_data=self.evaluation,
                model_factories={"dummy": factory},
                threshold=8.7,
            )
        )

        # Foi criado apenas um classificador.
        self.assertEqual(len(created), 1)
        self.assertIs(model, created[0])

        # Warm-up: 1 update por cidade.
        self.assertEqual(state["model_updates"], 2)

        # As duas previsoes de cada hora precedem
        # as duas atualizacoes dessa hora.
        self.assertEqual(
            model.prediction_states,
            [2, 2, 4, 4],
        )

        self.assertEqual(model.updates, 6)

        # Cada cidade tem o seu historico.
        self.assertEqual(len(histories["Porto"]), 2)
        self.assertEqual(len(histories["Braganca"]), 2)

        # As medias moveis nao se misturam.
        self.assertAlmostEqual(
            histories["Porto"].iloc[0]["temperature_mean_6h"],
            31.5,
        )

        self.assertAlmostEqual(
            histories["Braganca"].iloc[0]["temperature_mean_6h"],
            1.5,
        )

        self.assertEqual(
            summaries["Porto"]["protocol"],
            "joint",
        )

    def test_timestamps_must_be_aligned(self):
        bad = {
            city: data.copy()
            for city, data in self.warmup.items()
        }

        bad["Braganca"].loc[4, "timestamp_utc"] += (
            pd.Timedelta(hours=1)
        )

        with self.assertRaises(ValueError):
            list(aligned_rows(bad))

    def test_stream_lengths_must_match(self):
        bad = {
            city: data.copy()
            for city, data in self.warmup.items()
        }

        bad["Braganca"] = bad["Braganca"].iloc[:-1]

        with self.assertRaises(ValueError):
            list(aligned_rows(bad))


if __name__ == "__main__":
    unittest.main()
