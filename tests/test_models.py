import unittest

from config.experiment import MODEL_NAMES
from river import preprocessing

try:
    from src.models import make_model_factories
    RIVER_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent
    make_model_factories = None
    RIVER_IMPORT_ERROR = exc


@unittest.skipIf(
    RIVER_IMPORT_ERROR is not None,
    f"River unavailable in this environment: {RIVER_IMPORT_ERROR}",
)
class TestModelFactories(unittest.TestCase):
    def test_all_mandatory_models_have_factories(self):
        factories = make_model_factories()
        self.assertEqual(set(factories), set(MODEL_NAMES))

    def test_every_factory_returns_a_fresh_independent_instance(self):
        factories = make_model_factories()
        for name, factory in factories.items():
            with self.subTest(model=name):
                self.assertIsNot(factory(), factory())

    def test_every_model_supports_online_predict_and_learn_protocol(self):
        factories = make_model_factories()
        events = [
            ({"x0": 0.0, "x1": 0.0}, 0),
            ({"x0": 1.0, "x1": 1.0}, 1),
            ({"x0": 0.2, "x1": 0.1}, 0),
            ({"x0": 0.9, "x1": 0.8}, 1),
        ]

        for name, factory in factories.items():
            with self.subTest(model=name):
                model = factory()
                for x, y in events:
                    model.predict_one(x)
                    returned = model.learn_one(x, y)
                    # River learn_one commonly returns self; protocol support is the key check.
                    self.assertTrue(hasattr(model, "predict_one"))
                    self.assertTrue(hasattr(model, "learn_one"))

    def test_seeded_ensembles_are_reproducible_on_same_stream(self):
        factories = make_model_factories()
        events = [
            ({"x0": float(i % 3), "x1": float((i * 2) % 5)}, i % 2)
            for i in range(30)
        ]

        for name in ("online_bagging", "adaptive_random_forest"):
            with self.subTest(model=name):
                m1 = factories[name]()
                m2 = factories[name]()
                preds1 = []
                preds2 = []
                for x, y in events:
                    preds1.append(m1.predict_one(x))
                    preds2.append(m2.predict_one(x))
                    m1.learn_one(x, y)
                    m2.learn_one(x, y)
                self.assertEqual(preds1, preds2)

    def test_logistic_scaler_updates_only_during_learning(self):
        """
        The online scaler must not learn from
        the current event before prediction.
        """

        model = make_model_factories()[
            "logistic_regression"
        ]()

        scalers = [
            step
            for step in model.steps.values()
            if isinstance(
                step,
                preprocessing.StandardScaler,
            )
        ]

        self.assertEqual(
            len(scalers),
            1,
        )

        scaler = scalers[0]

        x1 = {
            "temperature_2m_lag1": 10.0,
            "pressure_msl_lag1": 1000.0,
        }

        x2 = {
            "temperature_2m_lag1": 20.0,
            "pressure_msl_lag1": 1020.0,
        }

        # First observation is learned.
        model.learn_one(
            x1,
            0,
        )

        counts_before_prediction = dict(
            scaler.counts
        )

        means_before_prediction = dict(
            scaler.means
        )

        # Prediction must be read-only.
        model.predict_one(
            x2
        )

        self.assertEqual(
            dict(scaler.counts),
            counts_before_prediction,
        )

        self.assertEqual(
            dict(scaler.means),
            means_before_prediction,
        )

        # Only learning may update the scaler.
        model.learn_one(
            x2,
            1,
        )

        self.assertEqual(
            scaler.counts[
                "temperature_2m_lag1"
            ],
            2,
        )

        self.assertEqual(
            scaler.counts[
                "pressure_msl_lag1"
            ],
            2,
        )


if __name__ == "__main__":
    unittest.main()
