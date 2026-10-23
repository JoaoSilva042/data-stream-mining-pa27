import math
import unittest
from unittest.mock import patch

import pandas as pd

from config.experiment import (
    EVALUATION_WINDOW,
    TARGET_COLUMN,
)
from src.evaluation import (
    OnlineConfusionMatrix,
    StreamingEvaluator,
    build_evaluation_summary,
    evaluate_model,
    warm_up_online,
)
from src.features import StreamFeatureBuilder
from tests.helpers import make_stream_dataframe


class TestOnlineConfusionMatrix(unittest.TestCase):
    def test_metrics_match_manual_confusion_matrix(self):
        cm = OnlineConfusionMatrix((0, 1))

        pairs = [
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, 1),
        ]

        for y_true, y_pred in pairs:
            cm.update(
                y_true,
                y_pred,
            )

        self.assertEqual(
            cm.matrix.tolist(),
            [[1, 1], [1, 2]],
        )

        self.assertEqual(
            cm.total,
            5,
        )

        self.assertEqual(
            cm.correct,
            3,
        )

        self.assertAlmostEqual(
            cm.accuracy(),
            3 / 5,
        )

        self.assertAlmostEqual(
            cm.precision(1),
            2 / 3,
        )

        self.assertAlmostEqual(
            cm.recall(1),
            2 / 3,
        )

        self.assertAlmostEqual(
            cm.f1(1),
            2 / 3,
        )

        self.assertAlmostEqual(
            cm.f1(0),
            1 / 2,
        )

        self.assertAlmostEqual(
            cm.macro_f1(),
            (
                2 / 3
                + 1 / 2
            ) / 2,
        )

        cm.remove(
            0,
            1,
        )

        self.assertEqual(
            cm.matrix.tolist(),
            [[1, 0], [1, 2]],
        )


class TestStreamingEvaluator(unittest.TestCase):
    def test_recent_window_contains_only_last_predictions(
        self
    ):
        evaluator = (
            StreamingEvaluator(
                (0, 1),
                window_size=2,
            )
        )

        evaluator.observe(
            0,
            0,
        )

        evaluator.observe(
            1,
            0,
        )

        snap = evaluator.observe(
            1,
            1,
        )

        self.assertEqual(
            evaluator
            .cumulative
            .matrix
            .tolist(),
            [[1, 0], [1, 1]],
        )

        self.assertEqual(
            evaluator
            .recent
            .matrix
            .tolist(),
            [[0, 0], [1, 1]],
        )

        self.assertAlmostEqual(
            snap[
                "cumulative_accuracy"
            ],
            2 / 3,
        )

        self.assertAlmostEqual(
            snap[
                "window_accuracy"
            ],
            1 / 2,
        )

        self.assertAlmostEqual(
            snap[
                "cumulative_precision_positive"
            ],
            1.0,
        )

        self.assertAlmostEqual(
            snap[
                "cumulative_recall_positive"
            ],
            1 / 2,
        )

        self.assertAlmostEqual(
            snap[
                "window_precision_positive"
            ],
            1.0,
        )

        self.assertAlmostEqual(
            snap[
                "window_recall_positive"
            ],
            1 / 2,
        )

    def test_none_prediction_counts_as_cold_start_but_not_scored_prediction(
        self
    ):
        evaluator = (
            StreamingEvaluator(
                (0, 1),
                window_size=2,
            )
        )

        snap = evaluator.observe(
            1,
            None,
        )

        self.assertEqual(
            evaluator.observations_seen,
            1,
        )

        self.assertEqual(
            evaluator.predictions_evaluated,
            0,
        )

        self.assertEqual(
            evaluator.cold_starts,
            1,
        )

        self.assertTrue(
            math.isnan(
                snap[
                    "cumulative_accuracy"
                ]
            )
        )

    def test_non_positive_recent_window_is_rejected(
        self
    ):
        with self.assertRaises(
            ValueError
        ):
            StreamingEvaluator(
                (0, 1),
                window_size=0,
            )

    def test_positive_metrics_are_nan_until_defined(
        self
    ):
        evaluator = (
            StreamingEvaluator(
                (0, 1),
                window_size=2,
            )
        )

        snap = evaluator.observe(
            0,
            0,
        )

        self.assertTrue(
            math.isnan(
                snap[
                    "cumulative_precision_positive"
                ]
            )
        )

        self.assertTrue(
            math.isnan(
                snap[
                    "cumulative_recall_positive"
                ]
            )
        )

        self.assertTrue(
            math.isnan(
                snap[
                    "window_precision_positive"
                ]
            )
        )

        self.assertTrue(
            math.isnan(
                snap[
                    "window_recall_positive"
                ]
            )
        )

    def test_default_recent_window_is_one_week(
        self
    ):
        evaluator = (
            StreamingEvaluator(
                (0, 1)
            )
        )

        self.assertEqual(
            evaluator.window_size,
            EVALUATION_WINDOW,
        )

        self.assertEqual(
            EVALUATION_WINDOW,
            168,
        )


class EventLogModel:
    def __init__(
        self,
        events,
        prediction=0,
    ):
        self.events = events
        self.prediction = prediction
        self.learned = []
        self.predicted_x = []

    def predict_one(
        self,
        x,
    ):
        self.events.append(
            "predict"
        )

        self.predicted_x.append(
            dict(x)
        )

        return self.prediction

    def learn_one(
        self,
        x,
        y,
    ):
        self.events.append(
            "learn"
        )

        self.learned.append(
            (
                dict(x),
                int(y),
            )
        )

        return self


class EventLogBuilder:
    def __init__(
        self,
        events,
        transformed=None,
    ):
        self.events = events

        self.transformed = (
            {"feature": 1.0}
            if transformed is None
            else transformed
        )

    def transform_one(self):
        self.events.append(
            "transform"
        )

        return self.transformed

    def update_one(
        self,
        row,
        y,
    ):
        self.events.append(
            "update"
        )


class EventLogEvaluator:
    def __init__(
        self,
        labels,
        window_size,
    ):
        self.events = EVENT_LOG
        self.observations_seen = 0
        self.predictions_evaluated = 0
        self.cold_starts = 0

    def observe(
        self,
        y_true,
        y_pred,
    ):
        self.events.append(
            "observe"
        )

        self.observations_seen += 1

        if y_pred is None:
            self.cold_starts += 1
        else:
            self.predictions_evaluated += 1

        return {
            "cumulative_accuracy":
                1.0,

            "window_accuracy":
                1.0,

            "cumulative_macro_f1":
                1.0,

            "window_macro_f1":
                1.0,

            "cumulative_precision_positive":
                1.0,

            "window_precision_positive":
                1.0,

            "cumulative_recall_positive":
                1.0,

            "window_recall_positive":
                1.0,
        }


EVENT_LOG = []


class TestEvaluationOrder(
    unittest.TestCase
):
    def setUp(self):
        EVENT_LOG.clear()

    def test_prequential_order_is_transform_predict_evaluate_learn_update(
        self
    ):
        data = pd.DataFrame(
            {
                "timestamp_utc":
                    pd.date_range(
                        "2020-01-01",
                        periods=2,
                        freq="h",
                        tz="UTC",
                    ),

                TARGET_COLUMN:
                    [0, 1],
            }
        )

        builder = (
            EventLogBuilder(
                EVENT_LOG
            )
        )

        model = (
            EventLogModel(
                EVENT_LOG
            )
        )

        with patch(
            "src.evaluation.StreamingEvaluator",
            EventLogEvaluator,
        ):
            evaluate_model(
                model=model,
                data=data,
                feature_builder=
                    builder,
                target_column=
                    TARGET_COLUMN,
                learn=True,
                window_size=2,
            )

        self.assertEqual(
            EVENT_LOG,
            [
                "transform",
                "predict",
                "observe",
                "learn",
                "update",

                "transform",
                "predict",
                "observe",
                "learn",
                "update",
            ],
        )

    def test_feature_state_still_advances_when_model_learning_is_disabled(
        self
    ):
        data = pd.DataFrame(
            {
                "timestamp_utc":
                    pd.date_range(
                        "2020-01-01",
                        periods=1,
                        freq="h",
                        tz="UTC",
                    ),

                TARGET_COLUMN:
                    [1],
            }
        )

        builder = (
            EventLogBuilder(
                EVENT_LOG
            )
        )

        model = (
            EventLogModel(
                EVENT_LOG
            )
        )

        with patch(
            "src.evaluation.StreamingEvaluator",
            EventLogEvaluator,
        ):
            evaluate_model(
                model=model,
                data=data,
                feature_builder=
                    builder,
                target_column=
                    TARGET_COLUMN,
                learn=False,
                window_size=2,
            )

        self.assertEqual(
            EVENT_LOG,
            [
                "transform",
                "predict",
                "observe",
                "update",
            ],
        )

        self.assertEqual(
            model.learned,
            [],
        )

    def test_warmup_does_not_predict_and_updates_feature_state_after_learning(
        self
    ):
        class WarmupBuilder:
            def __init__(self):
                self.calls = 0

            def transform_one(self):
                EVENT_LOG.append(
                    "transform"
                )

                self.calls += 1

                return (
                    None
                    if self.calls == 1
                    else {
                        "feature": 1.0
                    }
                )

            def update_one(
                self,
                row,
                y,
            ):
                EVENT_LOG.append(
                    "update"
                )

        data = pd.DataFrame(
            {
                TARGET_COLUMN:
                    [0, 1]
            }
        )

        builder = WarmupBuilder()

        model = (
            EventLogModel(
                EVENT_LOG
            )
        )

        warm_up_online(
            model,
            data,
            builder,
            TARGET_COLUMN,
        )

        self.assertEqual(
            EVENT_LOG,
            [
                "transform",
                "update",
                "transform",
                "learn",
                "update",
            ],
        )

        self.assertEqual(
            len(model.learned),
            1,
        )

        self.assertEqual(
            model.predicted_x,
            [],
        )

    def test_warmup_state_is_built_only_after_complete_warmup(
        self
    ):
        warmup = (
            make_stream_dataframe(
                30,
                city="Porto",
            )
        )

        builder = (
            StreamFeatureBuilder(
                "Porto",
                "B",
            )
        )

        model = (
            EventLogModel(
                [],
                prediction=0,
            )
        )

        (
            model,
            builder,
            state,
        ) = warm_up_online(
            model,
            warmup,
            builder,
            TARGET_COLUMN,
            return_state=True,
        )

        self.assertEqual(
            state[
                "warmup_rows"
            ],
            30,
        )

        self.assertEqual(
            state[
                "model_updates"
            ],
            6,
        )

        self.assertEqual(
            state[
                "feature_builder"
            ][
                "observations_seen"
            ],
            30,
        )

        self.assertTrue(
            state[
                "feature_builder"
            ][
                "ready"
            ]
        )

        self.assertEqual(
            len(model.learned),
            6,
        )

    def test_feature_state_continues_from_warmup_into_evaluation(
        self
    ):
        warmup = (
            make_stream_dataframe(
                24,
                city="Porto",
            )
        )

        evaluation = (
            make_stream_dataframe(
                1,
                city="Porto",
                start="2020-01-02",
            )
        )

        # The helper resets feature values to
        # zero for its first row. This is useful:
        # the first evaluation feature vector
        # must still come entirely from warm-up.
        builder = (
            StreamFeatureBuilder(
                "Porto",
                "B",
            )
        )

        model = (
            EventLogModel(
                [],
                prediction=0,
            )
        )

        (
            model,
            builder,
        ) = warm_up_online(
            model,
            warmup,
            builder,
            TARGET_COLUMN,
        )

        (
            evaluator,
            history,
            _,
            _,
        ) = evaluate_model(
            model,
            evaluation,
            builder,
            TARGET_COLUMN,
        )

        self.assertEqual(
            len(history),
            1,
        )

        self.assertEqual(
            model.predicted_x[0][
                "temperature_2m_lag1"
            ],
            23.0,
        )

        self.assertEqual(
            model.predicted_x[0][
                "temperature_mean_6h"
            ],
            20.5,
        )

        self.assertEqual(
            history.loc[
                0,
                "temperature_2m_lag1",
            ],
            23.0,
        )

        self.assertEqual(
            history.loc[
                0,
                "temperature_mean_6h",
            ],
            20.5,
        )

        self.assertIn(
            "window_recall_positive",
            history.columns,
        )

        self.assertIn(
            "cumulative_precision_positive",
            history.columns,
        )

        self.assertEqual(
            evaluator.observations_seen,
            1,
        )


class TestEvaluationSummary(
    unittest.TestCase
):
    def test_summary_matches_manual_counts_and_costs(
        self
    ):
        evaluator = (
            StreamingEvaluator(
                (0, 1),
                window_size=10,
            )
        )

        for y_true, y_pred in [
            (0, 0),
            (1, 0),
            (1, 1),
        ]:
            evaluator.observe(
                y_true,
                y_pred,
            )

        summary = (
            build_evaluation_summary(
                evaluator,
                predict_seconds=0.003,
                update_seconds=0.006,
            )
        )

        self.assertEqual(
            summary[
                "observations_processed"
            ],
            3,
        )

        self.assertEqual(
            summary[
                "events"
            ],
            3,
        )

        self.assertEqual(
            (
                summary["tn"],
                summary["fp"],
                summary["fn"],
                summary["tp"],
            ),
            (1, 0, 1, 1),
        )

        self.assertAlmostEqual(
            summary[
                "accuracy"
            ],
            2 / 3,
        )

        self.assertAlmostEqual(
            summary[
                "precision_positive"
            ],
            1.0,
        )

        self.assertAlmostEqual(
            summary[
                "recall_positive"
            ],
            1 / 2,
        )

        self.assertAlmostEqual(
            summary[
                "prediction_ms_per_event"
            ],
            1.0,
        )

        self.assertAlmostEqual(
            summary[
                "update_ms_per_event"
            ],
            2.0,
        )


if __name__ == "__main__":
    unittest.main()
