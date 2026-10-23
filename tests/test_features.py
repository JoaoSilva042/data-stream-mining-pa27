import math
import unittest

from config.experiment import BASE_FEATURES
from src.features import (
    InstanceSlidingWindow,
    StreamFeatureBuilder,
    get_base_features,
    get_feature_columns,
)
from tests.helpers import make_stream_row


class TestInstanceSlidingWindow(unittest.TestCase):
    def test_known_sequence_expires_oldest_and_matches_hand_calculation(self):
        window = InstanceSlidingWindow(3)

        self.assertIsNone(window.append(2))
        self.assertIsNone(window.append(4))
        self.assertIsNone(window.append(8))

        self.assertEqual(window.values, [2.0, 4.0, 8.0])
        self.assertEqual(window.count, 3)
        self.assertAlmostEqual(window.mean, 14 / 3)

        # Hand calculation for sample variance of [2, 4, 8].
        mean = 14 / 3
        expected_variance = (
            (2 - mean) ** 2
            + (4 - mean) ** 2
            + (8 - mean) ** 2
        ) / 2
        self.assertAlmostEqual(window.variance, expected_variance)
        self.assertAlmostEqual(window.std, math.sqrt(expected_variance))

        expired = window.append(10)
        self.assertEqual(expired, 2.0)
        self.assertEqual(window.values, [4.0, 8.0, 10.0])
        self.assertAlmostEqual(window.mean, 22 / 3)

    def test_empty_and_singleton_statistics(self):
        window = InstanceSlidingWindow(3)
        self.assertTrue(math.isnan(window.mean))
        self.assertTrue(math.isnan(window.variance))
        self.assertTrue(math.isnan(window.std))

        window.append(5)
        self.assertEqual(window.mean, 5.0)
        self.assertTrue(math.isnan(window.variance))
        self.assertTrue(math.isnan(window.std))

    def test_non_positive_window_size_is_rejected(self):
        with self.assertRaises(ValueError):
            InstanceSlidingWindow(0)
        with self.assertRaises(ValueError):
            InstanceSlidingWindow(-1)


class TestFeatureDefinitions(unittest.TestCase):
    def test_porto_base_features_are_the_frozen_selected_set(self):
        expected = [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "dew_point_2m",
            "soil_temperature_7_to_28cm",
            "soil_moisture_7_to_28cm",
            "sunshine_duration",
            "shortwave_radiation",
            "pressure_msl",
            "cloud_cover",
        ]
        self.assertEqual(get_base_features("Porto"), expected)

    def test_braganca_base_features_are_the_frozen_selected_set(
        self
    ):
        expected = [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "dew_point_2m",
            "soil_temperature_7_to_28cm",
            "soil_moisture_7_to_28cm",
            "shortwave_radiation",
            "sunshine_duration",
            "pressure_msl",
            "cloud_cover",
            "snowfall",
        ]

        self.assertEqual(
            get_base_features(
                "Braganca"
            ),
            expected,
        )

    def test_representation_b_is_exactly_a_plus_required_temporal_context(self):
        a = get_feature_columns("Porto", "A")
        b = get_feature_columns("Porto", "B")

        self.assertEqual(b[: len(a)], a)
        self.assertEqual(
            b[len(a):],
            [
                "temperature_delta_1h",
                "temperature_mean_6h",
                "temperature_std_6h",
                "cold_stress_hours_24h",
            ],
        )

    def test_unknown_city_and_representation_fail_fast(self):
        with self.assertRaises(ValueError):
            get_base_features("Lisboa")
        with self.assertRaises(ValueError):
            get_feature_columns("Porto", "C")
        with self.assertRaises(ValueError):
            StreamFeatureBuilder("Lisboa", "A")
        with self.assertRaises(ValueError):
            StreamFeatureBuilder("Porto", "C")


class TestStreamFeatureBuilder(unittest.TestCase):
    def _prime(self, representation="B", hours=24):
        builder = StreamFeatureBuilder("Porto", representation)
        for i in range(hours):
            row = make_stream_row(i)
            x = builder.transform_one()
            if i < 24:
                self.assertIsNone(x)
            builder.update_one(row, row.cold_stress)
        return builder

    def test_common_initialization_requires_24_past_hours(self):
        for representation in ("A", "B"):
            builder = StreamFeatureBuilder("Porto", representation)
            for i in range(24):
                self.assertFalse(builder.ready)
                row = make_stream_row(i)
                self.assertIsNone(builder.transform_one())
                builder.update_one(row, row.cold_stress)

            self.assertTrue(builder.ready)
            self.assertIsNotNone(builder.transform_one())

    def test_representation_b_at_t24_matches_hand_calculation(self):
        builder = self._prime("B", 24)
        x = builder.transform_one()

        self.assertEqual(x["temperature_2m_lag1"], 23.0)
        self.assertEqual(x["temperature_delta_1h"], 1.0)
        self.assertAlmostEqual(x["temperature_mean_6h"], 20.5)

        # Sample std of [18, 19, 20, 21, 22, 23].
        self.assertAlmostEqual(
            x["temperature_std_6h"],
            math.sqrt(3.5),
        )

        # Alternating labels 0,1,... over 24 observations contain twelve ones.
        self.assertEqual(x["cold_stress_hours_24h"], 12.0)

    def test_transform_is_read_only_until_update(self):
        builder = self._prime("B", 24)

        before_state = (
            builder.observations_seen,
            list(builder.temperature_history),
            list(builder.temperature_window.values),
            list(builder.cold_stress_window.values),
            dict(builder.previous_values),
        )

        x1 = builder.transform_one()
        x2 = builder.transform_one()

        after_state = (
            builder.observations_seen,
            list(builder.temperature_history),
            list(builder.temperature_window.values),
            list(builder.cold_stress_window.values),
            dict(builder.previous_values),
        )

        self.assertEqual(x1, x2)
        self.assertEqual(before_state, after_state)

    def test_current_observation_cannot_affect_current_features(self):
        builder = self._prime("B", 24)

        x_before = builder.transform_one()
        current = make_stream_row(
            24,
            overrides={"temperature_2m": 999.0},
        )

        # Merely having the current row available cannot alter X_t because it has
        # not been passed to update_one yet.
        x_still_before = builder.transform_one()
        self.assertEqual(x_before, x_still_before)
        self.assertEqual(x_before["temperature_2m_lag1"], 23.0)

        builder.update_one(current, current.cold_stress)
        x_next = builder.transform_one()
        self.assertEqual(x_next["temperature_2m_lag1"], 999.0)

    def test_a_and_b_have_identical_values_for_all_shared_features(self):
        a = StreamFeatureBuilder("Porto", "A")
        b = StreamFeatureBuilder("Porto", "B")

        for i in range(24):
            row = make_stream_row(i)
            a.update_one(row, row.cold_stress)
            b.update_one(row, row.cold_stress)

        xa = a.transform_one()
        xb = b.transform_one()

        self.assertEqual(set(xa), set(get_feature_columns("Porto", "A")))
        for key, value in xa.items():
            self.assertEqual(value, xb[key])

    def test_non_binary_numeric_target_is_rejected_instead_of_silently_truncated(self):
        builder = StreamFeatureBuilder("Porto", "A")
        row = make_stream_row(0)

        # A target such as 1.9 is not a binary label and must not silently become 1.
        with self.assertRaises(ValueError):
            builder.update_one(row, 1.9)

    def test_temporal_features_match_independent_oracle_across_many_updates(self):
        builder = StreamFeatureBuilder("Porto", "B")
        temperatures = []
        targets = []

        for i in range(60):
            temperature = float((i * i) % 17 + i / 10)
            target = 1 if i % 5 in (0, 1) else 0
            row = make_stream_row(
                i,
                target=target,
                overrides={"temperature_2m": temperature},
            )

            x = builder.transform_one()

            if i >= 24:
                last_six = temperatures[-6:]
                mean = sum(last_six) / 6
                variance = sum((v - mean) ** 2 for v in last_six) / 5

                self.assertAlmostEqual(x["temperature_2m_lag1"], temperatures[-1])
                self.assertAlmostEqual(
                    x["temperature_delta_1h"],
                    temperatures[-1] - temperatures[-2],
                )
                self.assertAlmostEqual(x["temperature_mean_6h"], mean)
                self.assertAlmostEqual(x["temperature_std_6h"], math.sqrt(variance))
                self.assertEqual(x["cold_stress_hours_24h"], float(sum(targets[-24:])))

            builder.update_one(row, target)
            temperatures.append(temperature)
            targets.append(target)

    def test_state_snapshot_is_read_only_and_contains_next_features(
        self
    ):
        builder = self._prime(
            "B",
            24,
        )

        before = (
            builder.observations_seen,
            list(
                builder
                .temperature_history
            ),
            list(
                builder
                .temperature_window
                .values
            ),
            list(
                builder
                .cold_stress_window
                .values
            ),
        )

        state = (
            builder.state_snapshot()
        )

        after = (
            builder.observations_seen,
            list(
                builder
                .temperature_history
            ),
            list(
                builder
                .temperature_window
                .values
            ),
            list(
                builder
                .cold_stress_window
                .values
            ),
        )

        self.assertEqual(
            before,
            after,
        )

        self.assertTrue(
            state["ready"]
        )

        self.assertEqual(
            state[
                "observations_seen"
            ],
            24,
        )

        self.assertEqual(
            state[
                "temperature_window"
            ],
            [
                18.0,
                19.0,
                20.0,
                21.0,
                22.0,
                23.0,
            ],
        )

        self.assertEqual(
            state[
                "next_features"
            ][
                "temperature_mean_6h"
            ],
            20.5,
        )

        self.assertEqual(
            state[
                "next_features"
            ][
                "cold_stress_hours_24h"
            ],
            12.0,
        )

    def test_builder_memory_is_bounded(self):
        builder = StreamFeatureBuilder("Porto", "B")
        for i in range(500):
            row = make_stream_row(i)
            builder.update_one(row, row.cold_stress)

        self.assertEqual(len(builder.temperature_history), 2)
        self.assertEqual(builder.temperature_window.count, 6)
        self.assertEqual(builder.cold_stress_window.count, 24)


if __name__ == "__main__":
    unittest.main()
