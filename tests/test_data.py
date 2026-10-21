import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from config.experiment import RAW_WEATHER_VARIABLES
from src.data import (
    basic_stream_quality_report,
    fetch_json,
    save_json,
    validate_weather_payload,
    weather_json_to_dataframe,
)


def valid_payload(times=(0, 3600)):
    hourly = {"time": list(times)}
    for i, variable in enumerate(RAW_WEATHER_VARIABLES):
        hourly[variable] = [float(i), float(i + 1)]
    return {"hourly": hourly}


class TestDataAcquisitionHelpers(unittest.TestCase):
    def test_weather_payload_requires_all_requested_variables(self):
        payload = valid_payload()
        payload["hourly"].pop(RAW_WEATHER_VARIABLES[0])
        with self.assertRaises(ValueError):
            validate_weather_payload(payload, "Porto")

    def test_weather_payload_requires_equal_lengths(self):
        payload = valid_payload()
        payload["hourly"][RAW_WEATHER_VARIABLES[0]] = [1.0]
        with self.assertRaises(ValueError):
            validate_weather_payload(payload, "Porto")

    def test_json_to_dataframe_builds_chronological_utc_and_local_timestamps(self):
        df = weather_json_to_dataframe(valid_payload(), "Porto")
        self.assertEqual(len(df), 2)
        self.assertTrue(df["timestamp_utc"].is_monotonic_increasing)
        self.assertEqual(df["city"].unique().tolist(), ["Porto"])
        self.assertIn("timestamp_local", df.columns)
        self.assertNotIn("time", df.columns)

    def test_json_to_dataframe_rejects_out_of_order_timestamps(self):
        with self.assertRaises(ValueError):
            weather_json_to_dataframe(valid_payload(times=(3600, 0)), "Porto")

    def test_basic_quality_report_detects_duplicate_timestamp_and_missing_value(self):
        df = weather_json_to_dataframe(valid_payload(), "Porto")
        duplicate = df.iloc[[1]].copy()
        duplicate.loc[:, RAW_WEATHER_VARIABLES[0]] = float("nan")
        combined = pd.concat([df, duplicate], ignore_index=True)

        report = basic_stream_quality_report(combined)
        self.assertEqual(report["duplicate_timestamps"], 1)
        self.assertEqual(report["missing_by_variable"][RAW_WEATHER_VARIABLES[0]], 1)

    def test_fetch_json_retries_429_then_returns_success(self):
        rate_limited = Mock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "1"}
        rate_limited.raise_for_status = Mock()

        success = Mock()
        success.status_code = 200
        success.headers = {}
        success.raise_for_status = Mock()
        success.json.return_value = {"ok": True}

        with patch("src.data.requests.get", side_effect=[rate_limited, success]) as get_mock, patch("src.data.time.sleep") as sleep_mock:
            data, response = fetch_json("https://example.invalid", max_retries=1)

        self.assertEqual(data, {"ok": True})
        self.assertIs(response, success)
        self.assertEqual(get_mock.call_count, 2)
        sleep_mock.assert_called_once_with(1)

    def test_save_json_creates_parent_and_valid_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "sample.json"
            save_json({"a": 1}, path)
            self.assertTrue(path.exists())
            self.assertIn('"a": 1', path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
