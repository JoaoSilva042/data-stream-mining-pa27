from types import SimpleNamespace

import pandas as pd

from config.experiment import BASE_FEATURES, TARGET_COLUMN


def make_stream_row(i, city="Porto", *, target=None, overrides=None):
    """Create one deterministic synthetic raw observation for stream tests."""
    values = {
        feature: float(i + (j + 1) / 100)
        for j, feature in enumerate(BASE_FEATURES[city])
    }
    # Keep temperature especially simple so lag/window expectations are hand-checkable.
    values["temperature_2m"] = float(i)
    values["timestamp_utc"] = (
        pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(hours=i)
    )
    values[TARGET_COLUMN] = int(i % 2 if target is None else target)

    if overrides:
        values.update(overrides)

    return SimpleNamespace(**values)


def make_stream_dataframe(n, city="Porto", start="2020-01-01"):
    """Create n chronological synthetic observations as a DataFrame."""
    rows = []
    start_ts = pd.Timestamp(start, tz="UTC")

    for i in range(n):
        row = {
            feature: float(i + (j + 1) / 100)
            for j, feature in enumerate(BASE_FEATURES[city])
        }
        row["temperature_2m"] = float(i)
        row["timestamp_utc"] = start_ts + pd.Timedelta(hours=i)
        row[TARGET_COLUMN] = int(i % 2)
        rows.append(row)

    return pd.DataFrame(rows)
