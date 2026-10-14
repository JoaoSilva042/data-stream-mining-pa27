from pathlib import Path

import numpy as np
import pandas as pd

from config.experiment import (
    HISTORICAL_START,
    HISTORICAL_END,
    WARMUP_START,
    WARMUP_END,
    EVALUATION_START,
    EVALUATION_END,
    TARGET_QUANTILE,
    TARGET_COLUMN,
    TARGET_SOURCE_COLUMN,
    LOCAL_TIMEZONE,
    RAW_WEATHER_VARIABLES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_city_data(city_name):
    """Load the processed hourly dataset for one city."""

    MISSING_VALUE_TOKENS = [
        "",
        " ",
        "NA",
        "N/A",
        "n/a",
        "null",
        "NULL",
        "None",
        "none",
        "NaN",
        "nan",
        "?",
        "-",
        "--",
    ]


    city_slug = city_name.lower()

    path = (
        PROCESSED_DIR
        / f"{city_slug}_hourly.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(
        path,
        na_values=MISSING_VALUE_TOKENS,
        keep_default_na=True,
    )

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="raise",
    )

    # Rebuild local timestamps from UTC.
    # This correctly handles daylight-saving changes.
    df["timestamp_local"] = (
        df["timestamp_utc"]
        .dt.tz_convert(LOCAL_TIMEZONE)
    )

    return df


def expected_hourly_index():
    """Return the complete expected UTC timeline."""

    start = pd.Timestamp(
        HISTORICAL_START,
        tz="UTC",
    )

    end = (
        pd.Timestamp(
            HISTORICAL_END,
            tz="UTC",
        )
        + pd.Timedelta(days=1)
        - pd.Timedelta(hours=1)
    )

    return pd.date_range(
        start=start,
        end=end,
        freq="h",
    )


def structural_quality_report(
    df,
    city_name,
):
    """Evaluate the structural integrity of a city stream."""

    required_columns = {
        "timestamp_utc",
        "timestamp_local",
        "city",
        *RAW_WEATHER_VARIABLES,
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{city_name}: missing columns: "
            f"{sorted(missing_columns)}"
        )

    expected_index = expected_hourly_index()

    actual_index = pd.DatetimeIndex(
        df["timestamp_utc"]
    )

    missing_timestamps = (
        expected_index.difference(
            actual_index
        )
    )

    unexpected_timestamps = (
        actual_index.difference(
            expected_index
        )
    )

    numeric_columns = (
        df.select_dtypes(
            include="number"
        ).columns
    )

    infinite_values = int(
        np.isinf(
            df[numeric_columns]
        ).sum().sum()
    )

    non_numeric_weather_columns = [
        column
        for column in RAW_WEATHER_VARIABLES
        if not pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    return {
        "city": city_name,
        "rows": len(df),
        "expected_rows": len(
            expected_index
        ),
        "start": (
            df["timestamp_utc"]
            .iloc[0]
            .isoformat()
        ),
        "end": (
            df["timestamp_utc"]
            .iloc[-1]
            .isoformat()
        ),
        "missing_values": int(
            df.isna().sum().sum()
        ),
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
        "duplicate_timestamps": int(
            df["timestamp_utc"]
            .duplicated()
            .sum()
        ),
        "missing_timestamps": int(
            len(missing_timestamps)
        ),
        "unexpected_timestamps": int(
            len(unexpected_timestamps)
        ),
        "infinite_values": (
            infinite_values
        ),
        "chronological_order": bool(
            df["timestamp_utc"]
            .is_monotonic_increasing
        ),
        "non_numeric_weather_columns":
            non_numeric_weather_columns,
        "city_values": (
            df["city"]
            .unique()
            .tolist()
        ),
    }

def missing_values_report(df):
    """Count missing values for each column."""

    report = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percentage": (
            df.isna().mean() * 100
        ),
    })

    return report

def invalid_values_report(df):
    """
    Detect numeric observations that violate
    basic physical or semantic constraints.

    Non-numeric values are coerced to NaN here
    and are reported separately by
    unexpected_non_numeric_report().
    """

    invalid = {}

    percentage_variables = [
        "relative_humidity_2m",
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
    ]

    for variable in percentage_variables:
        if variable in df.columns:

            values = pd.to_numeric(
                df[variable],
                errors="coerce",
            )

            mask = (
                (values < 0)
                | (values > 100)
            )

            invalid[variable] = int(
                mask.sum()
            )

    non_negative_variables = [
        "precipitation",
        "rain",
        "snowfall",
        "wind_speed_10m",
        "vapour_pressure_deficit",
        "shortwave_radiation",
        "sunshine_duration",
        "soil_moisture_0_to_7cm",
        "soil_moisture_7_to_28cm",
    ]

    for variable in non_negative_variables:
        if variable in df.columns:

            values = pd.to_numeric(
                df[variable],
                errors="coerce",
            )

            mask = values < 0

            invalid[variable] = int(
                mask.sum()
            )

    if "wind_direction_10m" in df.columns:

        values = pd.to_numeric(
            df["wind_direction_10m"],
            errors="coerce",
        )

        mask = (
            (values < 0)
            | (values > 360)
        )

        invalid[
            "wind_direction_10m"
        ] = int(
            mask.sum()
        )

    positive_variables = [
        "pressure_msl",
        "surface_pressure",
    ]

    for variable in positive_variables:
        if variable in df.columns:

            values = pd.to_numeric(
                df[variable],
                errors="coerce",
            )

            mask = values <= 0

            invalid[variable] = int(
                mask.sum()
            )

    return pd.Series(
        invalid,
        name="invalid_count",
    )


def compare_city_timelines(
    city_a,
    city_b,
):
    """Check whether two city streams share the same timestamps."""

    return (
        city_a["timestamp_utc"]
        .equals(
            city_b["timestamp_utc"]
        )
    )

def unexpected_non_numeric_report(df):
    """Find unexpected non-numeric values in weather columns."""

    report = {}

    for column in RAW_WEATHER_VARIABLES:

        numeric_values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid_mask = (
            numeric_values.isna()
            & df[column].notna()
        )

        if invalid_mask.any():
            report[column] = (
                df.loc[
                    invalid_mask,
                    column,
                ]
                .astype(str)
                .value_counts()
                .to_dict()
            )

    return report

def select_period(
    df,
    start_date,
    end_date,
):
    """Select a complete UTC date interval."""

    start = pd.Timestamp(
        start_date,
        tz="UTC",
    )

    end = (
        pd.Timestamp(
            end_date,
            tz="UTC",
        )
        + pd.Timedelta(days=1)
        - pd.Timedelta(hours=1)
    )

    return df[
        df["timestamp_utc"].between(
            start,
            end,
        )
    ].copy()


def get_warmup(df):
    """Return the common warm-up period."""

    return select_period(
        df,
        WARMUP_START,
        WARMUP_END,
    )

def get_evaluation_stream(df):
    """Return the official evaluation period."""

    return select_period(
        df,
        EVALUATION_START,
        EVALUATION_END,
    )


def compute_pooled_cold_threshold(
    city_datasets,
):
    """Compute the T3 cold-stress threshold from pooled warm-up data."""

    warmup_temperatures = []

    for df in city_datasets:

        warmup = get_warmup(df)

        warmup_temperatures.append(
            warmup[TARGET_SOURCE_COLUMN]
        )

    pooled_temperature = pd.concat(
        warmup_temperatures,
        ignore_index=True,
    )

    threshold = pooled_temperature.quantile(
        TARGET_QUANTILE
    )

    return float(threshold)

def add_cold_stress_target(
    df,
    threshold,
):
    """Create the binary T3 cold-stress target."""

    result = df.copy()

    result[TARGET_COLUMN] = (
        result[TARGET_SOURCE_COLUMN]
        <= threshold
    ).astype(int)

    return result