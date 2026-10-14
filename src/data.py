import json
from pathlib import Path

import pandas as pd
import requests
import time

from config.experiment import (
    CITIES,
    HISTORICAL_START,
    HISTORICAL_END,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_GEOCODING_URL,
    OPEN_METEO_MODEL,
    RAW_WEATHER_VARIABLES,
    API_TIMEZONE,
    LOCAL_TIMEZONE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def fetch_json(
    url,
    params=None,
    headers=None,
    timeout=60,
    max_retries=3,
):
    """Fetch JSON from an HTTP endpoint."""

    default_headers = {
        "User-Agent":
            "NOVA-IMS-Data-Stream-Mining-PA27/1.0"
    }

    if headers:
        default_headers.update(headers)

    for attempt in range(max_retries + 1):

        response = requests.get(
            url,
            params=params,
            headers=default_headers,
            timeout=timeout,
        )

        if response.status_code == 429:

            if attempt == max_retries:
                response.raise_for_status()

            retry_after = response.headers.get(
                "Retry-After"
            )

            wait_seconds = (
                int(retry_after)
                if retry_after
                else 65
            )

            print(
                f"Rate limit reached. "
                f"Waiting {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)
            continue

        response.raise_for_status()

        return response.json(), response

def geocode_city(city_name):
    """Resolve a Portuguese city using Open-Meteo Geocoding API."""

    city_config = CITIES[city_name]

    params = {
        "name": city_config["query"],
        "count": 5,
        "language": "en",
        "format": "json",
        "countryCode": city_config["country_code"],
    }

    data, response = fetch_json(
        OPEN_METEO_GEOCODING_URL,
        params=params,
    )

    results = data.get("results", [])

    if not results:
        raise ValueError(f"No geocoding results found for {city_name}")

    location = results[0]

    return location, data, response

def save_json(data, path):
    """Save a Python object as formatted JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def fetch_historical_weather( latitude, longitude):
    """Fetch hourly historical weather data from Open-Meteo."""

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": HISTORICAL_START,
        "end_date": HISTORICAL_END,
        "hourly": ",".join(RAW_WEATHER_VARIABLES),
        "timezone": API_TIMEZONE,
        "timeformat": "unixtime",
        "models": OPEN_METEO_MODEL,
    }

    data, response = fetch_json(
        OPEN_METEO_ARCHIVE_URL,
        params=params,
    )

    return data, response


def validate_weather_payload(data, city_name):
    """Validate the structure returned by Open-Meteo."""

    hourly = data.get("hourly")

    if not isinstance(hourly, dict):
        raise ValueError(
            f"No valid hourly data found for {city_name}"
        )

    if "time" not in hourly:
        raise ValueError(
            f"No time variable found for {city_name}"
        )

    returned_variables = set(hourly.keys()) - {"time"}
    requested_variables = set(RAW_WEATHER_VARIABLES)

    missing_variables = (
        requested_variables - returned_variables
    )

    if missing_variables:
        raise ValueError(
            f"{city_name}: variables not returned by API: "
            f"{sorted(missing_variables)}"
        )

    lengths = {
        variable: len(values)
        for variable, values in hourly.items()
    }

    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"{city_name}: hourly variables have "
            f"different lengths: {lengths}"
        )

def weather_json_to_dataframe(data, city_name):
    """Convert Open-Meteo hourly JSON into a DataFrame."""

    validate_weather_payload(
        data,
        city_name,
    )

    hourly = data["hourly"]

    df = pd.DataFrame(hourly)

    df["timestamp_utc"] = pd.to_datetime(
        df["time"],
        unit="s",
        utc=True,
    )

    df["timestamp_local"] = (
        df["timestamp_utc"]
        .dt.tz_convert(LOCAL_TIMEZONE)
    )

    df["city"] = city_name

    df = df.drop(columns=["time"])

    ordered_columns = [
        "timestamp_utc",
        "timestamp_local",
        "city",
        *RAW_WEATHER_VARIABLES,
    ]

    df = df[ordered_columns]

    if not df["timestamp_utc"].is_monotonic_increasing:
        raise ValueError(
            f"{city_name}: timestamps are not "
            f"in chronological order"
        )

    return df

def basic_stream_quality_report(df):
    """Create an initial data-quality report."""

    report = {
        "rows": len(df),
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
        "duplicate_timestamps": int(
            df["timestamp_utc"]
            .duplicated()
            .sum()
        ),
        "missing_by_variable": {
            column: int(df[column].isna().sum())
            for column in RAW_WEATHER_VARIABLES
        },
        "chronological_order": bool(
            df["timestamp_utc"]
            .is_monotonic_increasing
        ),
    }

    if not df.empty:
        report["start_utc"] = (
            df["timestamp_utc"]
            .iloc[0]
            .isoformat()
        )

        report["end_utc"] = (
            df["timestamp_utc"]
            .iloc[-1]
            .isoformat()
        )

    return report

def build_provenance(
    city_name,
    latitude,
    longitude,
    weather_data,
    response,
    raw_path,
    processed_path,
):
    """Build metadata describing the retrieved dataset."""

    return {
        "provider": "Open-Meteo",
        "dataset": "Historical Weather API",
        "city": city_name,
        "requested_latitude": latitude,
        "requested_longitude": longitude,
        "response_latitude": weather_data.get("latitude"),
        "response_longitude": weather_data.get("longitude"),
        "response_elevation": weather_data.get("elevation"),
        "api_timezone": API_TIMEZONE,
        "local_timezone": LOCAL_TIMEZONE,
        "historical_start": HISTORICAL_START,
        "historical_end": HISTORICAL_END,
        "model": OPEN_METEO_MODEL,
        "variables_requested": RAW_WEATHER_VARIABLES,
        "hourly_units":
            weather_data.get(
                "hourly_units",
                {},
            ),
        "retrieved_at_utc": (
            pd.Timestamp.now(tz="UTC")
            .isoformat()
        ),
        "request_url": response.url,
        "raw_file": str(
            raw_path.relative_to(PROJECT_ROOT)
        ),
        "processed_file": str(
            processed_path.relative_to(PROJECT_ROOT)
        ),
    }

def download_city(city_name):
    """Download and save historical weather for one city."""

    print(f"\n{'=' * 50}")
    print(f"Downloading: {city_name}")
    print(f"{'=' * 50}")

    # 1. Resolve city coordinates
    city_config = CITIES[city_name]

    latitude = city_config["latitude"]
    longitude = city_config["longitude"]

    print("\nFixed location:")
    print("Latitude:", latitude)
    print("Longitude:", longitude)

    city_slug = city_name.lower()

    

    # Download historical weather
    weather_data, weather_response = (
        fetch_historical_weather(
            latitude,
            longitude,
        )
    )

    # Validate API response
    validate_weather_payload(
        weather_data,
        city_name,
    )

    # Save untouched API response
    raw_weather_path = (
        RAW_DIR
        / f"{city_slug}_weather.json"
    )

    save_json(
        weather_data,
        raw_weather_path,
    )

    # Convert to tabular format
    df = weather_json_to_dataframe(
        weather_data,
        city_name,
    )

    # Save normalized table
    processed_path = (
        PROCESSED_DIR
        / f"{city_slug}_hourly.csv"
    )

    df.to_csv(
        processed_path,
        index=False,
    )

    # Save provenance
    provenance = build_provenance(
        city_name=city_name,
        latitude=latitude,
        longitude=longitude,
        response=weather_response,
        raw_path=raw_weather_path,
        processed_path=processed_path,
        weather_data=weather_data,
    )

    provenance_path = (
        RAW_DIR
        / f"{city_slug}_provenance.json"
    )

    save_json(
        provenance,
        provenance_path,
    )

    # Initial quality report
    quality_report = (
        basic_stream_quality_report(df)
    )

    quality_path = (
        PROCESSED_DIR
        / f"{city_slug}_quality_initial.json"
    )

    save_json(
        quality_report,
        quality_path,
    )

    print("\nDownload summary:")
    print("Rows:", len(df))
    print(
        "Period:",
        df["timestamp_utc"].iloc[0],
        "→",
        df["timestamp_utc"].iloc[-1],
    )
    print(
        "Missing values:",
        sum(
            quality_report[
                "missing_by_variable"
            ].values()
        ),
    )
    print(
        "Duplicate timestamps:",
        quality_report[
            "duplicate_timestamps"
        ],
    )

    return df


# Main function to download the data 
def main():
    porto_df = download_city("Porto")

    print(
        "\nWaiting 65 seconds to respect "
        "Open-Meteo rate limits..."
    )
    time.sleep(65)

    braganca_df = download_city(
        "Braganca"
    )

    print("\n" + "=" * 50)
    print("DATA ACQUISITION COMPLETE")
    print("=" * 50)

    print(
        f"Porto: {len(porto_df):,} rows"
    )

    print(
        f"Braganca: {len(braganca_df):,} rows"
    )

if __name__ == "__main__":
    main()