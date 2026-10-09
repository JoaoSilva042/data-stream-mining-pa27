ASSIGNMENT_CODE = "PA27"

CITY_A = "Porto"
CITY_B = "Braganca"

BUSINESS_CONTEXT = "Winter Road-Service Operator"

TARGET_ID = "T3"
TARGET_NAME = "Cold Stress Event"

HISTORICAL_START = "2021-01-01"
HISTORICAL_END = "2025-12-31"

WARMUP_START = "2021-01-01"
WARMUP_END = "2021-12-31"

EVALUATION_START = "2022-01-01"
EVALUATION_END = "2025-12-31"

TARGET_QUANTILE = 0.20

RANDOM_SEED = 42

RAW_WEATHER_VARIABLES = [
    # Core variables 
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",

    # Temperature 
    "dew_point_2m",
    "apparent_temperature",
    "vapour_pressure_deficit",

    # Precipitation 
    "rain",
    "snowfall",
    "weather_code",

    # Pressure
    "pressure_msl",
    "surface_pressure",

    # Clouds
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",

    # Wind
    "wind_direction_10m",

    # Sunlight
    "shortwave_radiation",
    "sunshine_duration",

    # Ground conditions
    "soil_temperature_0_to_7cm",
    "soil_temperature_7_to_28cm",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
]

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

API_TIMEZONE = "GMT"
LOCAL_TIMEZONE = "Europe/Lisbon"

CITIES = {
    "Porto": {
        "query": "Porto",
        "country_code": "PT",
        "latitude": 41.1485,
        "longitude": -8.61097,
    },
    "Braganca": {
        "query": "Braganca",
        "country_code": "PT",
        "latitude": 41.80716,
        "longitude": -6.75898,
    },
}

OPEN_METEO_MODEL = "era5"