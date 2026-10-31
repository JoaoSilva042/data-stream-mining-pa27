ASSIGNMENT_CODE = "PA27"

CITY_A = "Porto"
CITY_B = "Braganca"

BUSINESS_CONTEXT = "Winter Road-Service Operator"

TARGET_ID = "T3"
TARGET_NAME = "Cold Stress Event"

HISTORICAL_START = "2016-01-01"
HISTORICAL_END = "2025-12-31"

WARMUP_START = "2016-01-01"
WARMUP_END = "2017-12-31"

EVALUATION_START = "2018-01-01"
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

# Target configuration
TARGET_COLUMN = "cold_stress"
TARGET_SOURCE_COLUMN = "temperature_2m"
POSITIVE_CLASS = 1

# Required representations
REPRESENTATIONS = (
    "A",
    "B",
)
# City-specific base predictors selected from warm-up EDA.
BASE_FEATURES = {
    "Porto": (
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
    ),

    "Braganca": (
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
    ),
}

# Representation B temporal windows.
TEMPORAL_WINDOW_HOURS = 6
RECENT_EVENT_WINDOW_HOURS = 24

# Streaming evaluation
EVALUATION_LABELS = (0, 1)

# Recent-performance window:
# 168 hourly observations = 7 complete days.
#
# This provides a recent operational view of model
# performance while covering seven full daily cycles.
EVALUATION_WINDOW = 168



# Model settings fixed a priori before the
# official evaluation period.
#
# The configurations are chosen as reasonable
# initial settings for an imbalanced streaming
# classification problem. They are not tuned
# on the 2018-2025 evaluation data.
MODEL_PARAMS = {
    "gaussian_nb": {},

    "logistic_regression": {
        "learning_rate": 0.05,
        "standardize": True,
    },

    "hoeffding_tree": {
        "grace_period": 200,
        "split_criterion": "hellinger",
        "delta": 1e-7,
        "tau": 0.05,
        "leaf_prediction": "nba",
    },

    "online_bagging": {
        "n_models": 10,
    },

    "adaptive_random_forest": {
        "n_models": 10,
        "max_features": "sqrt",
        "lambda_value": 6,
        "split_criterion": "hellinger",
        "leaf_prediction": "nba",
        "grace_period": 50,
        "delta": 0.01,
        "tau": 0.05,
    },
}


MODEL_NAMES = tuple(MODEL_PARAMS.keys())