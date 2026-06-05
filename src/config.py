"""
Pearls AQI Predictor — Central Configuration
Loads all environment variables and defines project-wide constants.
Supports both local .env files and Streamlit Cloud secrets.
"""

import os
from dotenv import load_dotenv

# Load .env file from project root (for local development)
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Load from Streamlit Secrets or Environment Variables
# ─────────────────────────────────────────────────────────────────────────────

def get_secret(key: str, default: str = "") -> str:
    """
    Get secret from Streamlit Cloud secrets or environment variables.
    Priority: Streamlit secrets > Environment variables > Default
    """
    try:
        # Try Streamlit secrets first (when deployed on Streamlit Cloud)
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except (ImportError, FileNotFoundError):
        # Streamlit not available or secrets not configured
        pass
    
    # Fall back to environment variables (local development)
    return os.getenv(key, default)


# ─────────────────────────────────────────────────────────────────────────────
# API Keys
# ─────────────────────────────────────────────────────────────────────────────
AQICN_API_KEY = get_secret("AQICN_API_KEY", "")
# Note: OpenMeteo is free and requires no API key

# ─────────────────────────────────────────────────────────────────────────────
# Hopsworks
# ─────────────────────────────────────────────────────────────────────────────
HOPSWORKS_API_KEY = get_secret("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT_NAME = get_secret("HOPSWORKS_PROJECT_NAME", "aqi_predictor99")

# Feature Store settings
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "aqi_feature_view"
FEATURE_VIEW_VERSION = 1

# Model Registry settings
MODEL_NAME = "aqi_predictor_model"
MODEL_VERSION = 1

# ─────────────────────────────────────────────────────────────────────────────
# City Configuration
# ─────────────────────────────────────────────────────────────────────────────
CITY = get_secret("CITY", "karachi")

# Supported cities with coordinates and AQICN station IDs
CITIES = {
    "karachi": {
        "name": "Karachi",
        "country": "Pakistan",
        "lat": float(get_secret("CITY_LAT", "24.8607")),
        "lon": float(get_secret("CITY_LON", "67.0011")),
        "aqicn_station": "@11348",       # Karachi station ID on AQICN
        "timezone": "Asia/Karachi",
    },
    "lahore": {
        "name": "Lahore",
        "country": "Pakistan",
        "lat": 31.5204,
        "lon": 74.3587,
        "aqicn_station": "@8430",
        "timezone": "Asia/Karachi",
    },
    "islamabad": {
        "name": "Islamabad",
        "country": "Pakistan",
        "lat": 33.6844,
        "lon": 73.0479,
        "aqicn_station": "@11420",
        "timezone": "Asia/Karachi",
    },
}

# Get active city config (fallback to karachi)
CITY_CONFIG = CITIES.get(CITY, CITIES["karachi"])

# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering Settings
# ─────────────────────────────────────────────────────────────────────────────

# Pollutant columns fetched from APIs
# Pollutant columns fetched from APIs
POLLUTANT_COLUMNS = ["pm25", "pm10", "o3", "no2", "so2", "co"]

# Weather columns fetched from OpenMeteo
WEATHER_COLUMNS = ["temperature", "humidity", "wind_speed", "pressure", "visibility"]

# Rolling window sizes (in hours) for derived features
ROLLING_WINDOWS = [3, 6, 12, 24]

# Forecast horizons (in hours ahead) — next 3 days
FORECAST_HORIZONS = [24, 48, 72]

# ─────────────────────────────────────────────────────────────────────────────
# AQI Health Categories (US EPA standard)
# ─────────────────────────────────────────────────────────────────────────────
AQI_CATEGORIES = [
    {"min": 0,   "max": 50,  "label": "Good",                    "color": "#00e400"},
    {"min": 51,  "max": 100, "label": "Moderate",                "color": "#ffff00"},
    {"min": 101, "max": 150, "label": "Unhealthy for Sensitive",  "color": "#ff7e00"},
    {"min": 151, "max": 200, "label": "Unhealthy",               "color": "#ff0000"},
    {"min": 201, "max": 300, "label": "Very Unhealthy",          "color": "#8f3f97"},
    {"min": 301, "max": 500, "label": "Hazardous",               "color": "#7e0023"},
]

# ─────────────────────────────────────────────────────────────────────────────
# Validation Helper
# ─────────────────────────────────────────────────────────────────────────────
def validate_config() -> bool:
    """
    Checks that all required environment variables are set.
    Returns True if valid, raises ValueError with details if not.
    """
    missing = []
    if not AQICN_API_KEY:
        missing.append("AQICN_API_KEY")
    # OpenMeteo is free and requires no API key
    if not HOPSWORKS_API_KEY:
        missing.append("HOPSWORKS_API_KEY")

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Please copy .env.example to .env and fill in your API keys."
        )
    return True


def get_aqi_category(aqi_value: float) -> dict:
    """Return the AQI health category dict for a given AQI value."""
    for cat in AQI_CATEGORIES:
        if cat["min"] <= aqi_value <= cat["max"]:
            return cat
    return AQI_CATEGORIES[-1]  # Hazardous as fallback


if __name__ == "__main__":
    print("=== AQI Predictor Configuration ===")
    print(f"  City         : {CITY_CONFIG['name']}, {CITY_CONFIG['country']}")
    print(f"  Coordinates  : {CITY_CONFIG['lat']}°N, {CITY_CONFIG['lon']}°E")
    print(f"  AQICN Station: {CITY_CONFIG['aqicn_station']}")
    print(f"  Hopsworks    : {HOPSWORKS_PROJECT_NAME}")
    print()
    try:
        validate_config()
        print("  ✅ All API keys are set.")
    except ValueError as e:
        print(f"  ⚠️  {e}")
