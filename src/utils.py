"""
Pearls AQI Predictor — Utility Functions
API wrappers, feature engineering, and AQI calculation helpers.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging

from config import (
    AQICN_API_KEY,
    OPENWEATHER_API_KEY,
    CITY_CONFIG,
    POLLUTANT_COLUMNS,
    WEATHER_COLUMNS,
    ROLLING_WINDOWS,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# API Wrapper Functions
# ─────────────────────────────────────────────────────────────────────────────

def fetch_aqicn_data(city_station: str = None) -> Optional[Dict]:
    """
    Fetch current air quality data from AQICN API.
    
    Args:
        city_station: AQICN station ID (e.g., '@11348' for Karachi)
                     If None, uses CITY_CONFIG['aqicn_station']
    
    Returns:
        Dictionary with pollutant data and AQI, or None if request fails
        
    Example response structure:
        {
            'aqi': 156,
            'pm25': 65.5,
            'pm10': 120.3,
            'o3': 45.2,
            'no2': 32.1,
            'so2': 12.5,
            'co': 0.8,
            'timestamp': '2024-01-15T10:00:00Z',
            'dominentpol': 'pm25'
        }
    """
    if city_station is None:
        city_station = CITY_CONFIG['aqicn_station']
    
    url = f"https://api.waqi.info/feed/{city_station}/"
    params = {'token': AQICN_API_KEY}
    
    try:
        logger.info(f"Fetching AQICN data for station {city_station}")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') != 'ok':
            logger.error(f"AQICN API error: {data.get('data', 'Unknown error')}")
            return None
        
        raw_data = data['data']
        
        # Extract pollutant concentrations (iaqi = individual air quality index)
        iaqi = raw_data.get('iaqi', {})
        
        result = {
            'aqi': raw_data.get('aqi', None),
            'pm25': iaqi.get('pm25', {}).get('v', None),
            'pm10': iaqi.get('pm10', {}).get('v', None),
            'o3': iaqi.get('o3', {}).get('v', None),
            'no2': iaqi.get('no2', {}).get('v', None),
            'so2': iaqi.get('so2', {}).get('v', None),
            'co': iaqi.get('co', {}).get('v', None),
            'timestamp': raw_data.get('time', {}).get('iso', datetime.utcnow().isoformat()),
            'dominentpol': raw_data.get('dominentpol', None),
        }
        
        logger.info(f"Successfully fetched AQICN data: AQI={result['aqi']}")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch AQICN data: {e}")
        return None


def fetch_openweather_data(lat: float = None, lon: float = None) -> Optional[Dict]:
    """
    Fetch current weather data from OpenWeatherMap API.
    
    Args:
        lat: Latitude (if None, uses CITY_CONFIG['lat'])
        lon: Longitude (if None, uses CITY_CONFIG['lon'])
    
    Returns:
        Dictionary with weather data, or None if request fails
        
    Example response structure:
        {
            'temperature': 28.5,      # Celsius
            'humidity': 65,           # Percentage
            'wind_speed': 3.5,        # m/s
            'pressure': 1013,         # hPa
            'visibility': 10000,      # meters
            'weather_main': 'Haze',
            'weather_description': 'haze',
            'clouds': 75,             # Percentage
            'timestamp': '2024-01-15T10:00:00Z'
        }
    """
    if lat is None:
        lat = CITY_CONFIG['lat']
    if lon is None:
        lon = CITY_CONFIG['lon']
    
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': OPENWEATHER_API_KEY,
        'units': 'metric'  # Celsius, m/s
    }
    
    try:
        logger.info(f"Fetching OpenWeather data for ({lat}, {lon})")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        result = {
            'temperature': data['main'].get('temp', None),
            'humidity': data['main'].get('humidity', None),
            'wind_speed': data['wind'].get('speed', None),
            'pressure': data['main'].get('pressure', None),
            'visibility': data.get('visibility', None),
            'weather_main': data['weather'][0].get('main', None) if data.get('weather') else None,
            'weather_description': data['weather'][0].get('description', None) if data.get('weather') else None,
            'clouds': data['clouds'].get('all', None) if 'clouds' in data else None,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        logger.info(f"Successfully fetched OpenWeather data: Temp={result['temperature']}°C")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch OpenWeather data: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering Functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_time_features(timestamp: datetime) -> Dict:
    """
    Extract time-based features from a timestamp.
    
    Args:
        timestamp: datetime object
    
    Returns:
        Dictionary with time features
    """
    return {
        'hour': timestamp.hour,
        'day_of_week': timestamp.weekday(),  # 0=Monday, 6=Sunday
        'day_of_month': timestamp.day,
        'month': timestamp.month,
        'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
        'season': get_season(timestamp.month),
    }


def get_season(month: int) -> int:
    """
    Map month to season (for Northern Hemisphere).
    
    Returns:
        0: Winter (Dec, Jan, Feb)
        1: Spring (Mar, Apr, May)
        2: Summer (Jun, Jul, Aug)
        3: Fall (Sep, Oct, Nov)
    """
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Fall


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features from raw pollutant and weather data.
    
    Derived features include:
    - Rolling averages (3h, 6h, 12h, 24h) for pollutants and AQI
    - AQI change rate (difference from previous hour)
    - Pollutant ratios (PM2.5/PM10, NO2/O3)
    - Temperature-humidity interaction
    
    Args:
        df: DataFrame with raw features (must be sorted by timestamp)
    
    Returns:
        DataFrame with additional derived features
    """
    df = df.copy()
    
    # Ensure sorted by timestamp
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    # ── Rolling averages ──────────────────────────────────────────────────────
    for window in ROLLING_WINDOWS:
        # AQI rolling average
        if 'aqi' in df.columns:
            df[f'aqi_rolling_{window}h'] = df['aqi'].rolling(window=window, min_periods=1).mean()
        
        # PM2.5 rolling average
        if 'pm25' in df.columns:
            df[f'pm25_rolling_{window}h'] = df['pm25'].rolling(window=window, min_periods=1).mean()
        
        # PM10 rolling average
        if 'pm10' in df.columns:
            df[f'pm10_rolling_{window}h'] = df['pm10'].rolling(window=window, min_periods=1).mean()
    
    # ── AQI change rate ───────────────────────────────────────────────────────
    if 'aqi' in df.columns:
        df['aqi_change_1h'] = df['aqi'].diff(1).fillna(0)
        df['aqi_change_3h'] = df['aqi'].diff(3).fillna(0)
    
    # ── Pollutant ratios ──────────────────────────────────────────────────────
    if 'pm25' in df.columns and 'pm10' in df.columns:
        df['pm25_pm10_ratio'] = df['pm25'] / (df['pm10'] + 1e-6)  # Avoid division by zero
    
    if 'no2' in df.columns and 'o3' in df.columns:
        df['no2_o3_ratio'] = df['no2'] / (df['o3'] + 1e-6)
    
    # ── Weather interactions ──────────────────────────────────────────────────
    if 'temperature' in df.columns and 'humidity' in df.columns:
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
    
    if 'wind_speed' in df.columns and 'pm25' in df.columns:
        # Wind dispersion effect (higher wind = lower PM2.5 accumulation)
        df['wind_pm25_interaction'] = df['wind_speed'] * df['pm25']
    
    return df


def compute_features(aqi_data: Dict, weather_data: Dict) -> Dict:
    """
    Merge raw API data and compute all features for a single timestamp.
    
    Args:
        aqi_data: Dictionary from fetch_aqicn_data()
        weather_data: Dictionary from fetch_openweather_data()
    
    Returns:
        Dictionary with all features ready for Feature Store
    """
    # Parse timestamp
    timestamp_str = aqi_data.get('timestamp', datetime.utcnow().isoformat())
    timestamp = pd.to_datetime(timestamp_str)
    
    # Start with time features
    features = compute_time_features(timestamp)
    
    # Add pollutant data
    features.update({
        'aqi': aqi_data.get('aqi'),
        'pm25': aqi_data.get('pm25'),
        'pm10': aqi_data.get('pm10'),
        'o3': aqi_data.get('o3'),
        'no2': aqi_data.get('no2'),
        'so2': aqi_data.get('so2'),
        'co': aqi_data.get('co'),
        'dominentpol': aqi_data.get('dominentpol'),
    })
    
    # Add weather data (OpenMeteo fields only)
    features.update({
        'temperature': weather_data.get('temperature'),   # temperature_2m
        'humidity': weather_data.get('humidity'),         # relative_humidity_2m
        'wind_speed': weather_data.get('wind_speed'),     # wind_speed_10m
        'pressure': weather_data.get('pressure'),         # pressure_msl
        'visibility': weather_data.get('visibility'),     # visibility
        'clouds': weather_data.get('clouds'),             # cloud_cover
    })
    
    # Add timestamp
    features['timestamp'] = timestamp
    
    # Replace this block at the end of compute_features()
    FLOAT_FIELDS = [
        'pm25', 'pm10', 'o3', 'no2', 'so2', 'co',
        'temperature', 'wind_speed',
    ]
    INT_FIELDS = [
        'aqi', 'humidity', 'pressure', 'visibility', 'clouds',
        'hour', 'day_of_week', 'day_of_month', 'month', 'season', 'is_weekend',
    ]

    for field in FLOAT_FIELDS:
        val = features.get(field)
        try:
            features[field] = float(val) if val is not None else np.nan
        except (TypeError, ValueError):
            features[field] = np.nan

    for field in INT_FIELDS:
        val = features.get(field)
        try:
            features[field] = int(float(val)) if val is not None else 0
        except (TypeError, ValueError):
            features[field] = 0

    return features


# ─────────────────────────────────────────────────────────────────────────────
# AQI Calculation (EPA Standard)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_aqi_from_pollutant(pollutant: str, concentration: float) -> Optional[float]:
    """
    Calculate AQI for a specific pollutant using EPA breakpoints.
    
    Args:
        pollutant: One of 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co'
        concentration: Pollutant concentration in µg/m³ (or ppm for gases)
    
    Returns:
        AQI value (0-500), or None if invalid
    """
    # EPA AQI breakpoints (simplified version)
    # Full implementation would include all pollutants and 8-hour/24-hour averages
    
    breakpoints = {
        'pm25': [  # µg/m³ (24-hour average)
            (0.0, 12.0, 0, 50),
            (12.1, 35.4, 51, 100),
            (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 500.4, 301, 500),
        ],
        'pm10': [  # µg/m³ (24-hour average)
            (0, 54, 0, 50),
            (55, 154, 51, 100),
            (155, 254, 101, 150),
            (255, 354, 151, 200),
            (355, 424, 201, 300),
            (425, 604, 301, 500),
        ],
    }
    
    if pollutant not in breakpoints:
        return None
    
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints[pollutant]:
        if bp_lo <= concentration <= bp_hi:
            # Linear interpolation
            aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + aqi_lo
            return round(aqi)
    
    # If concentration exceeds all breakpoints, return max AQI
    return 500


def compute_aqi_target(pollutant_data: Dict) -> float:
    """
    Compute overall AQI from multiple pollutants (take the maximum).
    
    Args:
        pollutant_data: Dictionary with pollutant concentrations
    
    Returns:
        Overall AQI value
    """
    aqi_values = []
    
    if pollutant_data.get('pm25') is not None:
        aqi_pm25 = calculate_aqi_from_pollutant('pm25', pollutant_data['pm25'])
        if aqi_pm25:
            aqi_values.append(aqi_pm25)
    
    if pollutant_data.get('pm10') is not None:
        aqi_pm10 = calculate_aqi_from_pollutant('pm10', pollutant_data['pm10'])
        if aqi_pm10:
            aqi_values.append(aqi_pm10)
    
    # If we have AQI values from pollutants, return max
    if aqi_values:
        return max(aqi_values)
    
    # Otherwise, return the AQI from API if available
    return pollutant_data.get('aqi', 0)


# ─────────────────────────────────────────────────────────────────────────────
# Data Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_feature_data(features: Dict) -> Tuple[bool, str]:
    """
    Validate that feature data is complete and within expected ranges.
    
    Args:
        features: Dictionary of features
    
    Returns:
        (is_valid, error_message)
    """
    # Check required fields
    required_fields = ['timestamp', 'aqi', 'pm25', 'temperature', 'humidity']
    missing = [f for f in required_fields if features.get(f) is None]
    
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"
    
    # Check AQI range
    if not (0 <= features['aqi'] <= 500):
        return False, f"AQI out of range: {features['aqi']}"
    
    # Check temperature range (reasonable for Earth)
    if not (-50 <= features['temperature'] <= 60):
        return False, f"Temperature out of range: {features['temperature']}"
    
    # Check humidity range
    if not (0 <= features['humidity'] <= 100):
        return False, f"Humidity out of range: {features['humidity']}"
    
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# OpenMeteo Historical Data (Free, No API Key Required)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_openmeteo_weather(lat: float, lon: float, start_date: str, end_date: str):
    """
    Fetch historical hourly weather data from OpenMeteo Archive API.
    Free tier, no API key required. 10,000 requests/day limit.

    Docs: https://open-meteo.com/en/docs/historical-weather-api

    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD

    Returns:
        DataFrame with columns: timestamp, temperature, humidity, wind_speed,
        pressure, visibility, clouds. Returns None on failure.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,visibility,cloud_cover",
        "timezone": "auto",
    }

    try:
        logger.info(f"Fetching OpenMeteo weather: {start_date} to {end_date}")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        hourly = data["hourly"]
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly["time"]),
            "temperature": hourly["temperature_2m"],
            "humidity": hourly["relative_humidity_2m"],
            "wind_speed": hourly["wind_speed_10m"],
            "pressure": hourly["pressure_msl"],
            "visibility": hourly.get("visibility", [10000] * len(hourly["time"])),
            "clouds": hourly.get("cloud_cover", [0] * len(hourly["time"])),
        })
        print(hourly["visibility"])
        # Drop rows where all weather values are null
        weather_cols = ["temperature", "humidity", "wind_speed", "pressure"]
        df = df.dropna(subset=weather_cols, how="all")

        logger.info(f"✅ OpenMeteo weather: {len(df)} hourly records")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch OpenMeteo weather: {e}")
        return None


def fetch_openmeteo_aqi(lat: float, lon: float, start_date: str, end_date: str):
    """
    Fetch historical hourly air quality data from OpenMeteo Air Quality API.
    Free tier, no API key required.

    Docs: https://open-meteo.com/en/docs/air-quality-api

    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD

    Returns:
        DataFrame with columns: timestamp, pm25, pm10, o3, no2, so2, co.
        Returns None on failure.
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "pm2_5,pm10,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide",
        "timezone": "auto",
    }

    try:
        logger.info(f"Fetching OpenMeteo AQI: {start_date} to {end_date}")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        hourly = data["hourly"]
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly["time"]),
            "pm25": hourly["pm2_5"],
            "pm10": hourly["pm10"],
            "o3": hourly["ozone"],
            "no2": hourly["nitrogen_dioxide"],
            "so2": hourly["sulphur_dioxide"],
            "co": hourly["carbon_monoxide"],
        })

        # Drop rows where all pollutants are null
        pollutant_cols = ["pm25", "pm10", "o3", "no2", "so2", "co"]
        df = df.dropna(subset=pollutant_cols, how="all")

        logger.info(f"✅ OpenMeteo AQI: {len(df)} hourly records")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch OpenMeteo AQI: {e}")
        return None


if __name__ == "__main__":
    # Test API functions
    print("=== Testing API Functions ===\n")
    
    print("1. Fetching AQICN data...")
    aqi_data = fetch_aqicn_data()
    if aqi_data:
        print(f"   ✅ AQI: {aqi_data['aqi']}, PM2.5: {aqi_data['pm25']}")
    else:
        print("   ❌ Failed to fetch AQICN data")
    
    print("\n2. Fetching OpenWeather data...")
    weather_data = fetch_openweather_data()
    if weather_data:
        print(f"   ✅ Temp: {weather_data['temperature']}°C, Humidity: {weather_data['humidity']}%")
    else:
        print("   ❌ Failed to fetch OpenWeather data")
    
    if aqi_data and weather_data:
        print("\n3. Computing features...")
        features = compute_features(aqi_data, weather_data)
        print(f"   ✅ Generated {len(features)} features")
        
        print("\n4. Validating features...")
        is_valid, error = validate_feature_data(features)
        if is_valid:
            print("   ✅ Features are valid")
        else:
            print(f"   ❌ Validation error: {error}")
    
    print("\n5. Testing OpenMeteo historical APIs (free)...")
    from datetime import timedelta
    end = datetime.utcnow()
    start = end - timedelta(days=3)
    om_weather = fetch_openmeteo_weather(
        CITY_CONFIG['lat'], CITY_CONFIG['lon'],
        start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
    )
    if om_weather is not None:
        print(f"   ✅ OpenMeteo weather: {len(om_weather)} rows")
    else:
        print("   ❌ OpenMeteo weather failed")
    
    om_aqi = fetch_openmeteo_aqi(
        CITY_CONFIG['lat'], CITY_CONFIG['lon'],
        start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
    )
    if om_aqi is not None:
        print(f"   ✅ OpenMeteo AQI: {len(om_aqi)} rows")
    else:
        print("   ❌ OpenMeteo AQI failed")
