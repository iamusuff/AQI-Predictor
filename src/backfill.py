"""
Pearls AQI Predictor — Historical Data Backfill
Fetches historical air quality and weather data from OpenMeteo (free, no key).
Falls back to synthetic data if the API is unavailable.

This script backfills 90-180 days of historical data for model training.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import logging
import time
import sys
import os

from config import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    CITY_CONFIG,
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
)
from utils import (
    fetch_openmeteo_weather,
    fetch_openmeteo_aqi,
    compute_time_features,
    compute_derived_features,
    compute_aqi_target,
    validate_feature_data,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# OpenMeteo Backfill (Primary)
# ─────────────────────────────────────────────────────────────────────────────

def backfill_via_openmeteo(lat: float, lon: float, start_date: str, end_date: str):
    """
    Fetch historical data from OpenMeteo APIs (free, no API key required).

    Makes 2 batch calls total (weather + AQI) instead of per-hour looping.

    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD

    Returns:
        DataFrame with hourly features, or None if both APIs fail
    """
    logger.info("─" * 70)
    logger.info("Backfill via OpenMeteo (free APIs, no key required)")
    logger.info("─" * 70)

    # Step 1: Fetch weather data
    logger.info("\n[1/3] Fetching historical weather from OpenMeteo...")
    weather_df = fetch_openmeteo_weather(lat, lon, start_date, end_date)
    if weather_df is None or weather_df.empty:
        logger.error("❌ OpenMeteo weather fetch returned no data")
        return None
    logger.info(f"✅ Weather: {len(weather_df)} hourly records")

    # Step 2: Fetch AQI data
    logger.info("\n[2/3] Fetching historical air quality from OpenMeteo...")
    aqi_df = fetch_openmeteo_aqi(lat, lon, start_date, end_date)
    if aqi_df is None or aqi_df.empty:
        logger.error("❌ OpenMeteo AQI fetch returned no data")
        return None
    logger.info(f"✅ AQI: {len(aqi_df)} hourly records")

    # Step 3: Merge weather + AQI on timestamp
    logger.info("\n[3/3] Merging and computing features...")
    df = pd.merge(aqi_df, weather_df, on="timestamp", how="inner")
    logger.info(f"✅ Merged: {len(df)} records")

    if df.empty:
        logger.error("❌ Merge produced empty DataFrame (no overlapping timestamps)")
        return None

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Compute AQI from raw pollutant concentrations (EPA method)
    df["aqi"] = df.apply(lambda row: compute_aqi_target(row.to_dict()), axis=1)

    # Compute time features
    time_data = df["timestamp"].apply(compute_time_features)
    time_df = pd.DataFrame(time_data.tolist())
    df = pd.concat([df, time_df], axis=1)

    # Add dominant pollutant placeholder (will be refined later)
    df["dominentpol"] = "pm25"

    # Fill null weather descriptions (OpenMeteo doesn't provide them)
    df["weather_main"] = None
    df["weather_description"] = None

    logger.info(f"✅ Generated {len(df)} rows with {len(df.columns)} features")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Data Generation (Fallback)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_weather(base_temp: float, base_humidity: float,
                               timestamp: datetime) -> Dict:
    """
    Generate synthetic weather data based on seasonal patterns.
    This is a fallback when historical weather API is not available.

    Args:
        base_temp: Base temperature for the city
        base_humidity: Base humidity for the city
        timestamp: Target datetime

    Returns:
        Dictionary with synthetic weather data
    """
    # Seasonal adjustment
    month = timestamp.month
    if month in [12, 1, 2]:  # Winter
        temp_adj = -5
    elif month in [3, 4, 5]:  # Spring
        temp_adj = 0
    elif month in [6, 7, 8]:  # Summer
        temp_adj = 5
    else:  # Fall
        temp_adj = -2

    # Daily variation (cooler at night, warmer during day)
    hour = timestamp.hour
    if 6 <= hour <= 18:  # Daytime
        hour_adj = 3 * np.sin((hour - 6) * np.pi / 12)
    else:  # Nighttime
        hour_adj = -3

    # Add some randomness
    np.random.seed(int(timestamp.timestamp()))
    noise = np.random.normal(0, 2)

    temperature = base_temp + temp_adj + hour_adj + noise
    humidity = base_humidity + np.random.normal(0, 5)
    humidity = max(20, min(100, humidity))  # Clamp to valid range

    return {
        'temperature': round(temperature, 2),
        'humidity': round(humidity, 0),
        'wind_speed': round(abs(np.random.normal(3, 1.5)), 2),
        'pressure': round(np.random.normal(1013, 5), 0),
        'visibility': 10000,
        'clouds': round(np.random.uniform(0, 100), 0),
        'weather_main': 'Clear',
        'weather_description': 'clear sky',
    }


def generate_synthetic_aqi(timestamp: datetime, base_aqi: float = 100) -> Dict:
    """
    Generate synthetic AQI data based on typical patterns.
    This is a fallback when historical AQI data is not available.

    Args:
        timestamp: Target datetime
        base_aqi: Base AQI level for the city

    Returns:
        Dictionary with synthetic AQI data
    """
    # Seasonal pattern (worse in winter due to heating, better in summer)
    month = timestamp.month
    if month in [12, 1, 2]:  # Winter
        seasonal_factor = 1.3
    elif month in [3, 4, 5]:  # Spring
        seasonal_factor = 1.1
    elif month in [6, 7, 8]:  # Summer
        seasonal_factor = 0.9
    else:  # Fall
        seasonal_factor = 1.0

    # Daily pattern (worse during rush hours)
    hour = timestamp.hour
    if hour in [7, 8, 9, 17, 18, 19]:  # Rush hours
        hourly_factor = 1.2
    elif hour in [0, 1, 2, 3, 4, 5]:  # Night
        hourly_factor = 0.8
    else:
        hourly_factor = 1.0

    # Add randomness
    np.random.seed(int(timestamp.timestamp()))
    noise = np.random.normal(1, 0.15)

    aqi = base_aqi * seasonal_factor * hourly_factor * noise
    aqi = max(20, min(300, aqi))  # Clamp to reasonable range

    # Generate pollutant concentrations based on AQI
    pm25 = aqi * 0.6 + np.random.normal(0, 5)
    pm10 = pm25 * 1.5 + np.random.normal(0, 10)

    return {
        'aqi': round(aqi, 0),
        'pm25': round(max(0, pm25), 1),
        'pm10': round(max(0, pm10), 1),
        'o3': round(abs(np.random.normal(30, 10)), 1),
        'no2': round(abs(np.random.normal(25, 8)), 1),
        'so2': round(abs(np.random.normal(15, 5)), 1),
        'co': round(abs(np.random.normal(0.5, 0.2)), 2),
        'dominentpol': 'pm25',
    }


def backfill_via_synthetic(days_back: int, base_aqi: float = 120,
                            base_temp: float = 28.0, base_humidity: float = 65.0):
    """
    Generate synthetic dataset for the specified number of days.

    Args:
        days_back: Number of days to generate
        base_aqi: Base AQI level
        base_temp: Base temperature
        base_humidity: Base humidity

    Returns:
        DataFrame with synthetic features
    """
    logger.info("─" * 70)
    logger.info("Backfill via Synthetic Data Generation")
    logger.info("─" * 70)

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)

    timestamps = []
    current = start_date
    while current <= end_date:
        timestamps.append(current)
        current += timedelta(hours=1)

    total_hours = len(timestamps)
    logger.info(f"Generating {total_hours} synthetic hourly records...")

    all_features = []
    for ts in timestamps:
        aqi_data = generate_synthetic_aqi(ts, base_aqi)
        weather_data = generate_synthetic_weather(base_temp, base_humidity, ts)
        time_features = compute_time_features(ts)
        features = {
            'timestamp': ts,
            **aqi_data,
            **weather_data,
            **time_features,
        }
        all_features.append(features)

    df = pd.DataFrame(all_features)
    logger.info(f"✅ Generated {len(df)} synthetic rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main Backfill Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def backfill_historical_data(
    days_back: int = 90,
    use_openmeteo: bool = True,
    save_local: bool = True,
    use_hopsworks: bool = False,
) -> pd.DataFrame:
    """
    Backfill historical data for training.

    Priority:
      1. OpenMeteo APIs (free, no key required) — 2 batch calls for all data
      2. Synthetic data (fallback if OpenMeteo unavailable)

    Args:
        days_back: Number of days to backfill (default: 90)
        use_openmeteo: Try OpenMeteo first (default: True).
                       Falls back to synthetic automatically on failure.
        save_local: Save to local CSV (default: True)
        use_hopsworks: Upload to Hopsworks (default: False)

    Returns:
        DataFrame with historical features
    """
    logger.info("=" * 70)
    logger.info("HISTORICAL DATA BACKFILL STARTED")
    logger.info(f"City: {CITY_CONFIG['name']}, {CITY_CONFIG['country']}")
    logger.info(f"Days to backfill: {days_back}")
    logger.info("=" * 70)

    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)
    logger.info(f"\nDate range: {start_date.date()} to {end_date.date()}")

    df = None

    # ── Try OpenMeteo (free, no key) ────────────────────────────────────────
    if use_openmeteo:
        logger.info("\n>>> Attempting OpenMeteo backfill (free, no API key)...")
        df = backfill_via_openmeteo(
            CITY_CONFIG['lat'], CITY_CONFIG['lon'],
            start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'),
        )
        if df is not None:
            logger.info("✅ OpenMeteo backfill succeeded")
        else:
            logger.warning("⚠️  OpenMeteo backfill failed — falling back to synthetic")

    # ── Fallback to Synthetic ───────────────────────────────────────────────
    if df is None:
        logger.info("\n>>> Using synthetic data generation (fallback)...")
        df = backfill_via_synthetic(days_back)
        if df.empty:
            logger.error("❌ Synthetic backfill also failed")
            return pd.DataFrame()
        logger.info("✅ Synthetic data generated")

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Compute derived features (rolling averages, ratios, etc.)
    logger.info("\nComputing derived features...")
    df = compute_derived_features(df)
    logger.info(f"✅ Final dataset: {len(df)} rows with {len(df.columns)} features")

    print(f"\n📊 Dataset summary: {len(df)} rows, {len(df.columns)} columns")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   AQI range: [{df['aqi'].min():.0f}, {df['aqi'].max():.0f}], mean: {df['aqi'].mean():.0f}")

    # ── Save locally ────────────────────────────────────────────────────────
    if save_local:
        logger.info("\nSaving to local CSV...")
        output_dir = "data/backfill"
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{output_dir}/backfill_{days_back}days_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"✅ Saved to {filename}")

        # Also save to main features file
        main_file = "data/features.csv"
        if os.path.exists(main_file):
            existing_df = pd.read_csv(main_file, parse_dates=['timestamp'])
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
            combined_df = combined_df.sort_values('timestamp')
            combined_df.to_csv(main_file, index=False)
            logger.info(f"✅ Merged with existing data: {len(combined_df)} total rows")
        else:
            df.to_csv(main_file, index=False)
            logger.info(f"✅ Created new features file: {main_file}")

    # ── Upload to Hopsworks ─────────────────────────────────────────────────
    if use_hopsworks:
        logger.info("\nUploading to Hopsworks...")
        try:
            import hopsworks

            project = hopsworks.login(
                api_key_value=HOPSWORKS_API_KEY,
                project=HOPSWORKS_PROJECT_NAME
            )
            feature_store = project.get_feature_store()

            feature_group = feature_store.get_or_create_feature_group(
                name=FEATURE_GROUP_NAME,
                version=FEATURE_GROUP_VERSION,
                description="AQI prediction features with historical backfill",
                primary_key=["timestamp"],
                event_time="timestamp",
                online_enabled=True,
            )

            feature_group.insert(df, write_options={"wait_for_job": True})
            logger.info(f"✅ Uploaded {len(df)} rows to Hopsworks")

        except Exception as e:
            logger.error(f"❌ Failed to upload to Hopsworks: {e}")

    logger.info("\n" + "=" * 70)
    logger.info("BACKFILL COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)

    return df


def run(days: int = 90, openmeteo: bool = True) -> bool:
    """
    Run the backfill pipeline. Tries OpenMeteo first, falls back to synthetic.

    Args:
        days: Number of days to backfill
        openmeteo: Try OpenMeteo first (default True)

    Returns:
        True if successful, False otherwise
    """
    try:
        df = backfill_historical_data(
            days_back=days,
            use_openmeteo=openmeteo,
            save_local=True,
            use_hopsworks=False,
        )
        return not df.empty
    except Exception as e:
        logger.error(f"Backfill failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AQI Historical Data Backfill")
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days to backfill (default: 90)"
    )
    parser.add_argument(
        "--no-openmeteo",
        action="store_true",
        help="Skip OpenMeteo and use synthetic data only"
    )
    parser.add_argument(
        "--hopsworks",
        action="store_true",
        help="Upload to Hopsworks Feature Store"
    )

    args = parser.parse_args()

    try:
        logger.info("Starting backfill pipeline...")

        df = backfill_historical_data(
            days_back=args.days,
            use_openmeteo=not args.no_openmeteo,
            save_local=True,
            use_hopsworks=args.hopsworks,
        )

        if not df.empty:
            logger.info(f"\n✅ Backfill successful! Generated {len(df)} rows.")
            sys.exit(0)
        else:
            logger.error("\n❌ Backfill failed - no data generated.")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n⚠️  Backfill interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Backfill failed with error: {e}", exc_info=True)
        sys.exit(1)
