"""
Pearls AQI Predictor — Feature Pipeline
Fetches real-time air quality data (AQICN) & weather data (OpenMeteo), engineers features, 
and stores in Hopsworks.

Data Sources:
  - AQICN API: Real-time pollutants (PM2.5, PM10, O3, NO2, SO2, CO)
  - OpenMeteo API: Weather data (free, no API key required)

This pipeline runs hourly via GitHub Actions to continuously update the feature store.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import logging
import sys

from config import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    CITY_CONFIG,
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
)
from utils import (
    fetch_aqicn_data,
    fetch_openmeteo_weather,
    compute_features,
    validate_feature_data,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Hopsworks Integration
# ─────────────────────────────────────────────────────────────────────────────

def connect_to_hopsworks():
    try:
        import hopsworks

        logger.info("Connecting to Hopsworks...")
        project = hopsworks.login(
            host="eu-west.cloud.hopsworks.ai",
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT_NAME
        )

        feature_store = project.get_feature_store()
        logger.info(f"✅ Connected to Hopsworks project: {HOPSWORKS_PROJECT_NAME}")
        return project, feature_store

    except ImportError:
        logger.warning("⚠️  Hopsworks library not installed. Features will be saved locally.")
        return None, None
    except Exception as e:
        logger.error(f"❌ Failed to connect to Hopsworks: {e}")
        return None, None


def get_or_create_feature_group(feature_store, sample_df=None):
    # ── Try fetching existing group first ─────────────────────────────────────
    try:
        fg = feature_store.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION
        )
        if fg is not None:
            logger.info(f"✅ Found existing feature group: {FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}")
            return fg
        else:
            logger.warning("⚠️  get_feature_group returned None — will attempt creation.")
    except Exception as e:
        logger.warning(f"⚠️  get_feature_group raised exception — will attempt creation. Reason: {e}")

    # ── Create feature group ──────────────────────────────────────────────────
    try:
        logger.info(f"Creating feature group: {FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}")

        feature_group = feature_store.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            primary_key=["timestamp"],
            event_time="timestamp",
            description="AQI prediction features for Karachi",
            online_enabled=True,
            stream=True,
        )

        if feature_group is None:
            logger.error("❌ get_or_create_feature_group returned None.")
            return None

        logger.info(f"✅ Feature group ready: {feature_group.name} v{feature_group.version}")
        return feature_group

    except Exception as e:
        logger.error(f"❌ Failed to create feature group: {e}", exc_info=True)
        return None


def insert_features_to_hopsworks(features_df: pd.DataFrame, feature_store) -> bool:
    try:
        logger.info("Resolving feature group...")
        feature_group = get_or_create_feature_group(feature_store, sample_df=features_df)

        if feature_group is None:
            logger.error("❌ feature_group is None — aborting insert.")
            return False

        logger.info(f"✅ Feature group confirmed: {feature_group.name} v{feature_group.version}")

        features_df = features_df.copy()

        features_df = features_df.drop(columns=['dominentpol'], errors='ignore')

        # ── Enforce exact dtypes ──────────────────────────────────────────────
        int_columns = [
            'aqi',
            'humidity', 'pressure', 'visibility', 'clouds',
            'hour', 'day_of_week', 'day_of_month', 'month',
            'season', 'is_weekend',
        ]
        float_columns = [
            'pm25', 'pm10', 'o3', 'no2', 'so2', 'co',
            'temperature', 'wind_speed',
        ]

        for col in int_columns:
            if col in features_df.columns:
                features_df[col] = features_df[col].astype('int64')

        for col in float_columns:
            if col in features_df.columns:
                features_df[col] = features_df[col].astype('float64')

        # ── Timestamp naive UTC ───────────────────────────────────────────────
        features_df["timestamp"] = pd.to_datetime(
            features_df["timestamp"], utc=True
        ).dt.tz_localize(None)

        logger.info(f"Dtypes after cast:\n{features_df.dtypes}")
        logger.info(f"Inserting {len(features_df)} row(s)...")

        feature_group.insert(
            features_df,
            write_options={"wait_for_job": False},
            validation_options={"run_validation": False}
        )

        logger.info(f"✅ Inserted {len(features_df)} row(s) into '{FEATURE_GROUP_NAME}'")

        # ── Verify data after insert ──────────────────────────────────────────
        # Taake UI pe confirm ho sake ke data gaya
        try:
            logger.info("Verifying insert via fg.read()...")
            df_check = feature_group.read(
                read_options={"use_hive": False}
            )
            logger.info(f"✅ Verification: {len(df_check)} row(s) readable from feature store")
            logger.info(f"Latest timestamp: {df_check['timestamp'].max()}")
        except Exception as e:
            logger.warning(f"⚠️  Verification read failed (insert may still be ok): {e}")

        return True

    except Exception as e:
        logger.error(f"❌ Insert failed: {e}", exc_info=True)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Local Storage (Fallback)
# ─────────────────────────────────────────────────────────────────────────────

def save_features_locally(features_df: pd.DataFrame, filepath: str = "data/features.csv"):
    import os

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    features_df['timestamp'] = pd.to_datetime(features_df['timestamp'], utc=True)

    if os.path.exists(filepath):
        existing_df = pd.read_csv(filepath, parse_dates=['timestamp'])
        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'], utc=True)

        combined_df = pd.concat([existing_df, features_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
        combined_df = combined_df.sort_values('timestamp')
        combined_df.to_csv(filepath, index=False)

        logger.info(f"✅ Appended features to {filepath} (total rows: {len(combined_df)})")
    else:
        features_df.to_csv(filepath, index=False)
        logger.info(f"✅ Saved features to {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(use_hopsworks: bool = True, save_local: bool = True) -> bool:
    logger.info("=" * 60)
    logger.info("FEATURE PIPELINE STARTED")
    logger.info(f"City: {CITY_CONFIG['name']}, {CITY_CONFIG['country']}")
    logger.info(f"Timestamp: {pd.Timestamp.utcnow().isoformat()}")
    logger.info("=" * 60)

    # ── Step 1: Fetch AQI data ────────────────────────────────────────────────
    logger.info("\n[1/5] Fetching air quality data from AQICN...")
    aqi_data = fetch_aqicn_data()
    if aqi_data is None:
        logger.error("❌ Failed to fetch AQI data. Aborting pipeline.")
        return False
    logger.info(f"✅ AQI: {aqi_data['aqi']}, PM2.5: {aqi_data['pm25']}, Dominant: {aqi_data['dominentpol']}")

    # ── Step 2: Fetch weather data ────────────────────────────────────────────
    logger.info("\n[2/5] Fetching weather data from OpenMeteo (free, no API key required)...")
    today = pd.Timestamp.utcnow().strftime('%Y-%m-%d')
    weather_df = fetch_openmeteo_weather(
        lat=CITY_CONFIG['lat'],
        lon=CITY_CONFIG['lon'],
        start_date=today,
        end_date=today
    )
    if weather_df is None or weather_df.empty:
        logger.error("❌ Failed to fetch weather data. Aborting pipeline.")
        return False
    
    # Extract latest weather record (last row of today's data)
    weather_data = weather_df.iloc[-1].to_dict()
    logger.info(f"✅ Temp: {weather_data.get('temperature')}°C, Humidity: {weather_data.get('humidity')}%, Wind: {weather_data.get('wind_speed')} m/s")

    # ── Step 3: Compute features ──────────────────────────────────────────────
    logger.info("\n[3/5] Computing features...")
    features = compute_features(aqi_data, weather_data)

    is_valid, error_msg = validate_feature_data(features)
    if not is_valid:
        logger.error(f"❌ Feature validation failed: {error_msg}")
        return False

    logger.info(f"✅ Generated {len(features)} features")
    features_df = pd.DataFrame([features])

    logger.info("\nSample features:")
    for col in ['timestamp', 'aqi', 'pm25', 'temperature', 'humidity', 'hour', 'day_of_week', 'season']:
        if col in features_df.columns:
            logger.info(f"  {col}: {features_df[col].values[0]}")

    # ── Step 4: Store in Hopsworks ────────────────────────────────────────────
    hopsworks_success = False
    if use_hopsworks:
        logger.info("\n[4/5] Storing features in Hopsworks Feature Store...")
        project, feature_store = connect_to_hopsworks()
        if feature_store is not None:
            hopsworks_success = insert_features_to_hopsworks(features_df, feature_store)
        else:
            logger.warning("⚠️  Skipping Hopsworks storage (connection failed)")
    else:
        logger.info("\n[4/5] Skipping Hopsworks storage (disabled)")

    # ── Step 5: Save locally ──────────────────────────────────────────────────
    if save_local:
        logger.info("\n[5/5] Saving features locally (backup)...")
        save_features_locally(features_df)
    else:
        logger.info("\n[5/5] Skipping local storage (disabled)")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE PIPELINE COMPLETED")
    logger.info(f"  Hopsworks: {'✅ Success' if hopsworks_success else '⚠️  Skipped/Failed'}")
    logger.info(f"  Local Storage: {'✅ Success' if save_local else '⚠️  Skipped'}")
    logger.info("=" * 60)

    return True


def run_batch(hours_back: int = 24, use_hopsworks: bool = True, save_local: bool = True):
    logger.info(f"Running batch feature pipeline for last {hours_back} hours...")
    success_count = 0
    for i in range(hours_back):
        logger.info(f"\n--- Processing hour {i+1}/{hours_back} ---")
        if run(use_hopsworks=use_hopsworks, save_local=save_local):
            success_count += 1
        if i < hours_back - 1:
            import time
            time.sleep(2)
    logger.info(f"\nBatch pipeline completed: {success_count}/{hours_back} successful")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AQI Feature Pipeline")
    parser.add_argument("--no-hopsworks", action="store_true", help="Skip Hopsworks storage")
    parser.add_argument("--no-local", action="store_true", help="Skip local storage")
    parser.add_argument("--batch", type=int, default=None, help="Run batch mode for N hours back")

    args = parser.parse_args()

    try:
        if args.batch:
            run_batch(
                hours_back=args.batch,
                use_hopsworks=not args.no_hopsworks,
                save_local=not args.no_local
            )
        else:
            success = run(
                use_hopsworks=not args.no_hopsworks,
                save_local=not args.no_local
            )
            sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)