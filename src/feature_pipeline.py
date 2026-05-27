"""
Pearls AQI Predictor — Feature Pipeline
Fetches real-time air quality & weather data, engineers features, and stores in Hopsworks.

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
    fetch_openweather_data,
    compute_features,
    validate_feature_data,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Hopsworks Integration
# ─────────────────────────────────────────────────────────────────────────────

def connect_to_hopsworks():
    """
    Connect to Hopsworks Feature Store.
    
    Returns:
        Tuple of (project, feature_store) or (None, None) if connection fails
    """
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
        if feature_store:
            debug_feature_store(feature_store)
        # feature_groups = feature_store.get_feature_groups()
        # for fg in feature_groups:

        #     if fg.name == FEATURE_GROUP_NAME:

        #         print(f"Deleting {fg.name} v{fg.version}")

        #         fg.delete()

        # print("✅ Cleanup completed")

        return project, feature_store
        
    except ImportError:
        logger.warning("⚠️  Hopsworks library not installed. Features will be saved locally.")
        return None, None
    except Exception as e:
        logger.error(f"❌ Failed to connect to Hopsworks: {e}")
        return None, None

def debug_feature_store(feature_store):
    """Paste this call right after connect_to_hopsworks() to diagnose."""
    logger.info(f"Feature store type: {type(feature_store)}")
    logger.info(f"Feature store value: {feature_store}")

    try:
        fg = feature_store.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION
        )
        logger.info(f"get_feature_group result type : {type(fg)}")
        logger.info(f"get_feature_group result value: {fg}")
        logger.info(f"Has insert method: {hasattr(fg, 'insert')}")
    except Exception as e:
        logger.info(f"get_feature_group raised: {type(e).__name__}: {e}")

    try:
        fg2 = feature_store.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            primary_key=["timestamp"],
            event_time="timestamp",
            online_enabled=False,
        )
        logger.info(f"get_or_create result type : {type(fg2)}")
        logger.info(f"get_or_create result value: {fg2}")
        logger.info(f"Has insert method: {hasattr(fg2, 'insert')}")
    except Exception as e:
        logger.info(f"get_or_create raised: {type(e).__name__}: {e}")

def get_or_create_feature_group(feature_store, sample_df=None):
    """
    Get existing feature group or create a new one.
    """
    feature_group = None

    # ── Try fetching existing group first ────────────────────────────────────
    try:
        fg = feature_store.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION
        )

        # Some Hopsworks versions return None instead of raising when not found
        if fg is not None:
            logger.info(
                f"✅ Found existing feature group: "
                f"{FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}"
            )
            return fg
        else:
            logger.warning("⚠️  get_feature_group returned None — will attempt creation.")

    except Exception as e:
        logger.warning(f"⚠️  get_feature_group raised exception — will attempt creation. Reason: {e}")

    # ── Create / upsert feature group ─────────────────────────────────────────
    try:
        logger.info(
            f"Creating feature group: "
            f"{FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}"
        )

        feature_group = feature_store.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            primary_key=["timestamp"],
            event_time="timestamp",
            description="AQI prediction features for Karachi",
            online_enabled=False,
        )

        if feature_group is None:
            logger.error("❌ get_or_create_feature_group returned None.")
            return None

        logger.info(
            f"✅ Feature group ready: "
            f"{feature_group.name} v{feature_group.version}"
        )
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

        if not hasattr(feature_group, 'insert'):
            logger.error(f"❌ feature_group has no insert method. Type: {type(feature_group)}")
            return False

        logger.info(f"✅ Feature group confirmed: {feature_group.name} v{feature_group.version}")

        features_df = features_df.copy()

        # ── Enforce exact types so schema never mismatches ────────────────────
        # These must be int (bigint in Hopsworks)
        int_columns = [
            'aqi', 'pm25', 'pm10',
            'humidity', 'pressure', 'visibility',
            'hour', 'day_of_week', 'month', 'season',
            'is_weekend', 'is_peak_hour'
        ]

        # These must be float (double in Hopsworks)
        float_columns = [
            'o3', 'no2', 'so2', 'co',
            'temperature', 'wind_speed',
            'dew_point', 'wind_gust',
            'temp_humidity_index', 'aqi_rolling_mean',
            'aqi_rolling_std', 'wind_chill'
        ]

        for col in int_columns:
            if col in features_df.columns:
                features_df[col] = features_df[col].astype('int64')

        for col in float_columns:
            if col in features_df.columns:
                features_df[col] = features_df[col].astype('float64')

        # ── Timestamp must be naive UTC ───────────────────────────────────────
        features_df["timestamp"] = pd.to_datetime(
            features_df["timestamp"], utc=True
        ).dt.tz_localize(None)

        logger.info(f"Dtypes after cast:\n{features_df.dtypes}")
        logger.info(f"Inserting {len(features_df)} row(s)...")

        feature_group.insert(
            features_df,
            write_options={"wait_for_job": False}
        )

        logger.info(f"✅ Inserted {len(features_df)} row(s) into '{FEATURE_GROUP_NAME}'")
        return True

    except Exception as e:
        logger.error(f"❌ Insert failed: {e}", exc_info=True)
        return False

# ─────────────────────────────────────────────────────────────────────────────
# Local Storage (Fallback)
# ─────────────────────────────────────────────────────────────────────────────

def save_features_locally(features_df: pd.DataFrame, filepath: str = "data/features.csv"):
    """
    Save features to local CSV file (fallback when Hopsworks is unavailable).
    
    Args:
        features_df: DataFrame with features
        filepath: Path to save CSV file
    """
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Ensure timestamp is timezone-aware UTC
    features_df['timestamp'] = pd.to_datetime(
        features_df['timestamp'],
        utc=True
    )

    # Append to existing file or create new one
    if os.path.exists(filepath):

        existing_df = pd.read_csv(
            filepath,
            parse_dates=['timestamp']
        )

        # Normalize existing timestamps too
        existing_df['timestamp'] = pd.to_datetime(
            existing_df['timestamp'],
            utc=True
        )

        combined_df = pd.concat(
            [existing_df, features_df],
            ignore_index=True
        )

        # Remove duplicates
        combined_df = combined_df.drop_duplicates(
            subset=['timestamp'],
            keep='last'
        )

        # Sort safely
        combined_df = combined_df.sort_values('timestamp')

        combined_df.to_csv(filepath, index=False)

        logger.info(
            f"✅ Appended features to {filepath} "
            f"(total rows: {len(combined_df)})"
        )

    else:

        features_df.to_csv(filepath, index=False)

        logger.info(f"✅ Saved features to {filepath}")

# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(use_hopsworks: bool = True, save_local: bool = True) -> bool:
    """
    Run the feature pipeline: fetch data → compute features → store.
    
    Args:
        use_hopsworks: Whether to store features in Hopsworks (default: True)
        save_local: Whether to save features locally as backup (default: True)
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info("FEATURE PIPELINE STARTED")
    logger.info(f"City: {CITY_CONFIG['name']}, {CITY_CONFIG['country']}")
    logger.info(f"Timestamp: {pd.Timestamp.utcnow().isoformat()}")
    logger.info("=" * 60)
    
    # ── Step 1: Fetch raw data ────────────────────────────────────────────────
    logger.info("\n[1/5] Fetching air quality data from AQICN...")
    aqi_data = fetch_aqicn_data()
    
    if aqi_data is None:
        logger.error("❌ Failed to fetch AQI data. Aborting pipeline.")
        return False
    
    logger.info(f"✅ AQI: {aqi_data['aqi']}, PM2.5: {aqi_data['pm25']}, Dominant: {aqi_data['dominentpol']}")
    
    logger.info("\n[2/5] Fetching weather data from OpenWeather...")
    weather_data = fetch_openweather_data()
    
    if weather_data is None:
        logger.error("❌ Failed to fetch weather data. Aborting pipeline.")
        return False
    
    logger.info(f"✅ Temp: {weather_data['temperature']}°C, Humidity: {weather_data['humidity']}%, Wind: {weather_data['wind_speed']} m/s")
    
    # ── Step 2: Compute features ──────────────────────────────────────────────
    logger.info("\n[3/5] Computing features...")
    features = compute_features(aqi_data, weather_data)
    
    # Validate features
    is_valid, error_msg = validate_feature_data(features)
    if not is_valid:
        logger.error(f"❌ Feature validation failed: {error_msg}")
        return False
    
    logger.info(f"✅ Generated {len(features)} features")
    
    # Convert to DataFrame
    features_df = pd.DataFrame([features])
    
    # Display sample features
    logger.info("\nSample features:")
    sample_cols = ['timestamp', 'aqi', 'pm25', 'temperature', 'humidity', 'hour', 'day_of_week', 'season']
    for col in sample_cols:
        if col in features_df.columns:
            logger.info(f"  {col}: {features_df[col].values[0]}")
    
    # ── Step 3: Store in Hopsworks ────────────────────────────────────────────
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
    
    # ── Step 4: Save locally (backup) ─────────────────────────────────────────
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
    """
    Run feature pipeline for multiple past hours (useful for catching up missed runs).
    
    Args:
        hours_back: Number of hours to backfill
        use_hopsworks: Whether to store in Hopsworks
        save_local: Whether to save locally
    """
    logger.info(f"Running batch feature pipeline for last {hours_back} hours...")
    
    # Note: This is a simplified version. For true historical backfill,
    # we would need to use historical weather APIs (see backfill.py)
    
    success_count = 0
    for i in range(hours_back):
        logger.info(f"\n--- Processing hour {i+1}/{hours_back} ---")
        
        if run(use_hopsworks=use_hopsworks, save_local=save_local):
            success_count += 1
        
        # Sleep to avoid rate limiting (if running multiple times)
        if i < hours_back - 1:
            import time
            time.sleep(2)  # 2 seconds between requests
    
    logger.info(f"\nBatch pipeline completed: {success_count}/{hours_back} successful")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AQI Feature Pipeline")
    parser.add_argument(
        "--no-hopsworks",
        action="store_true",
        help="Skip Hopsworks storage (local only)"
    )
    parser.add_argument(
        "--no-local",
        action="store_true",
        help="Skip local storage (Hopsworks only)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Run batch mode for N hours back"
    )
    
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
