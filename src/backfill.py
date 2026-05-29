"""
Pearls AQI Predictor — Historical Backfill Feature Pipeline
Fetches 90 days of hourly air quality + weather data from OpenMeteo (free, no API key),
engineers features, and stores in Hopsworks.

Strategy:
  - OpenMeteo Air Quality API  → PM2.5, PM10, O3, NO2, SO2, CO (historical)
  - OpenMeteo Archive API      → Temperature, Humidity, Wind, Pressure, Visibility, Clouds (historical)
  - AQI is computed from pollutants using EPA breakpoints (AQICN not used — real-time only)
  - Both APIs support arbitrary date ranges and return hourly data

Run:
  python feature_pipeline.py --backfill 90        # last 90 days
  python feature_pipeline.py --backfill 30        # last 30 days
  python feature_pipeline.py                      # current hour (live mode)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
import logging
import sys
import time

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
    compute_aqi_target,
    validate_feature_data,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# AQI Computation from Pollutants (EPA Standard)
# ─────────────────────────────────────────────────────────────────────────────

# EPA AQI breakpoints: (concentration_low, concentration_high, aqi_low, aqi_high)
EPA_BREAKPOINTS = {
    'pm25': [
        (0.0,   12.0,  0,   50),
        (12.1,  35.4,  51,  100),
        (35.5,  55.4,  101, 150),
        (55.5,  150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ],
    'pm10': [
        (0,   54,  0,   50),
        (55,  154, 51,  100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500),
    ],
    'o3': [  # µg/m³ — OpenMeteo returns µg/m³
        (0,    54,   0,   50),
        (55,   70,   51,  100),
        (71,   85,   101, 150),
        (86,   105,  151, 200),
        (106,  200,  201, 300),
    ],
    'no2': [  # µg/m³
        (0,    53,   0,   50),
        (54,   100,  51,  100),
        (101,  360,  101, 150),
        (361,  649,  151, 200),
        (650,  1249, 201, 300),
        (1250, 2049, 301, 500),
    ],
}


def _interpolate_aqi(concentration: float, breakpoints: list) -> Optional[float]:
    """Linear interpolation of AQI from EPA breakpoints."""
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= concentration <= bp_hi:
            aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + aqi_lo
            return round(aqi)
    if concentration > breakpoints[-1][1]:
        return 500
    return None


def compute_aqi_from_pollutants(row: pd.Series) -> int:
    """
    Compute overall AQI as the maximum sub-index across available pollutants.
    Falls back to 0 if no pollutant data available.
    """
    sub_indices = []
    for pollutant, bps in EPA_BREAKPOINTS.items():
        val = row.get(pollutant)
        if pd.notna(val) and val >= 0:
            idx = _interpolate_aqi(float(val), bps)
            if idx is not None:
                sub_indices.append(idx)
    return int(max(sub_indices)) if sub_indices else 0


# ─────────────────────────────────────────────────────────────────────────────
# Historical Data Fetching
# ─────────────────────────────────────────────────────────────────────────────

# OpenMeteo archive API has a ~5-day lag; cap end_date accordingly
_OPENMETEO_LAG_DAYS = 5


def _safe_end_date() -> datetime:
    return datetime.utcnow() - timedelta(days=_OPENMETEO_LAG_DAYS)


def fetch_historical_features(days_back: int = 90) -> Optional[pd.DataFrame]:
    """
    Fetch and merge hourly weather + AQI data for the past `days_back` days.

    Returns a merged DataFrame with one row per hour, or None on failure.
    """
    end_dt   = _safe_end_date()
    start_dt = end_dt - timedelta(days=days_back)

    start_str = start_dt.strftime('%Y-%m-%d')
    end_str   = end_dt.strftime('%Y-%m-%d')

    logger.info(f"Fetching historical data: {start_str} → {end_str}  ({days_back} days)")

    # ── Weather ───────────────────────────────────────────────────────────────
    logger.info("  [1/2] Fetching OpenMeteo weather (archive)...")
    weather_df = fetch_openmeteo_weather(
        lat=CITY_CONFIG['lat'],
        lon=CITY_CONFIG['lon'],
        start_date=start_str,
        end_date=end_str,
    )
    if weather_df is None or weather_df.empty:
        logger.error("❌ Failed to fetch historical weather data.")
        return None
    logger.info(f"  ✅ Weather rows: {len(weather_df)}")

    # ── Air Quality ───────────────────────────────────────────────────────────
    logger.info("  [2/2] Fetching OpenMeteo air quality (historical)...")
    aqi_df = fetch_openmeteo_aqi(
        lat=CITY_CONFIG['lat'],
        lon=CITY_CONFIG['lon'],
        start_date=start_str,
        end_date=end_str,
    )
    if aqi_df is None or aqi_df.empty:
        logger.error("❌ Failed to fetch historical AQI data.")
        return None
    logger.info(f"  ✅ AQI rows: {len(aqi_df)}")

    # ── Normalise timestamps before merge ─────────────────────────────────────
    # Both APIs return timezone-aware or naive strings depending on `timezone=auto`.
    # Coerce everything to UTC-naive for a clean merge key.
    for df in (weather_df, aqi_df):
        df['timestamp'] = (
            pd.to_datetime(df['timestamp'], utc=True)
              .dt.tz_localize(None)
        )

    # ── Merge on timestamp (inner join keeps only hours with both datasets) ────
    merged = pd.merge(weather_df, aqi_df, on='timestamp', how='inner')
    logger.info(f"  ✅ Merged rows (inner): {len(merged)}")

    if merged.empty:
        logger.error("❌ Merge produced empty DataFrame — timestamp mismatch between APIs.")
        return None

    merged = merged.sort_values('timestamp').reset_index(drop=True)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the merged weather+AQI DataFrame and produce the final feature DataFrame
    matching the Hopsworks schema exactly.

    Columns produced
    ────────────────
    timestamp        : datetime64[ns]  (UTC-naive, used as primary key + event time)

    Pollutants (float64):
      pm25, pm10, o3, no2, so2, co

    Weather (float64 / int64):
      temperature, wind_speed        → float64
      humidity, pressure, visibility, clouds  → int64

    Computed (int64):
      aqi                            → computed from EPA breakpoints
      hour, day_of_week, day_of_month, month, season, is_weekend
    """
    df = raw_df.copy()

    logger.info(f"Engineering features for {len(df)} rows...")

    # ── 1. Compute AQI from pollutants ────────────────────────────────────────
    df['aqi'] = df.apply(compute_aqi_from_pollutants, axis=1)

    # ── 2. Time features ──────────────────────────────────────────────────────
    time_features = df['timestamp'].apply(
        lambda ts: pd.Series(compute_time_features(ts))
    )
    df = pd.concat([df, time_features], axis=1)

    # ── 3. Cast pollutants → float64 ─────────────────────────────────────────
    float_cols = ['o3', 'no2', 'so2', 'co', 'temperature', 'wind_speed']  # pm25, pm10 removed

    int_cols = [
        'aqi', 'pm25', 'pm10',   # ← add these
        'humidity', 'pressure', 'visibility', 'clouds',
        'hour', 'day_of_week', 'day_of_month', 'month', 'season', 'is_weekend',
    ]

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
        else:
            df[col] = np.nan

    # ── 4. Cast weather + time columns → int64 ───────────────────────────────
    # Hopsworks bigint columns cannot be null; fill with 0.
    
    for col in int_cols:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors='coerce')
                  .fillna(0)
                  .astype('int64')
            )
        else:
            df[col] = np.int64(0)

    # ── 5. Timestamp: ensure UTC-naive datetime64[ns] ─────────────────────────
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)

    # ── 6. Drop any unexpected extra columns ─────────────────────────────────
    keep_cols = [
        'timestamp',
        'aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co',
        'temperature', 'humidity', 'wind_speed', 'pressure', 'visibility', 'clouds',
        'hour', 'day_of_week', 'day_of_month', 'month', 'season', 'is_weekend',
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    # ── 7. Drop rows missing critical fields ──────────────────────────────────
    critical = ['timestamp', 'aqi', 'pm25', 'temperature', 'humidity']
    before = len(df)
    df = df.dropna(subset=[c for c in critical if c in df.columns])
    dropped = before - len(df)
    if dropped:
        logger.warning(f"  ⚠️  Dropped {dropped} rows with null in critical columns")

    # ── 8. Deduplicate on timestamp ───────────────────────────────────────────
    df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

    logger.info(f"  ✅ Final feature rows: {len(df)}")
    logger.info(f"  dtypes:\n{df.dtypes.to_string()}")
    return df


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


def get_or_create_feature_group(feature_store, sample_df: Optional[pd.DataFrame] = None):
    try:
        fg = feature_store.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
        )
        if fg is not None:
            logger.info(f"✅ Found existing feature group: {FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}")
            return fg
    except Exception as e:
        logger.warning(f"⚠️  get_feature_group raised exception — will attempt creation. Reason: {e}")

    try:
        logger.info(f"Creating feature group: {FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}")
        feature_group = feature_store.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            primary_key=["timestamp"],
            event_time="timestamp",
            description="AQI prediction features for Karachi (historical + live)",
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


def insert_to_hopsworks(
    features_df: pd.DataFrame,
    feature_store,
    chunk_size: int = 168,   # 1 week of hourly data per chunk
) -> bool:
    """
    Insert features into Hopsworks in chunks to avoid request size limits.
    For large backfills (2 160 rows for 90 days), chunking is essential.
    """
    feature_group = get_or_create_feature_group(feature_store, sample_df=features_df)
    if feature_group is None:
        logger.error("❌ feature_group is None — aborting insert.")
        return False

    total_rows = len(features_df)
    chunks = [features_df.iloc[i:i + chunk_size] for i in range(0, total_rows, chunk_size)]
    logger.info(f"Inserting {total_rows} rows in {len(chunks)} chunk(s) of up to {chunk_size} rows...")

    success_count = 0
    for idx, chunk in enumerate(chunks, start=1):
        try:
            feature_group.insert(
                chunk,
                write_options={"wait_for_job": True},
                validation_options={"run_validation": False},
            )
            logger.info(f"  ✅ Chunk {idx}/{len(chunks)} inserted ({len(chunk)} rows)")
            success_count += 1

            # Small delay between chunks to avoid rate-limiting
            if idx < len(chunks):
                time.sleep(1)

        except Exception as e:
            logger.error(f"  ❌ Chunk {idx}/{len(chunks)} failed: {e}", exc_info=True)
            # Continue with next chunk rather than aborting entirely

    logger.info(f"Insert complete: {success_count}/{len(chunks)} chunks succeeded")
    return success_count == len(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Local Storage (Backup / Fallback)
# ─────────────────────────────────────────────────────────────────────────────

def save_features_locally(features_df: pd.DataFrame, filepath: str = "data/features.csv"):
    import os

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    features_df = features_df.copy()
    features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])

    if os.path.exists(filepath):
        existing_df = pd.read_csv(filepath, parse_dates=['timestamp'])
        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])

        combined_df = pd.concat([existing_df, features_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        combined_df.to_csv(filepath, index=False)
        logger.info(f"✅ Appended to {filepath} (total rows: {len(combined_df)})")
    else:
        features_df.to_csv(filepath, index=False)
        logger.info(f"✅ Saved {len(features_df)} rows to {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Entry Points
# ─────────────────────────────────────────────────────────────────────────────

def run_backfill(
    days_back: int = 90,
    use_hopsworks: bool = True,
    save_local: bool = True,
) -> bool:
    """
    Main entry point for historical backfill.
    Fetches `days_back` days of hourly data (~24 * days_back rows).
    """
    logger.info("=" * 60)
    logger.info("HISTORICAL BACKFILL PIPELINE STARTED")
    logger.info(f"City      : {CITY_CONFIG['name']}, {CITY_CONFIG['country']}")
    logger.info(f"Days back : {days_back}  (≈{days_back * 24} hourly rows expected)")
    logger.info(f"UTC now   : {pd.Timestamp.utcnow().isoformat()}")
    logger.info("=" * 60)

    # ── Step 1: Fetch raw historical data ─────────────────────────────────────
    logger.info("\n[1/3] Fetching historical data from OpenMeteo...")
    raw_df = fetch_historical_features(days_back=days_back)
    if raw_df is None or raw_df.empty:
        logger.error("❌ Failed to fetch historical data. Aborting.")
        return False

    # ── Step 2: Engineer features ─────────────────────────────────────────────
    logger.info("\n[2/3] Engineering features...")
    features_df = engineer_features(raw_df)

    if features_df.empty:
        logger.error("❌ Feature engineering produced empty DataFrame. Aborting.")
        return False

    logger.info(f"\nSample row:\n{features_df.iloc[0].to_dict()}")
    logger.info(f"\nDate range: {features_df['timestamp'].min()} → {features_df['timestamp'].max()}")

    # ── Step 3: Store ─────────────────────────────────────────────────────────
    hopsworks_success = False
    local_success     = False

    if use_hopsworks:
        logger.info("\n[3/3] Storing in Hopsworks Feature Store (chunked)...")
        project, feature_store = connect_to_hopsworks()
        if feature_store is not None:
            hopsworks_success = insert_to_hopsworks(features_df, feature_store)
        else:
            logger.warning("⚠️  Skipping Hopsworks (connection failed)")
    else:
        logger.info("\n[3/3] Skipping Hopsworks (--no-hopsworks flag set)")

    if save_local:
        logger.info("Saving features locally (backup)...")
        save_features_locally(features_df)
        local_success = True
    else:
        logger.info("Skipping local save (--no-local flag set)")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("BACKFILL PIPELINE COMPLETED")
    logger.info(f"  Rows inserted   : {len(features_df)}")
    logger.info(f"  Date range      : {features_df['timestamp'].min()} → {features_df['timestamp'].max()}")
    logger.info(f"  Hopsworks       : {'✅ Success' if hopsworks_success else '⚠️  Skipped/Failed'}")
    logger.info(f"  Local Storage   : {'✅ Success' if local_success else '⚠️  Skipped'}")
    logger.info("=" * 60)

    return True


def run_live(use_hopsworks: bool = True, save_local: bool = True) -> bool:
    """
    Live/hourly mode: fetch data for the current hour only.
    Uses 2-day window to ensure the latest available hour is captured
    (OpenMeteo archive has a ~5-day lag, so real-time is not possible via this API).
    For true real-time data, use the AQICN-based pipeline instead.
    """
    logger.info("=" * 60)
    logger.info("LIVE FEATURE PIPELINE STARTED")
    logger.info(f"City  : {CITY_CONFIG['name']}, {CITY_CONFIG['country']}")
    logger.info(f"UTC   : {pd.Timestamp.utcnow().isoformat()}")
    logger.info("=" * 60)

    # Fetch last 2 days and take the most recent row
    raw_df = fetch_historical_features(days_back=2)
    if raw_df is None or raw_df.empty:
        logger.error("❌ Failed to fetch data.")
        return False

    features_df = engineer_features(raw_df)
    if features_df.empty:
        logger.error("❌ Feature engineering produced empty DataFrame.")
        return False

    # Keep only the latest hour
    latest_row = features_df.tail(1)
    logger.info(f"Latest available timestamp: {latest_row['timestamp'].values[0]}")

    hopsworks_success = False
    if use_hopsworks:
        project, feature_store = connect_to_hopsworks()
        if feature_store is not None:
            hopsworks_success = insert_to_hopsworks(latest_row, feature_store, chunk_size=1)

    if save_local:
        save_features_locally(latest_row)

    logger.info("\n" + "=" * 60)
    logger.info("LIVE PIPELINE COMPLETED")
    logger.info(f"  Hopsworks : {'✅ Success' if hopsworks_success else '⚠️  Skipped/Failed'}")
    logger.info(f"  Local     : {'✅ Success' if save_local else '⚠️  Skipped'}")
    logger.info("=" * 60)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AQI Feature Pipeline — historical backfill or live hourly mode"
    )
    parser.add_argument(
        "--backfill",
        type=int,
        default=None,
        metavar="DAYS",
        help="Run historical backfill for the last N days (e.g. --backfill 90)"
    )
    parser.add_argument("--no-hopsworks", action="store_true", help="Skip Hopsworks storage")
    parser.add_argument("--no-local",     action="store_true", help="Skip local CSV storage")

    args = parser.parse_args()

    try:
        if args.backfill:
            success = run_backfill(
                days_back=args.backfill,
                use_hopsworks=not args.no_hopsworks,
                save_local=not args.no_local,
            )
        else:
            success = run_live(
                use_hopsworks=not args.no_hopsworks,
                save_local=not args.no_local,
            )
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)