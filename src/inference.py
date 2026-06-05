"""
Pearls AQI Predictor — Inference Module
Load trained model from Hopsworks Model Registry and generate predictions for next 3 days.
Designed to be called from Streamlit UI.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import joblib
import json
import os
import tempfile
from typing import Dict, List, Optional

from utils import (
    fetch_aqicn_data,
    fetch_openweather_data,
    compute_features,
)

from config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    MODEL_NAME,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Hopsworks Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_from_hopsworks():
    """
    Download and load the latest registered model from Hopsworks Model Registry.

    Returns:
        model, scaler, feature_names, metrics
    """
    logger.info("Connecting to Hopsworks Model Registry...")

    import hopsworks

    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME
    )

    mr = project.get_model_registry()

    # Get the latest version of the registered model
    hw_model = mr.get_best_model(
        name=MODEL_NAME,
        metric="test_r2",
        direction="max"
    )

    logger.info(f"✅ Found model : {hw_model.name} v{hw_model.version}")
    logger.info(f"   Metrics     : {hw_model.training_metrics}")

    # Download artifacts to a temp directory
    model_dir = hw_model.download()
    logger.info(f"✅ Downloaded model artifacts to: {model_dir}")

    # Load artifacts
    model        = joblib.load(os.path.join(model_dir, "model.pkl"))
    scaler       = joblib.load(os.path.join(model_dir, "scaler.pkl"))

    with open(os.path.join(model_dir, "feature_names.json"), "r") as f:
        feature_names = json.load(f)

    with open(os.path.join(model_dir, "metrics.json"), "r") as f:
        metrics = json.load(f)

    logger.info(f"✅ Model loaded — Test R²: {metrics.get('test_r2', 'N/A')}")
    logger.info(f"   Features    : {len(feature_names)} features")

    return model, scaler, feature_names, metrics


def load_model_from_local(models_dir: str = "models"):
    """
    Fallback: load the latest model from local disk.

    Returns:
        model, scaler, feature_names, metrics
    """
    logger.info(f"Loading latest model from local disk: {models_dir}...")

    metadata_files = [f for f in os.listdir(models_dir) if f.endswith('_metadata.json')]
    if not metadata_files:
        raise FileNotFoundError(f"No model metadata found in {models_dir}")

    metadata_files.sort(reverse=True)
    latest = os.path.join(models_dir, metadata_files[0])

    with open(latest, "r") as f:
        metadata = json.load(f)

    model         = joblib.load(metadata['model_file'])
    scaler        = joblib.load(metadata['scaler_file'])
    feature_names = metadata['feature_names']
    metrics       = metadata['metrics']

    logger.info(f"✅ Loaded local model : {metadata['model_name']}")
    logger.info(f"   Trained at         : {metadata['timestamp']}")

    return model, scaler, feature_names, metrics


def load_model(models_dir: str = "models"):
    """
    Load model — tries Hopsworks first, falls back to local disk.

    Returns:
        model, scaler, feature_names, metrics
    """
    try:
        return load_model_from_hopsworks()
    except Exception as e:
        logger.warning(f"⚠️  Hopsworks load failed ({e}) — falling back to local model")
        return load_model_from_local(models_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_inference_features(current_data: Dict, feature_names: List[str]) -> np.ndarray:
    """
    Align current data dict to the exact feature order the model expects.

    Returns:
        numpy array of shape (1, n_features)
    """
    features = []
    for name in feature_names:
        val = current_data.get(name, 0)
        if val is None:
            val = 0
        features.append(float(val))

    arr = np.array(features, dtype=np.float64).reshape(1, -1)
    logger.info(f"✅ Feature vector shape: {arr.shape}")
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# AQI Category Helper
# ─────────────────────────────────────────────────────────────────────────────

def get_aqi_category(aqi: float) -> Dict:
    """Return health category, color, and level for a given AQI value."""
    aqi = max(0, aqi)
    if aqi <= 50:
        return {'category': 'Good',                            'color': '#00e400', 'level': 0,
                'advice': 'Air quality is satisfactory. Enjoy outdoor activities.'}
    elif aqi <= 100:
        return {'category': 'Moderate',                        'color': '#ffff00', 'level': 1,
                'advice': 'Acceptable air quality. Unusually sensitive people should limit prolonged outdoor exertion.'}
    elif aqi <= 150:
        return {'category': 'Unhealthy for Sensitive Groups',  'color': '#ff7e00', 'level': 2,
                'advice': 'Sensitive groups should reduce prolonged outdoor exertion.'}
    elif aqi <= 200:
        return {'category': 'Unhealthy',                       'color': '#ff0000', 'level': 3,
                'advice': 'Everyone may begin to experience health effects. Limit outdoor exertion.'}
    elif aqi <= 300:
        return {'category': 'Very Unhealthy',                  'color': '#8f3f97', 'level': 4,
                'advice': 'Health alert: everyone may experience serious health effects. Avoid outdoor activities.'}
    else:
        return {'category': 'Hazardous',                       'color': '#7e0023', 'level': 5,
                'advice': 'Health emergency. Everyone should avoid all outdoor exertion.'}


# ─────────────────────────────────────────────────────────────────────────────
# Prediction Engine
# ─────────────────────────────────────────────────────────────────────────────

def predict_next_3_days(
    model,
    scaler,
    current_features: np.ndarray,
    current_data: Dict,
    test_rmse: Optional[float] = None
) -> Dict:
    """
    Generate predictions for current, 24h, 48h, and 72h horizons.

    Confidence intervals use:  prediction ± 1.96 × RMSE × horizon_factor
    Horizon factors: current=1.0, 24h=1.15, 48h=1.25, 72h=1.40

    Args:
        model            : Trained sklearn/xgb/lgb/catboost model
        scaler           : Fitted StandardScaler
        current_features : (1, n_features) array
        current_data     : Raw feature dict (for trend modifiers)
        test_rmse        : From model metrics — used for CI width

    Returns:
        Dict with keys: current, 24h, 48h, 72h
    """
    logger.info("Generating 3-day AQI forecast...")

    rmse   = test_rmse if test_rmse else 10.0
    z      = 1.96  # 95% CI

    # Scale and predict baseline
    X_scaled    = scaler.transform(current_features)
    base_aqi    = float(model.predict(X_scaled)[0])
    base_aqi    = max(0, base_aqi)

    # Simple trend modifiers based on current conditions
    # These reflect typical AQI deterioration over time without future weather data
    humidity    = float(current_data.get('humidity',    50))
    wind_speed  = float(current_data.get('wind_speed',  5))

    # High humidity and low wind = AQI tends to worsen
    trend_factor = 1.0
    if humidity > 70 and wind_speed < 3:
        trend_factor = 1.08   # worse conditions expected
    elif wind_speed > 10:
        trend_factor = 0.97   # dispersion — slight improvement

    aqi_24h = max(0, base_aqi * trend_factor * 1.03)
    aqi_48h = max(0, base_aqi * trend_factor * 1.06)
    aqi_72h = max(0, base_aqi * trend_factor * 1.10)

    now = datetime.now()

    predictions = {
        'current': {
            'aqi':       round(base_aqi, 1),
            'timestamp': now.isoformat(),
            'label':     'Now',
            'confidence':'high',
            'ci_lower':  round(max(0, base_aqi - z * rmse * 1.00), 1),
            'ci_upper':  round(base_aqi + z * rmse * 1.00,         1),
            'health':    get_aqi_category(base_aqi),
        },
        '24h': {
            'aqi':       round(aqi_24h, 1),
            'timestamp': (now + timedelta(hours=24)).isoformat(),
            'label':     '+24 hours',
            'confidence':'medium',
            'ci_lower':  round(max(0, aqi_24h - z * rmse * 1.15), 1),
            'ci_upper':  round(aqi_24h + z * rmse * 1.15,         1),
            'health':    get_aqi_category(aqi_24h),
        },
        '48h': {
            'aqi':       round(aqi_48h, 1),
            'timestamp': (now + timedelta(hours=48)).isoformat(),
            'label':     '+48 hours',
            'confidence':'medium',
            'ci_lower':  round(max(0, aqi_48h - z * rmse * 1.25), 1),
            'ci_upper':  round(aqi_48h + z * rmse * 1.25,         1),
            'health':    get_aqi_category(aqi_48h),
        },
        '72h': {
            'aqi':       round(aqi_72h, 1),
            'timestamp': (now + timedelta(hours=72)).isoformat(),
            'label':     '+72 hours',
            'confidence':'low',
            'ci_lower':  round(max(0, aqi_72h - z * rmse * 1.40), 1),
            'ci_upper':  round(aqi_72h + z * rmse * 1.40,         1),
            'health':    get_aqi_category(aqi_72h),
        },
    }

    for key, pred in predictions.items():
        logger.info(
            f"  {key:<8} AQI: {pred['aqi']:>6.1f}  "
            f"CI: [{pred['ci_lower']:.1f}, {pred['ci_upper']:.1f}]  "
            f"→ {pred['health']['category']}"
        )

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Main Inference Function (called by Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def run(models_dir: str = "models") -> Dict:
    """
    Full inference pipeline. Call this from Streamlit.

    Steps:
        1. Load model from Hopsworks (fallback: local disk)
        2. Fetch live AQI + weather data
        3. Compute features
        4. Generate predictions for current / 24h / 48h / 72h
        5. Return structured response dict

    Args:
        models_dir: Local fallback directory if Hopsworks is unavailable

    Returns:
        {
            predictions:        { current, 24h, 48h, 72h },
            model_info:         { name, version, metrics },
            current_conditions: { temperature, humidity, ... },
            generated_at:       ISO timestamp string
        }
    """
    logger.info("=" * 70)
    logger.info("INFERENCE PIPELINE STARTED")
    logger.info("=" * 70)

    # ── Step 1: Load model ────────────────────────────────────────────────────
    logger.info("\n[1/4] Loading model...")
    model, scaler, feature_names, metrics = load_model(models_dir)

    # ── Step 2: Fetch live data ───────────────────────────────────────────────
    logger.info("\n[2/4] Fetching live AQI and weather data...")

    aqi_data = fetch_aqicn_data()
    if aqi_data is None:
        raise ValueError("❌ Failed to fetch AQI data from AQICN")

    weather_data = fetch_openweather_data()
    if weather_data is None:
        raise ValueError("❌ Failed to fetch weather data from OpenWeather")

    # ── Step 3: Compute features ──────────────────────────────────────────────
    logger.info("\n[3/4] Computing features...")
    current_data  = compute_features(aqi_data, weather_data)
    X_current     = prepare_inference_features(current_data, feature_names)

    # ── Step 4: Predict ───────────────────────────────────────────────────────
    logger.info("\n[4/4] Generating predictions...")
    test_rmse    = metrics.get('test_rmse', None)
    predictions  = predict_next_3_days(model, scaler, X_current, current_data, test_rmse)

    # ── Build response ────────────────────────────────────────────────────────
    response = {
        'predictions': predictions,
        'model_info': {
            'name':    metrics.get('model_name', MODEL_NAME),
            'metrics': {
                'test_r2':   metrics.get('test_r2',   'N/A'),
                'test_rmse': metrics.get('test_rmse',  'N/A'),
                'val_r2':    metrics.get('val_r2',    'N/A'),
                'val_rmse':  metrics.get('val_rmse',   'N/A'),
            },
        },
        'current_conditions': {
            'temperature':       current_data.get('temperature'),
            'humidity':          current_data.get('humidity'),
            'wind_speed':        current_data.get('wind_speed'),
            'pressure':          current_data.get('pressure'),
            'visibility':        current_data.get('visibility'),
            'clouds':            current_data.get('clouds'),
            'pm25':              current_data.get('pm25'),
            'pm10':              current_data.get('pm10'),
            'o3':                current_data.get('o3'),
            'no2':               current_data.get('no2'),
            'so2':               current_data.get('so2'),
            'co':                current_data.get('co'),
        },
        'generated_at': datetime.now().isoformat(),
    }

    logger.info("\n" + "=" * 70)
    logger.info("INFERENCE PIPELINE COMPLETED")
    logger.info(f"  Current AQI : {predictions['current']['aqi']}")
    logger.info(f"  Category    : {predictions['current']['health']['category']}")
    logger.info(f"  Model       : {response['model_info']['name']}")
    logger.info(f"  Test R²     : {metrics.get('test_r2', 'N/A')}")
    logger.info("=" * 70)

    return response


# ─────────────────────────────────────────────────────────────────────────────
# CLI (for testing without Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="AQI Inference Pipeline")
    parser.add_argument("--models-dir", type=str, default="models",
                        help="Local fallback directory for models")
    parser.add_argument("--output",     type=str, default=None,
                        help="Optional: save predictions to this JSON file")
    args = parser.parse_args()

    try:
        result = run(models_dir=args.models_dir)

        print("\n" + "=" * 70)
        print("PREDICTIONS")
        print("=" * 70)
        for key, pred in result['predictions'].items():
            print(f"\n  {pred['label']:<12}  AQI: {pred['aqi']:>6.1f}  "
                  f"CI: [{pred['ci_lower']:.1f}, {pred['ci_upper']:.1f}]  "
                  f"→ {pred['health']['category']}")

        print(f"\n  Model   : {result['model_info']['name']}")
        print(f"  Test R² : {result['model_info']['metrics']['test_r2']}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"✅ Saved predictions to {args.output}")

        sys.exit(0)

    except Exception as e:
        logger.error(f"Inference pipeline failed: {e}", exc_info=True)
        sys.exit(1)