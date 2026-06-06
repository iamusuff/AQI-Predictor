"""
Pearls AQI Predictor — Inference Module
Load trained model from Hopsworks Model Registry and generate predictions for next 3 days.
Designed to be called from Streamlit UI.

Multi-Horizon Forecasting Strategy:
  - Current (t+0): Use latest AQICN pollutants + OpenMeteo weather
  - 24h (t+24): Use OpenMeteo weather forecast for tomorrow
  - 48h (t+48): Use OpenMeteo weather forecast for day after tomorrow  
  - 72h (t+72): Use OpenMeteo weather forecast for 3 days ahead
  
  Since we don't have future pollutant data, we apply trend-based decay/growth
  based on current conditions and weather forecast.
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
    fetch_openmeteo_weather,
    compute_features,
    compute_time_features,
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
# Future Weather Forecasting
# ─────────────────────────────────────────────────────────────────────────────

def fetch_future_weather_forecasts(lat: float, lon: float, days: int = 3) -> Dict[str, Dict]:
    """
    Fetch weather forecasts from OpenMeteo for the next N days.
    
    Returns:
        {
            '24h': {temperature, humidity, wind_speed, pressure, ...},
            '48h': {temperature, humidity, wind_speed, pressure, ...},
            '72h': {temperature, humidity, wind_speed, pressure, ...},
        }
    """
    logger.info(f"Fetching {days}-day weather forecast from OpenMeteo...")
    
    from config import CITY_CONFIG
    
    # Get forecast data from OpenMeteo (next 7 days)
    today = pd.Timestamp.utcnow().strftime('%Y-%m-%d')
    end_date = (pd.Timestamp.utcnow() + timedelta(days=days)).strftime('%Y-%m-%d')
    
    forecast_df = fetch_openmeteo_weather(
        lat=lat,
        lon=lon,
        start_date=today,
        end_date=end_date,
        is_forecast=True  # Will add this parameter to utils.py
    )
    
    if forecast_df is None or forecast_df.empty:
        logger.warning("⚠️  Weather forecast unavailable, using persistence model")
        return None
    
    # Extract weather at 24h, 48h, 72h intervals
    forecasts = {}
    now = pd.Timestamp.utcnow().replace(tzinfo=None)  # tz-naive

    for horizon_hours in [24, 48, 72]:
        target_time = now + timedelta(hours=horizon_hours)

        forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp']).dt.tz_localize(None)
        time_diffs = (forecast_df['timestamp'] - target_time).abs()
        closest_idx = time_diffs.idxmin()
        
        forecast_row = forecast_df.loc[closest_idx]
        forecasts[f'{horizon_hours}h'] = {
            'temperature': float(forecast_row.get('temperature', 25)),
            'humidity': int(forecast_row.get('humidity', 50)),
            'wind_speed': float(forecast_row.get('wind_speed', 5)),
            'pressure': int(forecast_row.get('pressure', 1013)),
            'visibility': int(forecast_row.get('visibility', 10000)),
            'clouds': int(forecast_row.get('clouds', 50)),
        }
        
    logger.info(f"✅ Retrieved forecasts for 24h, 48h, 72h")
    return forecasts


def apply_pollutant_persistence_with_decay(current_pollutants: Dict, 
                                           weather_forecast: Dict,
                                           hours_ahead: int) -> Dict:
    """
    Predict future pollutant levels using persistence with meteorological decay.
    
    Strategy:
    - PM2.5/PM10: Decay based on wind speed increase (dispersion)
    - O3: Increases with temperature (photochemical reactions)
    - NO2/SO2/CO: Slow decay (assuming reduced emissions overnight)
    
    Args:
        current_pollutants: Current pollutant levels (pm25, pm10, o3, etc.)
        weather_forecast: Forecasted weather at target horizon
        hours_ahead: 24, 48, or 72
        
    Returns:
        Predicted pollutant dict
    """
    # Base decay factor increases with time
    base_decay = 1.0 - (0.02 * (hours_ahead / 24))  # 2% per day
    
    # Wind speed effect (stronger wind = more dispersion)
    wind_speed = weather_forecast.get('wind_speed', 5)
    wind_factor = 1.0 - (min(wind_speed, 15) / 100)  # Up to 15% reduction
    
    # Temperature effect on ozone
    temp = weather_forecast.get('temperature', 25)
    temp_factor = 1.0 + ((temp - 25) / 200)  # Warmer = more O3
    
    # Humidity effect (high humidity traps pollutants)
    humidity = weather_forecast.get('humidity', 50)
    humidity_factor = 1.0 + ((humidity - 50) / 300)
    
    predicted = {}
    
    # Particulate matter (affected by wind and humidity)
    for pm in ['pm25', 'pm10']:
        val = current_pollutants.get(pm, 0)
        predicted[pm] = max(0, val * base_decay * wind_factor * humidity_factor)
    
    # Ozone (affected by temperature)
    predicted['o3'] = max(0, current_pollutants.get('o3', 0) * base_decay * temp_factor)
    
    # Other gases (slow decay)
    for gas in ['no2', 'so2', 'co']:
        val = current_pollutants.get(gas, 0)
        predicted[gas] = max(0, val * base_decay * 0.95)
    
    return predicted


# ─────────────────────────────────────────────────────────────────────────────
# Prediction Engine (Multi-Horizon True Forecasting)
# ─────────────────────────────────────────────────────────────────────────────

def predict_next_3_days(
    model,
    scaler,
    feature_names: List[str],
    current_data: Dict,
    weather_forecasts: Optional[Dict] = None,
    test_rmse: Optional[float] = None
) -> Dict:
    """
    Generate TRUE multi-horizon predictions for current, 24h, 48h, and 72h.
    
    Each prediction uses:
    - Current pollutants (with persistence + meteorological decay)
    - Weather forecast at target horizon (from OpenMeteo)
    - Time features for target timestamp
    
    This is NOT simple trend scaling — each horizon gets its own model prediction
    with proper feature engineering.

    Args:
        model             : Trained sklearn/xgb/lgb/catboost model
        scaler            : Fitted StandardScaler
        feature_names     : List of feature names in training order
        current_data      : Current AQI + weather features dict
        weather_forecasts : {24h: {...}, 48h: {...}, 72h: {...}} or None
        test_rmse         : From model metrics — used for CI width

    Returns:
        Dict with keys: current, 24h, 48h, 72h (each with aqi, timestamp, CI, health)
    """
    logger.info("Generating TRUE multi-horizon 3-day AQI forecast...")

    rmse = test_rmse if test_rmse else 10.0
    z    = 1.96  # 95% CI
    now  = datetime.now()
    
    predictions = {}
    
    # ── Current (t+0) ─────────────────────────────────────────────────────────
    X_current = prepare_inference_features(current_data, feature_names)
    X_scaled  = scaler.transform(X_current)
    aqi_current = float(model.predict(X_scaled)[0])
    aqi_current = max(0, aqi_current)
    
    predictions['current'] = {
        'aqi':        round(aqi_current, 1),
        'timestamp':  now.isoformat(),
        'label':      'Now',
        'confidence': 'high',
        'ci_lower':   round(max(0, aqi_current - z * rmse * 1.00), 1),
        'ci_upper':   round(aqi_current + z * rmse * 1.00, 1),
        'health':     get_aqi_category(aqi_current),
    }
    
    # ── Future Horizons (t+24, t+48, t+72) ────────────────────────────────────
    for horizon_hours, label, conf_level, ci_factor in [
        (24, '+24 hours', 'medium', 1.15),
        (48, '+48 hours', 'medium', 1.25),
        (72, '+72 hours', 'low',    1.40),
    ]:
        horizon_key = f'{horizon_hours}h'
        target_time = now + timedelta(hours=horizon_hours)
        
        # Get weather forecast for this horizon
        if weather_forecasts and horizon_key in weather_forecasts:
            weather_future = weather_forecasts[horizon_key]
        else:
            # Fallback: use current weather (persistence model)
            logger.warning(f"⚠️  No forecast for {horizon_key}, using current weather")
            weather_future = {
                'temperature': current_data.get('temperature', 25),
                'humidity':    current_data.get('humidity', 50),
                'wind_speed':  current_data.get('wind_speed', 5),
                'pressure':    current_data.get('pressure', 1013),
                'visibility':  current_data.get('visibility', 10000),
                'clouds':      current_data.get('clouds', 50),
            }
        
        # Predict future pollutants (persistence with meteorological decay)
        pollutants_future = apply_pollutant_persistence_with_decay(
            current_pollutants={
                'pm25': current_data.get('pm25', 0),
                'pm10': current_data.get('pm10', 0),
                'o3':   current_data.get('o3', 0),
                'no2':  current_data.get('no2', 0),
                'so2':  current_data.get('so2', 0),
                'co':   current_data.get('co', 0),
            },
            weather_forecast=weather_future,
            hours_ahead=horizon_hours
        )
        
        # Compute time features for future timestamp
        time_features = compute_time_features(target_time)
        
        # Build complete feature dict for this horizon
        future_data = {
            **pollutants_future,
            **weather_future,
            **time_features,
        }
        
        # Predict AQI using the model
        X_future = prepare_inference_features(future_data, feature_names)
        X_scaled_future = scaler.transform(X_future)
        aqi_future = float(model.predict(X_scaled_future)[0])
        aqi_future = max(0, aqi_future)
        
        predictions[horizon_key] = {
            'aqi':        round(aqi_future, 1),
            'timestamp':  target_time.isoformat(),
            'label':      label,
            'confidence': conf_level,
            'ci_lower':   round(max(0, aqi_future - z * rmse * ci_factor), 1),
            'ci_upper':   round(aqi_future + z * rmse * ci_factor, 1),
            'health':     get_aqi_category(aqi_future),
        }

    # Log predictions
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
        2. Fetch live AQI + weather data (current)
        3. Fetch weather forecasts for next 3 days (OpenMeteo)
        4. Compute features for each horizon
        5. Generate predictions for current / 24h / 48h / 72h using the model
        6. Return structured response dict

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
    logger.info("\n[1/5] Loading model from Hopsworks Model Registry...")
    model, scaler, feature_names, metrics = load_model(models_dir)

    # ── Step 2: Fetch current live data ───────────────────────────────────────
    logger.info("\n[2/5] Fetching current AQI and weather data...")

    aqi_data = fetch_aqicn_data()
    if aqi_data is None:
        raise ValueError("❌ Failed to fetch AQI data from AQICN")

    # Fetch current weather from OpenMeteo
    from config import CITY_CONFIG
    today = pd.Timestamp.utcnow().strftime('%Y-%m-%d')
    weather_df = fetch_openmeteo_weather(
        lat=CITY_CONFIG['lat'],
        lon=CITY_CONFIG['lon'],
        start_date=today,
        end_date=today
    )
    
    if weather_df is None or weather_df.empty:
        raise ValueError("❌ Failed to fetch weather data from OpenMeteo")
    
    # Extract latest weather record
    weather_data = weather_df.iloc[-1].to_dict()
    logger.info(f"✅ Current — AQI: {aqi_data['aqi']}, Temp: {weather_data.get('temperature')}°C")

    # ── Step 3: Fetch weather forecasts ───────────────────────────────────────
    logger.info("\n[3/5] Fetching 3-day weather forecast from OpenMeteo...")
    weather_forecasts = fetch_future_weather_forecasts(
        lat=CITY_CONFIG['lat'],
        lon=CITY_CONFIG['lon'],
        days=3
    )
    
    if weather_forecasts:
        logger.info(f"✅ Weather forecasts retrieved for 24h, 48h, 72h")
    else:
        logger.warning("⚠️  Weather forecasts unavailable — using persistence model")

    # ── Step 4: Compute current features ──────────────────────────────────────
    logger.info("\n[4/5] Computing features...")
    current_data = compute_features(aqi_data, weather_data)

    # ── Step 5: Generate multi-horizon predictions ────────────────────────────
    logger.info("\n[5/5] Generating TRUE multi-horizon predictions...")
    test_rmse    = metrics.get('test_rmse', None)
    predictions  = predict_next_3_days(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        current_data=current_data,
        weather_forecasts=weather_forecasts,
        test_rmse=test_rmse
    )

    # ── Build response ────────────────────────────────────────────────────────
    response = {
        'predictions': predictions,
        'model_info': {
            'name':    metrics.get('model_name', MODEL_NAME),
            'forecast_method': 'Weather-informed' if weather_forecasts else 'Persistence model',
            'metrics': {
                'test_r2':   metrics.get('test_r2',   'N/A'),
                'test_rmse': metrics.get('test_rmse', 'N/A'),
                'test_mae':  metrics.get('test_mae',  'N/A'),
                'val_r2':    metrics.get('val_r2',    'N/A'),
                'val_rmse':  metrics.get('val_rmse',  'N/A'),
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
    logger.info(f"  Forecast    : {'Weather-informed' if weather_forecasts else 'Persistence model'}")
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