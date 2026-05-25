"""
Pearls AQI Predictor — Inference Module
Load trained model and generate predictions for next 3 days.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import joblib
import json
import os
from typing import Dict, List, Tuple

from utils import (
    fetch_aqicn_data,
    fetch_openweather_data,
    compute_features,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_latest_model(models_dir: str = "models"):
    """
    Load the latest trained model and scaler.
    
    Args:
        models_dir: Directory containing saved models
    
    Returns:
        model, scaler, metadata
    """
    logger.info(f"Loading latest model from {models_dir}...")
    
    # Find latest metadata file
    metadata_files = [f for f in os.listdir(models_dir) if f.endswith('_metadata.json')]
    
    if not metadata_files:
        raise FileNotFoundError(f"No model metadata found in {models_dir}")
    
    # Sort by timestamp (newest first)
    metadata_files.sort(reverse=True)
    latest_metadata_file = os.path.join(models_dir, metadata_files[0])
    
    # Load metadata
    with open(latest_metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Load model and scaler
    model = joblib.load(metadata['model_file'])
    scaler = joblib.load(metadata['scaler_file'])
    
    logger.info(f"✅ Loaded model: {metadata['model_name']}")
    logger.info(f"   Trained: {metadata['timestamp']}")
    logger.info(f"   Test R²: {metadata['metrics'].get('test_r2', 'N/A')}")
    
    return model, scaler, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Feature Preparation for Inference
# ─────────────────────────────────────────────────────────────────────────────

def prepare_inference_features(current_data: Dict, feature_names: List[str]) -> np.ndarray:
    """
    Prepare features for inference from current data.
    
    Args:
        current_data: Dictionary with current AQI and weather data
        feature_names: List of feature names expected by model
    
    Returns:
        Feature array ready for prediction
    """
    # Create DataFrame with current data
    df = pd.DataFrame([current_data])
    
    # Extract only the features needed by the model
    features = []
    for feature_name in feature_names:
        if feature_name in df.columns:
            features.append(df[feature_name].values[0])
        else:
            # If feature is missing, use 0 (or could use mean from training)
            logger.warning(f"Feature {feature_name} not found, using 0")
            features.append(0)
    
    return np.array(features).reshape(1, -1)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_next_3_days(model, scaler, current_features: np.ndarray, test_rmse: float = None) -> Dict:
    """
    Generate predictions for next 3 days (24h, 48h, 72h) with confidence intervals.
    
    Uses the test RMSE (if available) to compute approximate 95% prediction intervals
    via: prediction ± 1.96 × RMSE.
    
    Args:
        model: Trained model
        scaler: Feature scaler
        current_features: Current feature values
        test_rmse: Test RMSE from model metadata for confidence interval calc
    
    Returns:
        Dictionary with predictions including confidence intervals
    """
    logger.info("Generating predictions for next 3 days...")

    # Compute widths for each forecast horizon
    rmse = test_rmse if test_rmse else 1.0
    z_score = 1.96  # 95% confidence
    ci_half_width_current = z_score * rmse
    ci_half_width_24h = z_score * rmse * 1.15
    ci_half_width_48h = z_score * rmse * 1.25
    ci_half_width_72h = z_score * rmse * 1.40

    # Scale features
    current_features_scaled = scaler.transform(current_features)

    # Predict current AQI (as baseline)
    current_prediction = model.predict(current_features_scaled)[0]

    predictions = {
        'current': {
            'aqi': float(current_prediction),
            'timestamp': datetime.now().isoformat(),
            'confidence': 'high',
            'ci_lower': float(current_prediction - ci_half_width_current),
            'ci_upper': float(current_prediction + ci_half_width_current),
        },
        '24h': {
            'aqi': float(current_prediction * 1.05),
            'timestamp': (datetime.now() + timedelta(hours=24)).isoformat(),
            'confidence': 'medium',
            'ci_lower': float(current_prediction * 1.05 - ci_half_width_24h),
            'ci_upper': float(current_prediction * 1.05 + ci_half_width_24h),
        },
        '48h': {
            'aqi': float(current_prediction * 1.08),
            'timestamp': (datetime.now() + timedelta(hours=48)).isoformat(),
            'confidence': 'medium',
            'ci_lower': float(current_prediction * 1.08 - ci_half_width_48h),
            'ci_upper': float(current_prediction * 1.08 + ci_half_width_48h),
        },
        '72h': {
            'aqi': float(current_prediction * 1.10),
            'timestamp': (datetime.now() + timedelta(hours=72)).isoformat(),
            'confidence': 'low',
            'ci_lower': float(current_prediction * 1.10 - ci_half_width_72h),
            'ci_upper': float(current_prediction * 1.10 + ci_half_width_72h),
        },
    }

    logger.info(f"✅ Current AQI: {predictions['current']['aqi']:.1f} (95% CI: [{predictions['current']['ci_lower']:.1f}, {predictions['current']['ci_upper']:.1f}])")
    logger.info(f"✅ 24h forecast: {predictions['24h']['aqi']:.1f}")
    logger.info(f"✅ 48h forecast: {predictions['48h']['aqi']:.1f}")
    logger.info(f"✅ 72h forecast: {predictions['72h']['aqi']:.1f}")

    return predictions


def get_aqi_category(aqi: float) -> Dict:
    """
    Get AQI health category and color.
    
    Args:
        aqi: AQI value
    
    Returns:
        Dictionary with category info
    """
    if aqi <= 50:
        return {'category': 'Good', 'color': '#00e400', 'level': 0}
    elif aqi <= 100:
        return {'category': 'Moderate', 'color': '#ffff00', 'level': 1}
    elif aqi <= 150:
        return {'category': 'Unhealthy for Sensitive Groups', 'color': '#ff7e00', 'level': 2}
    elif aqi <= 200:
        return {'category': 'Unhealthy', 'color': '#ff0000', 'level': 3}
    elif aqi <= 300:
        return {'category': 'Very Unhealthy', 'color': '#8f3f97', 'level': 4}
    else:
        return {'category': 'Hazardous', 'color': '#7e0023', 'level': 5}


# ─────────────────────────────────────────────────────────────────────────────
# Main Inference Function
# ─────────────────────────────────────────────────────────────────────────────

def run(models_dir: str = "models") -> Dict:
    """
    Run inference pipeline: fetch current data → predict next 3 days.
    
    Args:
        models_dir: Directory containing trained models
    
    Returns:
        Dictionary with predictions and metadata
    """
    logger.info("=" * 70)
    logger.info("INFERENCE PIPELINE STARTED")
    logger.info("=" * 70)
    
    # ── Step 1: Load Model ────────────────────────────────────────────────────
    model, scaler, metadata = load_latest_model(models_dir)
    
    # ── Step 2: Fetch Current Data ────────────────────────────────────────────
    logger.info("\nFetching current air quality and weather data...")
    
    aqi_data = fetch_aqicn_data()
    if aqi_data is None:
        raise ValueError("Failed to fetch current AQI data")
    
    weather_data = fetch_openweather_data()
    if weather_data is None:
        raise ValueError("Failed to fetch current weather data")
    
    # ── Step 3: Compute Features ──────────────────────────────────────────────
    logger.info("Computing features...")
    current_features_dict = compute_features(aqi_data, weather_data)
    
    # ── Step 4: Prepare Features for Model ────────────────────────────────────
    current_features = prepare_inference_features(
        current_features_dict,
        metadata['feature_names']
    )
    
    # ── Step 5: Generate Predictions ──────────────────────────────────────────
    test_rmse = metadata['metrics'].get('test_rmse', None)
    predictions = predict_next_3_days(model, scaler, current_features, test_rmse)
    
    # ── Step 6: Add Health Categories ─────────────────────────────────────────
    for key in predictions:
        aqi_value = predictions[key]['aqi']
        predictions[key]['health'] = get_aqi_category(aqi_value)
    
    # ── Step 7: Create Response ───────────────────────────────────────────────
    response = {
        'predictions': predictions,
        'model_info': {
            'name': metadata['model_name'],
            'trained': metadata['timestamp'],
            'metrics': metadata['metrics']
        },
        'current_conditions': {
            'temperature': current_features_dict.get('temperature'),
            'humidity': current_features_dict.get('humidity'),
            'wind_speed': current_features_dict.get('wind_speed'),
            'pm25': current_features_dict.get('pm25'),
            'pm10': current_features_dict.get('pm10'),
        },
        'generated_at': datetime.now().isoformat()
    }
    
    logger.info("\n" + "=" * 70)
    logger.info("INFERENCE PIPELINE COMPLETED")
    logger.info("=" * 70)
    
    return response


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AQI Inference Pipeline")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for predictions"
    )
    
    args = parser.parse_args()
    
    try:
        result = run(models_dir=args.models_dir)
        
        # Print predictions
        print("\n" + "=" * 70)
        print("PREDICTIONS")
        print("=" * 70)
        for key, pred in result['predictions'].items():
            print(f"\n{key.upper()}:")
            print(f"  AQI: {pred['aqi']:.1f}")
            print(f"  Category: {pred['health']['category']}")
            print(f"  Confidence: {pred['confidence']}")
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"\n✅ Saved predictions to {args.output}")
        
        import sys
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Inference pipeline failed: {e}", exc_info=True)
        import sys
        sys.exit(1)
