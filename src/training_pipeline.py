"""
Pearls AQI Predictor — Training Pipeline
Train multiple ML models, evaluate them, and register the best one.

Models: Ridge Regression, Random Forest, XGBoost, LSTM, GRU

Data Source: Hopsworks Feature Store (falls back to local CSV)
Model Storage: Hopsworks Model Registry
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
import os
import joblib
import json

# ML libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Deep learning (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if not TF_AVAILABLE:
    logger.warning("⚠️  TensorFlow not available — LSTM/GRU models will be skipped.")

from config import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    MODEL_NAME,
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
)


# ─────────────────────────────────────────────────────────────────────────────
# Hopsworks Connection (mirrors working feature_pipeline.py exactly)
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
        logger.warning("⚠️  Hopsworks library not installed.")
        return None, None
    except Exception as e:
        logger.error(f"❌ Failed to connect to Hopsworks: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def clean_hopsworks_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix stringified array values returned by Hopsworks fg.read()
    e.g. '[1.4101299E2]' → 142.01
    """

    float_cols = ['o3', 'no2', 'so2', 'co', 'temperature', 'wind_speed']
    int_cols   = [
        'aqi', 'pm25', 'pm10',
        'humidity', 'pressure', 'visibility', 'clouds',
        'hour', 'day_of_week', 'day_of_month', 'month',
        'season', 'is_weekend',
    ]

    # ── Float columns ─────────────────────────────────────────────────────────
    for col in float_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                       .str.replace(r'[\[\]]', '', regex=True)  # strip [ ]
                       .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

    # ── Int columns ───────────────────────────────────────────────────────────
    for col in int_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                       .str.replace(r'[\[\]]', '', regex=True)
                       .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')

    return df

def load_data_from_hopsworks(feature_store) -> pd.DataFrame:
    """
    Load training data from Hopsworks Feature Store.
    Uses the same feature group registered by feature_pipeline.py.
    """
    logger.info(f"Loading data from Hopsworks Feature Store ({FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION})...")

    try:
        fg = feature_store.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION
        )
        df = fg.read()

        # ── Print sample record to verify types ──────────────────────────────────
        logger.info("Sample record (first row) with dtypes:")
        sample = df.iloc[0]
        for col, val in sample.items():
            logger.info(f"  {col:<20} dtype={df[col].dtype!s:<12} val={val}")
        
        df = clean_hopsworks_dataframe(df)   # ← fix before anything else touches df
        logger.info(f"✅ Dtypes after clean:\n{df.dtypes.to_string()}")

        logger.info(f"✅ Loaded {len(df)} rows from Hopsworks Feature Store")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to load from Hopsworks Feature Store: {e}")
        raise


def load_data_from_csv(filepath: str = "data/features.csv") -> pd.DataFrame:
    """Fallback: load training data from local CSV."""
    logger.info(f"Loading data from local CSV: {filepath}...")
    try:
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        logger.info(f"✅ Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to load CSV: {e}")
        raise


def load_data(filepath: str = "data/features.csv") -> pd.DataFrame:
    """
    Load data — tries Hopsworks first, falls back to local CSV.
    """
    project, feature_store = connect_to_hopsworks()

    if feature_store is not None:
        try:
            return load_data_from_hopsworks(feature_store)
        except Exception as e:
            logger.warning(f"⚠️  Hopsworks load failed ({e}) — falling back to local CSV")

    return load_data_from_csv(filepath)


# ─────────────────────────────────────────────────────────────────────────────
# Data Preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_features_and_target(df: pd.DataFrame, target_col: str = 'aqi'):
    logger.info("Preparing features and target...")

    # ── Strip timezone from timestamp before exclusion ────────────────────────
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

    exclude_cols = ['timestamp', 'aqi', 'dominentpol', 'weather_main', 'weather_description']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # ── Catch any object columns that slipped through clean ───────────────────
    for col in feature_cols:
        if df[col].dtype == object:
            logger.warning(f"⚠️  Object dtype still present in '{col}' — forcing numeric")
            df[col] = (
                df[col].astype(str)
                       .str.replace(r'[\[\]]', '', regex=True)
                       .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # ── Force entire array to float64 — SHAP and scaler both require this ─────
    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)

    logger.info(f"✅ X dtype  : {X.dtype}  shape: {X.shape}")   # must say float64
    logger.info(f"✅ y dtype  : {y.dtype}")
    logger.info(f"✅ Features : {feature_cols}")

    return X, y, feature_cols


def split_data(X, y, test_size=0.15, val_size=0.15):
    """
    Split data into train/validation/test sets (temporal order preserved).
    """
    logger.info("Splitting data (temporal order preserved)...")

    n_samples = len(X)
    test_idx = int(n_samples * (1 - test_size))
    val_idx  = int(test_idx * (1 - val_size / (1 - test_size)))

    X_train, X_val, X_test = X[:val_idx], X[val_idx:test_idx], X[test_idx:]
    y_train, y_val, y_test = y[:val_idx], y[val_idx:test_idx], y[test_idx:]

    logger.info(f"✅ Train : {len(X_train)} samples ({len(X_train)/n_samples*100:.1f}%)")
    logger.info(f"✅ Val   : {len(X_val)} samples ({len(X_val)/n_samples*100:.1f}%)")
    logger.info(f"✅ Test  : {len(X_test)} samples ({len(X_test)/n_samples*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """Scale features using StandardScaler (fit on train only)."""
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)
    logger.info("✅ Features scaled (StandardScaler)")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def get_recency_weights(n_samples: int, decay: float = 0.995) -> np.ndarray:
    """Exponential decay: most recent samples have weight ~1.0, oldest ~decay^n"""
    weights = np.array([decay ** (n_samples - i) for i in range(n_samples)])
    return weights / weights.mean()

# ─────────────────────────────────────────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(y_true_train, y_pred_train, y_true_val, y_pred_val) -> dict:
    return {
        'train_rmse': float(np.sqrt(mean_squared_error(y_true_train, y_pred_train))),
        'train_mae':  float(mean_absolute_error(y_true_train, y_pred_train)),
        'train_r2':   float(r2_score(y_true_train, y_pred_train)),
        'val_rmse':   float(np.sqrt(mean_squared_error(y_true_val, y_pred_val))),
        'val_mae':    float(mean_absolute_error(y_true_val, y_pred_val)),
        'val_r2':     float(r2_score(y_true_val, y_pred_val)),
    }


def train_ridge_regression(X_train, y_train, X_val, y_val, alpha=1.0):
    logger.info("Training Ridge Regression...")
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    metrics = _compute_metrics(y_train, model.predict(X_train), y_val, model.predict(X_val))
    logger.info(f"✅ Ridge — Val RMSE: {metrics['val_rmse']:.2f}, Val R²: {metrics['val_r2']:.4f}")
    return model, metrics


def train_random_forest(X_train, y_train, X_val, y_val, n_estimators=100, max_depth=20, sample_weight=None):
    logger.info("Training Random Forest...")
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    metrics = _compute_metrics(y_train, model.predict(X_train), y_val, model.predict(X_val))
    logger.info(f"✅ Random Forest — Val RMSE: {metrics['val_rmse']:.2f}, Val R²: {metrics['val_r2']:.4f}")
    return model, metrics


def train_xgboost(X_train, y_train, X_val, y_val, n_estimators=100, max_depth=6, learning_rate=0.1, sample_weight=None):
    logger.info("Training XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weight, eval_set=[(X_val, y_val)], verbose=False)
    metrics = _compute_metrics(y_train, model.predict(X_train), y_val, model.predict(X_val))
    logger.info(f"✅ XGBoost — Val RMSE: {metrics['val_rmse']:.2f}, Val R²: {metrics['val_r2']:.4f}")
    return model, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Deep Learning (optional)
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_sequences(X, y, time_steps=24):
    n = len(X)
    if n < time_steps:
        return None, None
    X_out = np.array([X[i - time_steps:i] for i in range(time_steps, n)])
    y_out = y[time_steps:]
    return X_out, y_out


def _train_rnn(layer_class, X_train_seq, y_train_seq, X_val_seq, y_val_seq,
               epochs=50, units=64, name="RNN"):
    if not TF_AVAILABLE:
        logger.warning(f"⚠️  Skipping {name} — TensorFlow not available")
        return None, None

    logger.info(f"Training {name}...")
    time_steps = X_train_seq.shape[1]
    n_features = X_train_seq.shape[2]

    model = Sequential([
        layer_class(units, activation='tanh', return_sequences=True, input_shape=(time_steps, n_features)),
        Dropout(0.2),
        layer_class(units // 2, activation='tanh'),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs, batch_size=32,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=0,
    )

    metrics = _compute_metrics(
        y_train_seq, model.predict(X_train_seq, verbose=0).flatten(),
        y_val_seq,   model.predict(X_val_seq, verbose=0).flatten(),
    )
    logger.info(f"✅ {name} — Val RMSE: {metrics['val_rmse']:.2f}, Val R²: {metrics['val_r2']:.4f}")
    return model, metrics


def train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=50, units=64):
    return _train_rnn(LSTM, X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs, units, "LSTM")


def train_gru(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=50, units=64):
    return _train_rnn(GRU, X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs, units, "GRU")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    logger.info(f"Evaluating {model_name} on test set...")
    y_pred = model.predict(X_test)
    if hasattr(y_pred, 'flatten'):
        y_pred = y_pred.flatten()
    metrics = {
        'model':      model_name,
        'test_rmse':  float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'test_mae':   float(mean_absolute_error(y_test, y_pred)),
        'test_r2':    float(r2_score(y_test, y_pred)),
    }
    logger.info(f"   RMSE: {metrics['test_rmse']:.2f} | MAE: {metrics['test_mae']:.2f} | R²: {metrics['test_r2']:.4f}")
    return metrics


def compare_models(results: dict):
    logger.info("\n" + "=" * 70)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 70)

    comparison_data = [
        {
            'Model':     name,
            'Val RMSE':  r['metrics']['val_rmse'],
            'Val MAE':   r['metrics']['val_mae'],
            'Val R²':    r['metrics']['val_r2'],
            'Test RMSE': r['test_metrics']['test_rmse'],
            'Test MAE':  r['test_metrics']['test_mae'],
            'Test R²':   r['test_metrics']['test_r2'],
        }
        for name, r in results.items()
    ]

    comparison_df = pd.DataFrame(comparison_data).sort_values('Val RMSE')
    print("\n" + comparison_df.to_string(index=False))

    best_model_name = comparison_df.iloc[0]['Model']
    logger.info(f"\n🏆 Best Model : {best_model_name}")
    logger.info(f"   Val RMSE   : {comparison_df.iloc[0]['Val RMSE']:.2f}")
    logger.info(f"   Val R²     : {comparison_df.iloc[0]['Val R²']:.4f}")

    return best_model_name, comparison_df


# ─────────────────────────────────────────────────────────────────────────────
# SHAP Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_importance(model, X_train_scaled, feature_names, model_name: str):
    try:
        import shap
        logger.info("Computing SHAP feature importance...")

        # ── Debug input ───────────────────────────────────────────────────────
        logger.info(f"  X_train_scaled dtype  : {X_train_scaled.dtype}")
        logger.info(f"  X_train_scaled shape  : {X_train_scaled.shape}")
        logger.info(f"  Any NaN               : {np.isnan(X_train_scaled).any()}")
        logger.info(f"  Any Inf               : {np.isinf(X_train_scaled).any()}")
        logger.info(f"  model_name            : {model_name}")
        logger.info(f"  SHAP version          : {shap.__version__}")
        if model_name == "XGBoost":
            logger.info(f"  XGBoost version       : {xgb.__version__}")

        # ── Force float64 once ────────────────────────────────────────────────
        sample = np.asarray(X_train_scaled[:500], dtype=np.float64)
        logger.info(f"  Sample dtype after cast : {sample.dtype}")
        logger.info(f"  Sample shape            : {sample.shape}")

        # ── Build explainer ───────────────────────────────────────────────────
        if model_name == "XGBoost":
            logger.info("  Building TreeExplainer with model.get_booster()...")
            explainer = shap.TreeExplainer(model)
        elif model_name == "Random Forest":
            logger.info("  Building TreeExplainer for Random Forest...")
            explainer = shap.TreeExplainer(model)
        else:
            logger.info(f"  Building LinearExplainer for {model_name}...")
            explainer = shap.LinearExplainer(model, sample)

        logger.info("  Running explainer.shap_values()...")
        shap_values = explainer.shap_values(sample)
        logger.info(f"  shap_values type  : {type(shap_values)}")
        logger.info(f"  shap_values shape : {np.array(shap_values).shape}")

        importance_df = pd.DataFrame({
            'feature':          feature_names,
            'shap_importance':  np.abs(shap_values).mean(axis=0),
        }).sort_values('shap_importance', ascending=False)

        os.makedirs("models", exist_ok=True)
        importance_df.to_csv("models/shap_importance.csv", index=False)
        logger.info(f"✅ SHAP importance saved to models/shap_importance.csv")
        logger.info(f"\nTop 10 features:\n{importance_df.head(10).to_string(index=False)}")
        return importance_df

    except ImportError:
        logger.warning("⚠️  SHAP not installed — skipping (pip install shap)")
        return None
    except Exception as e:
        logger.warning(f"⚠️  SHAP failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TimeSeries Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_model(model_fn, X, y, n_splits=3, **model_kwargs):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_rmse, cv_r2 = [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        fold_scaler = StandardScaler()
        X_fold_train_s = fold_scaler.fit_transform(X_fold_train)
        X_fold_val_s   = fold_scaler.transform(X_fold_val)

        model, fold_metrics = model_fn(X_fold_train_s, y_fold_train, X_fold_val_s, y_fold_val, **model_kwargs)
        if model is not None:
            cv_rmse.append(fold_metrics['val_rmse'])
            cv_r2.append(fold_metrics['val_r2'])

    cv_scores = {}
    if cv_rmse:
        cv_scores = {
            'cv_rmse_mean': float(np.mean(cv_rmse)),
            'cv_rmse_std':  float(np.std(cv_rmse)),
            'cv_r2_mean':   float(np.mean(cv_r2)),
            'cv_r2_std':    float(np.std(cv_r2)),
        }
        logger.info(
            f"   CV {n_splits}-fold — "
            f"RMSE: {cv_scores['cv_rmse_mean']:.3f} ± {cv_scores['cv_rmse_std']:.3f}, "
            f"R²: {cv_scores['cv_r2_mean']:.4f} ± {cv_scores['cv_r2_std']:.4f}"
        )
    return cv_scores


# ─────────────────────────────────────────────────────────────────────────────
# Model Persistence & Hopsworks Model Registry
# ─────────────────────────────────────────────────────────────────────────────

def save_model_locally(model, scaler, feature_names, metrics, model_name: str,
                       output_dir: str = "models"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_file    = f"{output_dir}/{model_name}_{timestamp}.pkl"
    scaler_file   = f"{output_dir}/scaler_{timestamp}.pkl"
    metadata_file = f"{output_dir}/{model_name}_{timestamp}_metadata.json"

    joblib.dump(model,  model_file)
    joblib.dump(scaler, scaler_file)

    metadata = {
        'model_name':    model_name,
        'timestamp':     timestamp,
        'feature_names': feature_names,
        'metrics':       metrics,
        'model_file':    model_file,
        'scaler_file':   scaler_file,
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✅ Saved model    : {model_file}")
    logger.info(f"✅ Saved scaler   : {scaler_file}")
    logger.info(f"✅ Saved metadata : {metadata_file}")
    return model_file, scaler_file, metadata_file


def register_model_in_hopsworks(model, scaler, feature_names, metrics, model_name: str) -> bool:
    """
    Register the best model in Hopsworks Model Registry.
    Saves artifacts locally first, then uploads the folder — mirrors
    the pattern used in the working feature_pipeline.py.
    """
    try:
        import hopsworks

        logger.info("Connecting to Hopsworks for model registry...")
        project = hopsworks.login(
            host="eu-west.cloud.hopsworks.ai",
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT_NAME
        )

        mr = project.get_model_registry()

        # ── Save artifacts to a local staging folder ──────────────────────────
        artifact_dir = "model_artifacts"
        os.makedirs(artifact_dir, exist_ok=True)

        joblib.dump(model,  f"{artifact_dir}/model.pkl")
        joblib.dump(scaler, f"{artifact_dir}/scaler.pkl")

        with open(f"{artifact_dir}/feature_names.json", "w") as f:
            json.dump(feature_names, f)

        with open(f"{artifact_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # ── Register and upload to Hopsworks ──────────────────────────────────
        hw_model = mr.python.create_model(
            name=MODEL_NAME,
            description=f"AQI Predictor — {model_name}",
            metrics={
                "val_rmse":  round(float(metrics.get("val_rmse",  0)), 4),
                "val_r2":    round(float(metrics.get("val_r2",    0)), 4),
                "test_rmse": round(float(metrics.get("test_rmse", 0)), 4),
                "test_r2":   round(float(metrics.get("test_r2",   0)), 4),
            },
        )
        hw_model.save(artifact_dir)   # uploads entire folder to registry

        logger.info(f"✅ Model '{MODEL_NAME}' registered in Hopsworks Model Registry")
        return True

    except ImportError:
        logger.warning("⚠️  Hopsworks library not installed — skipping model registry")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to register model in Hopsworks: {e}", exc_info=True)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(data_path: str = "data/features.csv", save_models: bool = True,
        use_hopsworks: bool = True):
    logger.info("=" * 70)
    logger.info("TRAINING PIPELINE STARTED")
    logger.info("=" * 70)

    # ── Step 1: Load data from Hopsworks (fallback to CSV) ────────────────────
    logger.info("\n[1/8] Loading data...")
    df = load_data(filepath=data_path)

    # Sort by timestamp — critical for temporal split correctness
    df = df.sort_values('timestamp').reset_index(drop=True)
    logger.info(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    # ── Step 2: Prepare features and target ───────────────────────────────────
    logger.info("\n[2/8] Preparing features and target...")
    X, y, feature_names = prepare_features_and_target(df)

    # ── Step 3: Split data ────────────────────────────────────────────────────
    logger.info("\n[3/8] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # ── Step 4: Scale features ────────────────────────────────────────────────
    logger.info("\n[4/8] Scaling features...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test
    )

    # ── Step 5: TimeSeries cross-validation (Ridge only for speed) ────────────
    logger.info("\n[5/8] Time-Series Cross-Validation (3-fold, Ridge)...")
    cv_results = {}
    cv_results['Ridge Regression'] = cross_validate_model(train_ridge_regression, X, y)

    # ── Step 6: Train all models ──────────────────────────────────────────────
    logger.info("\n[6/8] Training models...")
    results = {}

    # 1. Ridge Regression
    logger.info("\n  [1/5] Ridge Regression")
    sample_weights = get_recency_weights(len(X_train_scaled))
    ridge_model, ridge_metrics = train_ridge_regression(X_train_scaled, y_train, X_val_scaled, y_val)
    results['Ridge Regression'] = {
        'model':        ridge_model,
        'metrics':      {**ridge_metrics, **cv_results.get('Ridge Regression', {})},
        'test_metrics': evaluate_model(ridge_model, X_test_scaled, y_test, "Ridge Regression"),
    }

    # 2. Random Forest
    logger.info("\n  [2/5] Random Forest")
    rf_model, rf_metrics = train_random_forest(X_train_scaled, y_train, X_val_scaled, y_val, sample_weight=sample_weights)
    results['Random Forest'] = {
        'model':        rf_model,
        'metrics':      rf_metrics,
        'test_metrics': evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest"),
    }

    # 3. XGBoost
    logger.info("\n  [3/5] XGBoost")
    xgb_model, xgb_metrics = train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val, sample_weight=sample_weights)
    results['XGBoost'] = {
        'model':        xgb_model,
        'metrics':      xgb_metrics,
        'test_metrics': evaluate_model(xgb_model, X_test_scaled, y_test, "XGBoost"),
    }

    # 4 & 5. LSTM / GRU (optional)
    X_train_seq = X_val_seq = X_test_seq = None   # init for GRU reuse
    y_train_seq = y_val_seq = y_test_seq = None

    if TF_AVAILABLE:
        X_train_seq, y_train_seq = _prepare_sequences(X_train_scaled, y_train)
        X_val_seq,   y_val_seq   = _prepare_sequences(X_val_scaled,   y_val)
        X_test_seq,  y_test_seq  = _prepare_sequences(X_test_scaled,  y_test)

        if X_train_seq is not None:
            logger.info("\n  [4/5] LSTM")
            lstm_model, lstm_metrics = train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
            if lstm_model is not None:
                lstm_pred = lstm_model.predict(X_test_seq, verbose=0).flatten()
                results['LSTM'] = {
                    'model':   lstm_model,
                    'metrics': lstm_metrics,
                    'test_metrics': {
                        'test_rmse': float(np.sqrt(mean_squared_error(y_test_seq, lstm_pred))),
                        'test_mae':  float(mean_absolute_error(y_test_seq, lstm_pred)),
                        'test_r2':   float(r2_score(y_test_seq, lstm_pred)),
                    },
                }

            logger.info("\n  [5/5] GRU")
            gru_model, gru_metrics = train_gru(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
            if gru_model is not None:
                gru_pred = gru_model.predict(X_test_seq, verbose=0).flatten()
                results['GRU'] = {
                    'model':   gru_model,
                    'metrics': gru_metrics,
                    'test_metrics': {
                        'test_rmse': float(np.sqrt(mean_squared_error(y_test_seq, gru_pred))),
                        'test_mae':  float(mean_absolute_error(y_test_seq, gru_pred)),
                        'test_r2':   float(r2_score(y_test_seq, gru_pred)),
                    },
                }
        else:
            logger.warning("⚠️  Not enough data for LSTM/GRU sequences (need ≥ 24 samples)")
    else:
        logger.warning("⚠️  TensorFlow not available — skipping LSTM and GRU")

    # ── Step 7: Compare and select best model ─────────────────────────────────
    logger.info("\n[7/8] Comparing models...")
    best_model_name, comparison_df = compare_models(results)

    best_model   = results[best_model_name]['model']
    best_metrics = {
        **results[best_model_name]['metrics'],
        **results[best_model_name]['test_metrics'],
    }

    # ── SHAP feature importance ───────────────────────────────────────────────
    compute_shap_importance(best_model, X_train_scaled, feature_names, best_model_name)

    # ── Step 8: Save / register model ─────────────────────────────────────────
    logger.info("\n[8/8] Saving and registering model...")

    if save_models:
        os.makedirs("models", exist_ok=True)
        save_model_locally(
            best_model, scaler, feature_names, best_metrics,
            best_model_name.lower().replace(' ', '_')
        )
        comparison_df.to_csv("models/model_comparison.csv", index=False)
        logger.info("✅ Saved model comparison: models/model_comparison.csv")

    if use_hopsworks:
        register_model_in_hopsworks(
            best_model, scaler, feature_names, best_metrics, best_model_name
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING PIPELINE COMPLETED")
    logger.info(f"  Best Model : {best_model_name}")
    logger.info(f"  Test RMSE  : {results[best_model_name]['test_metrics']['test_rmse']:.2f}")
    logger.info(f"  Test R²    : {results[best_model_name]['test_metrics']['test_r2']:.4f}")
    logger.info(f"  Hopsworks  : {'✅ Registered' if use_hopsworks else '⚠️  Skipped'}")
    logger.info("=" * 70)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AQI Training Pipeline")
    parser.add_argument("--data",        type=str,  default="data/features.csv",
                        help="Path to local CSV (used only if Hopsworks fails)")
    parser.add_argument("--no-save",     action="store_true", help="Don't save trained models locally")
    parser.add_argument("--no-hopsworks",action="store_true", help="Skip Hopsworks model registry")

    args = parser.parse_args()

    try:
        results = run(
            data_path=args.data,
            save_models=not args.no_save,
            use_hopsworks=not args.no_hopsworks,
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        sys.exit(1)