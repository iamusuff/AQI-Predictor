"""
Pearls AQI Predictor — Training Pipeline
Train multiple ML models, evaluate them, and register the best one.

Models: Ridge Regression, Random Forest, XGBoost
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
import os
import joblib
import json

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from config import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    MODEL_NAME,
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
)

# ─────────────────────────────────────────────────────────────────────────────
# Data Preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath: str = "data/features.csv") -> pd.DataFrame:
    """
    Load training data from CSV or Hopsworks Feature Store.
    
    Args:
        filepath: Path to local CSV file
    
    Returns:
        DataFrame with features
    """
    logger.info(f"Loading data from {filepath}...")
    
    try:
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        logger.info(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        raise


def prepare_features_and_target(df: pd.DataFrame, target_col: str = 'aqi'):
    """
    Prepare features (X) and target (y) for training.
    
    Args:
        df: DataFrame with all data
        target_col: Name of target column
    
    Returns:
        X (features), y (target), feature_names
    """
    logger.info("Preparing features and target...")
    
    # Define feature columns (exclude non-feature columns)
    exclude_cols = ['timestamp', 'aqi', 'dominentpol', 'weather_main', 'weather_description']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Extract features and target
    X = df[feature_cols].values
    y = df[target_col].values
    
    logger.info(f"✅ Features: {len(feature_cols)} columns")
    logger.info(f"✅ Target: {target_col}")
    logger.info(f"✅ Samples: {len(X)}")
    
    return X, y, feature_cols


def split_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split data into train/validation/test sets (respecting temporal order).
    
    Args:
        X: Features
        y: Target
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info("Splitting data (temporal order preserved)...")
    
    # Calculate split indices
    n_samples = len(X)
    test_idx = int(n_samples * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size / (1 - test_size)))
    
    # Split data
    X_train = X[:val_idx]
    X_val = X[val_idx:test_idx]
    X_test = X[test_idx:]
    
    y_train = y[:val_idx]
    y_val = y[val_idx:test_idx]
    y_test = y[test_idx:]
    
    logger.info(f"✅ Train: {len(X_train)} samples ({len(X_train)/n_samples*100:.1f}%)")
    logger.info(f"✅ Val:   {len(X_val)} samples ({len(X_val)/n_samples*100:.1f}%)")
    logger.info(f"✅ Test:  {len(X_test)} samples ({len(X_test)/n_samples*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """
    Scale features using StandardScaler.
    
    Args:
        X_train, X_val, X_test: Feature arrays
    
    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled, scaler
    """
    logger.info("Scaling features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("✅ Features scaled (StandardScaler)")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler



# ─────────────────────────────────────────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────────────────────────────────────────

def train_ridge_regression(X_train, y_train, X_val, y_val, alpha=1.0):
    """
    Train Ridge Regression model (baseline).
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        alpha: Regularization strength
    
    Returns:
        model, metrics
    """
    logger.info("Training Ridge Regression...")
    
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'val_mae': mean_absolute_error(y_val, y_val_pred),
        'val_r2': r2_score(y_val, y_val_pred),
    }
    
    logger.info(f"✅ Ridge - Val RMSE: {metrics['val_rmse']:.2f}, Val R²: {metrics['val_r2']:.4f}")
    
    return model, metrics


def train_random_forest(X_train, y_train, X_val, y_val, n_estimators=100, max_depth=20):
    """
    Train Random Forest model.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_estimators: Number of trees
        max_depth: Maximum tree depth
    
    Returns:
        model, metrics
    """
    logger.info("Training Random Forest...")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'val_mae': mean_absolute_error(y_val, y_val_pred),
        'val_r2': r2_score(y_val, y_val_pred),
    }
    
    logger.info(f"✅ Random Forest - Val RMSE: {metrics['val_rmse']:.2f}, Val R²: {metrics['val_r2']:.4f}")
    
    return model, metrics


def train_xgboost(X_train, y_train, X_val, y_val, n_estimators=100, max_depth=6, learning_rate=0.1):
    """
    Train XGBoost model.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
    
    Returns:
        model, metrics
    """
    logger.info("Training XGBoost...")
    
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'val_mae': mean_absolute_error(y_val, y_val_pred),
        'val_r2': r2_score(y_val, y_val_pred),
    }
    
    logger.info(f"✅ XGBoost - Val RMSE: {metrics['val_rmse']:.2f}, Val R²: {metrics['val_r2']:.4f}")
    
    return model, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Model Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        X_test, y_test: Test data
        model_name: Name of the model
    
    Returns:
        test_metrics
    """
    logger.info(f"Evaluating {model_name} on test set...")
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'model': model_name,
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'test_mae': mean_absolute_error(y_test, y_pred),
        'test_r2': r2_score(y_test, y_pred),
    }
    
    logger.info(f"✅ {model_name} Test Results:")
    logger.info(f"   RMSE: {metrics['test_rmse']:.2f}")
    logger.info(f"   MAE:  {metrics['test_mae']:.2f}")
    logger.info(f"   R²:   {metrics['test_r2']:.4f}")
    
    return metrics


def compare_models(results: dict):
    """
    Compare all models and select the best one.
    
    Args:
        results: Dictionary of model results
    
    Returns:
        best_model_name, comparison_df
    """
    logger.info("\n" + "=" * 70)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 70)
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Val RMSE': result['metrics']['val_rmse'],
            'Val MAE': result['metrics']['val_mae'],
            'Val R²': result['metrics']['val_r2'],
            'Test RMSE': result['test_metrics']['test_rmse'],
            'Test MAE': result['test_metrics']['test_mae'],
            'Test R²': result['test_metrics']['test_r2'],
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Val RMSE')
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Select best model (lowest validation RMSE)
    best_model_name = comparison_df.iloc[0]['Model']
    logger.info(f"\n🏆 Best Model: {best_model_name}")
    logger.info(f"   Val RMSE: {comparison_df.iloc[0]['Val RMSE']:.2f}")
    logger.info(f"   Val R²:   {comparison_df.iloc[0]['Val R²']:.4f}")
    
    return best_model_name, comparison_df


# ─────────────────────────────────────────────────────────────────────────────
# Model Persistence & Hopsworks Registry
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, scaler, feature_names, metrics, model_name: str, output_dir: str = "models"):
    """
    Save model, scaler, and metadata to disk.
    
    Args:
        model: Trained model
        scaler: Feature scaler
        feature_names: List of feature names
        metrics: Model metrics
        model_name: Name of the model
        output_dir: Directory to save models
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{output_dir}/{model_name}_{timestamp}.pkl"
    scaler_filename = f"{output_dir}/scaler_{timestamp}.pkl"
    metadata_filename = f"{output_dir}/{model_name}_{timestamp}_metadata.json"
    
    # Save model
    joblib.dump(model, model_filename)
    logger.info(f"✅ Saved model: {model_filename}")
    
    # Save scaler
    joblib.dump(scaler, scaler_filename)
    logger.info(f"✅ Saved scaler: {scaler_filename}")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'timestamp': timestamp,
        'feature_names': feature_names,
        'metrics': metrics,
        'model_file': model_filename,
        'scaler_file': scaler_filename,
    }
    
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✅ Saved metadata: {metadata_filename}")
    
    return model_filename, scaler_filename, metadata_filename


def register_model_in_hopsworks(model, model_name: str, metrics: dict, feature_names: list):
    """
    Register the trained model in Hopsworks Model Registry.
    
    Args:
        model: Trained model
        model_name: Name of the model
        metrics: Model performance metrics
        feature_names: List of feature names
    """
    try:
        import hopsworks
        
        logger.info("Connecting to Hopsworks for model registry...")
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT_NAME
        )
        
        model_registry = project.get_model_registry()
        
        # Create model in registry
        model_registry.python.create_model(
            name=MODEL_NAME,
            description=f"AQI Predictor - {model_name}",
            metrics=metrics,
            input_example=feature_names[:5],  # Sample of feature names
        )
        
        logger.info(f"✅ Model '{MODEL_NAME}' registered in Hopsworks Model Registry")
        return True
        
    except ImportError:
        logger.warning("⚠️  Hopsworks library not installed — skipping model registry")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to register model in Hopsworks: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# TimeSeries Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_model(model_fn, X, y, n_splits=3, **model_kwargs):
    """
    Perform time-series cross-validation.
    
    Args:
        model_fn: Model training function (e.g., train_ridge_regression)
        X: Feature array
        y: Target array
        n_splits: Number of CV splits
        **model_kwargs: Additional kwargs for the model function
    
    Returns:
        cv_scores dict with mean/std of RMSE and R²
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_rmse = []
    cv_r2 = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        # Scale within fold
        fold_scaler = StandardScaler()
        X_fold_train_s = fold_scaler.fit_transform(X_fold_train)
        X_fold_val_s = fold_scaler.transform(X_fold_val)
        
        model, fold_metrics = model_fn(X_fold_train_s, y_fold_train, X_fold_val_s, y_fold_val, **model_kwargs)
        if model is not None:
            cv_rmse.append(fold_metrics['val_rmse'])
            cv_r2.append(fold_metrics['val_r2'])
        else:
            logger.warning(f"Fold {fold+1}: model returned None, skipping")
    
    cv_scores = {}
    if cv_rmse:
        cv_scores = {
            'cv_rmse_mean': float(np.mean(cv_rmse)),
            'cv_rmse_std': float(np.std(cv_rmse)),
            'cv_r2_mean': float(np.mean(cv_r2)),
            'cv_r2_std': float(np.std(cv_r2)),
        }
        logger.info(f"   CV {n_splits}-fold — RMSE: {cv_scores['cv_rmse_mean']:.3f} ± {cv_scores['cv_rmse_std']:.3f}, R²: {cv_scores['cv_r2_mean']:.4f} ± {cv_scores['cv_r2_std']:.4f}")
    
    return cv_scores


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(data_path: str = "data/features.csv", save_models: bool = True, use_hopsworks: bool = False):
    """
    Run the complete training pipeline.
    
    Args:
        data_path: Path to training data
        save_models: Whether to save trained models
    
    Returns:
        results dictionary
    """
    logger.info("=" * 70)
    logger.info("TRAINING PIPELINE STARTED")
    logger.info("=" * 70)
    
    # ── Step 1: Load Data ─────────────────────────────────────────────────────
    df = load_data(data_path)
    
    # ── Step 2: Prepare Features and Target ──────────────────────────────────
    X, y, feature_names = prepare_features_and_target(df)
    
    # ── Step 3: Split Data ────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # ── Step 4: Scale Features ────────────────────────────────────────────────
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test
    )
    
    # ── Step 4b: TimeSeries Cross-Validation ───────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("TIME-SERIES CROSS-VALIDATION (3-fold)")
    logger.info("=" * 70)
    
    cv_results = {}
    cv_results['Ridge Regression'] = cross_validate_model(train_ridge_regression, X, y)
    
    # ── Step 5: Train Models ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING MODELS")
    logger.info("=" * 70)
    
    results = {}
    
    # 1. Ridge Regression (Baseline)
    logger.info("\n[1/3] Ridge Regression (Baseline)")
    ridge_model, ridge_metrics = train_ridge_regression(
        X_train_scaled, y_train, X_val_scaled, y_val
    )
    ridge_test_metrics = evaluate_model(ridge_model, X_test_scaled, y_test, "Ridge Regression")
    results['Ridge Regression'] = {
        'model': ridge_model,
        'metrics': {**ridge_metrics, **cv_results.get('Ridge Regression', {})},
        'test_metrics': ridge_test_metrics
    }
    
    # 2. Random Forest
    logger.info("\n[2/3] Random Forest")
    rf_model, rf_metrics = train_random_forest(
        X_train_scaled, y_train, X_val_scaled, y_val
    )
    rf_test_metrics = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
    results['Random Forest'] = {
        'model': rf_model,
        'metrics': rf_metrics,
        'test_metrics': rf_test_metrics
    }
    
    # 3. XGBoost
    logger.info("\n[3/3] XGBoost")
    xgb_model, xgb_metrics = train_xgboost(
        X_train_scaled, y_train, X_val_scaled, y_val
    )
    xgb_test_metrics = evaluate_model(xgb_model, X_test_scaled, y_test, "XGBoost")
    results['XGBoost'] = {
        'model': xgb_model,
        'metrics': xgb_metrics,
        'test_metrics': xgb_test_metrics
    }
    
    # ── Step 6: Compare Models ────────────────────────────────────────────────
    best_model_name, comparison_df = compare_models(results)
    
    # ── Step 7: Save Best Model ───────────────────────────────────────────────
    if save_models:
        logger.info("\n" + "=" * 70)
        logger.info("SAVING MODELS")
        logger.info("=" * 70)
        
        best_model = results[best_model_name]['model']
        best_metrics = {
            **results[best_model_name]['metrics'],
            **results[best_model_name]['test_metrics']
        }
        
        model_file, scaler_file, metadata_file = save_model(
            best_model, scaler, feature_names, best_metrics, 
            best_model_name.lower().replace(' ', '_')
        )
        
        # Save comparison results
        comparison_df.to_csv('models/model_comparison.csv', index=False)
        logger.info("✅ Saved model comparison: models/model_comparison.csv")
    
    # ── Step 8: Register in Hopsworks (optional) ──────────────────────────────
    if use_hopsworks:
        logger.info("\n" + "=" * 70)
        logger.info("REGISTERING MODEL IN HOPSWORKS")
        logger.info("=" * 70)
        register_model_in_hopsworks(
            results[best_model_name]['model'],
            best_model_name,
            best_metrics,
            feature_names,
        )
    
    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING PIPELINE COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"Test RMSE: {results[best_model_name]['test_metrics']['test_rmse']:.2f}")
    logger.info(f"Test R²:   {results[best_model_name]['test_metrics']['test_r2']:.4f}")
    logger.info("=" * 70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AQI Training Pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="data/features.csv",
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save trained models"
    )
    parser.add_argument(
        "--hopsworks",
        action="store_true",
        help="Register best model in Hopsworks Model Registry"
    )
    
    args = parser.parse_args()
    
    try:
        results = run(
            data_path=args.data,
            save_models=not args.no_save,
            use_hopsworks=args.hopsworks
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        sys.exit(1)
