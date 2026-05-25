"""
Pearls AQI Predictor - SHAP Analysis & Explainability
Phase 6: Model interpretability using SHAP values

Features:
- Global feature importance (SHAP summary plots)
- Individual prediction explanations (SHAP force plots)
- Feature interactions (SHAP dependence plots)
- Alert system for hazardous AQI levels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import json
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("notebooks")
OUTPUT_DIR.mkdir(exist_ok=True)

# AQI Health Categories and Alert Thresholds
AQI_CATEGORIES = {
    'Good': (0, 50, 'green'),
    'Moderate': (51, 100, 'yellow'),
    'Unhealthy for Sensitive': (101, 150, 'orange'),
    'Unhealthy': (151, 200, 'red'),
    'Very Unhealthy': (201, 300, 'purple'),
    'Hazardous': (301, 500, 'maroon')
}

ALERT_THRESHOLDS = {
    100: {'level': 'YELLOW', 'category': 'Unhealthy for Sensitive Groups', 'action': 'Sensitive groups should reduce outdoor activity'},
    150: {'level': 'ORANGE', 'category': 'Unhealthy', 'action': 'Everyone should reduce prolonged outdoor exertion'},
    200: {'level': 'RED', 'category': 'Very Unhealthy', 'action': 'Everyone should avoid prolonged outdoor exertion'},
    300: {'level': 'MAROON', 'category': 'Hazardous', 'action': 'URGENT: Everyone should avoid all outdoor activities'}
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def load_latest_model(model_dir: str = "models"):
    """
    Load the most recent trained model and scaler.
    
    Returns:
        model, scaler, metadata
    """
    logger.info("Loading latest model...")
    
    model_dir = Path(model_dir)
    
    # Find latest metadata file
    metadata_files = sorted(model_dir.glob("*_metadata.json"))
    if not metadata_files:
        raise FileNotFoundError("No model metadata found")
    
    latest_metadata_file = metadata_files[-1]
    
    # Load metadata
    with open(latest_metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Load model and scaler
    model = joblib.load(metadata['model_file'])
    scaler = joblib.load(metadata['scaler_file'])
    
    logger.info(f"✅ Loaded model: {metadata['model_name']}")
    logger.info(f"   Test R²: {metadata['metrics']['test_r2']:.4f}")
    logger.info(f"   Test RMSE: {metadata['metrics']['test_rmse']:.2f}")
    
    return model, scaler, metadata


def load_test_data(data_path: str = "data/features.csv", test_size: float = 0.15):
    """
    Load and split data to get test set.
    
    Returns:
        X_test, y_test, feature_names
    """
    logger.info("Loading test data...")
    
    # Load data
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    
    # Define features
    exclude_cols = ['timestamp', 'aqi', 'dominentpol', 'weather_main', 'weather_description']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Extract features and target
    X = df[feature_cols].values
    y = df['aqi'].values
    
    # Get test set (last 15%)
    test_idx = int(len(X) * (1 - test_size))
    X_test = X[test_idx:]
    y_test = y[test_idx:]
    
    logger.info(f"✅ Test set: {len(X_test)} samples, {len(feature_cols)} features")
    
    return X_test, y_test, feature_cols


def get_aqi_category(aqi_value: float) -> tuple:
    """
    Get AQI health category and color.
    
    Args:
        aqi_value: AQI value
    
    Returns:
        (category_name, color)
    """
    for category, (min_val, max_val, color) in AQI_CATEGORIES.items():
        if min_val <= aqi_value <= max_val:
            return category, color
    return 'Hazardous', 'maroon'


def check_alert_threshold(aqi_value: float) -> dict:
    """
    Check if AQI exceeds alert thresholds.
    
    Args:
        aqi_value: AQI value
    
    Returns:
        Alert information or None
    """
    for threshold in sorted(ALERT_THRESHOLDS.keys(), reverse=True):
        if aqi_value > threshold:
            return {
                'aqi': aqi_value,
                'threshold': threshold,
                **ALERT_THRESHOLDS[threshold]
            }
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SHAP Analysis Functions
# ─────────────────────────────────────────────────────────────────────────────

def create_shap_explainer(model, X_background, model_type='linear'):
    """
    Create SHAP explainer for the model.
    
    Args:
        model: Trained model
        X_background: Background dataset for SHAP
        model_type: Type of model ('linear', 'tree', 'kernel')
    
    Returns:
        explainer
    """
    logger.info(f"Creating SHAP explainer ({model_type})...")
    
    if model_type == 'linear':
        explainer = shap.LinearExplainer(model, X_background)
    elif model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict, X_background)
    
    logger.info("✅ SHAP explainer created")
    return explainer


def compute_shap_values(explainer, X_test, max_samples=None):
    """
    Compute SHAP values for test set.
    
    Args:
        explainer: SHAP explainer
        X_test: Test data
        max_samples: Maximum number of samples to compute (for speed)
    
    Returns:
        shap_values
    """
    logger.info("Computing SHAP values...")
    
    if max_samples and len(X_test) > max_samples:
        X_sample = X_test[:max_samples]
        logger.info(f"   Using {max_samples} samples (for speed)")
    else:
        X_sample = X_test
    
    shap_values = explainer.shap_values(X_sample)
    
    logger.info(f"✅ SHAP values computed for {len(X_sample)} samples")
    return shap_values, X_sample


# ─────────────────────────────────────────────────────────────────────────────
# Visualization Functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_summary(shap_values, X_test, feature_names, output_path=None):
    """
    Create SHAP summary plot (global feature importance).
    
    Args:
        shap_values: SHAP values
        X_test: Test data
        feature_names: Feature names
        output_path: Path to save plot
    """
    logger.info("Creating SHAP summary plot...")
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot - Global Feature Importance', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('SHAP Value (impact on AQI prediction)', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {output_path}")
    
    plt.close()


def plot_shap_bar(shap_values, feature_names, output_path=None):
    """
    Create SHAP bar plot (mean absolute SHAP values).
    
    Args:
        shap_values: SHAP values
        feature_names: Feature names
        output_path: Path to save plot
    """
    logger.info("Creating SHAP bar plot...")
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP|': mean_abs_shap
    }).sort_values('Mean |SHAP|', ascending=True)
    
    # Plot top 20 features
    top_n = 20
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['Feature'].tail(top_n), importance_df['Mean |SHAP|'].tail(top_n), color='steelblue')
    plt.xlabel('Mean |SHAP Value|', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Features by Mean Absolute SHAP Value', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {output_path}")
    
    plt.close()
    
    return importance_df


def plot_shap_force(explainer, shap_values, X_test, feature_names, sample_idx=0, output_path=None):
    """
    Create SHAP force plot for individual prediction.
    
    Args:
        explainer: SHAP explainer
        shap_values: SHAP values
        X_test: Test data
        feature_names: Feature names
        sample_idx: Index of sample to explain
        output_path: Path to save plot
    """
    logger.info(f"Creating SHAP force plot for sample {sample_idx}...")
    
    # Create force plot
    shap.initjs()
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[sample_idx],
        X_test[sample_idx],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    
    plt.title(f'SHAP Force Plot - Sample {sample_idx}', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {output_path}")
    
    plt.close()


def plot_shap_dependence(shap_values, X_test, feature_names, feature_idx, interaction_idx=None, output_path=None):
    """
    Create SHAP dependence plot (feature interactions).
    
    Args:
        shap_values: SHAP values
        X_test: Test data
        feature_names: Feature names
        feature_idx: Index of feature to plot
        interaction_idx: Index of interaction feature (auto if None)
        output_path: Path to save plot
    """
    feature_name = feature_names[feature_idx]
    logger.info(f"Creating SHAP dependence plot for {feature_name}...")
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature_idx,
        shap_values,
        X_test,
        feature_names=feature_names,
        interaction_index=interaction_idx,
        show=False
    )
    plt.title(f'SHAP Dependence Plot - {feature_name}', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {output_path}")
    
    plt.close()


def plot_shap_waterfall(explainer, shap_values, X_test, feature_names, sample_idx=0, output_path=None):
    """
    Create SHAP waterfall plot for individual prediction.
    
    Args:
        explainer: SHAP explainer
        shap_values: SHAP values
        X_test: Test data
        feature_names: Feature names
        sample_idx: Index of sample to explain
        output_path: Path to save plot
    """
    logger.info(f"Creating SHAP waterfall plot for sample {sample_idx}...")
    
    # Create Explanation object
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X_test[sample_idx],
        feature_names=feature_names
    )
    
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(explanation, show=False)
    plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {output_path}")
    
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Alert System Functions
# ─────────────────────────────────────────────────────────────────────────────

def generate_alert_report(predictions, output_path=None):
    """
    Generate alert report for predictions.
    
    Args:
        predictions: List of AQI predictions
        output_path: Path to save report
    
    Returns:
        alert_report
    """
    logger.info("Generating alert report...")
    
    alerts = []
    for i, aqi in enumerate(predictions):
        alert = check_alert_threshold(aqi)
        if alert:
            category, color = get_aqi_category(aqi)
            alerts.append({
                'sample_idx': i,
                'aqi': aqi,
                'category': category,
                'alert_level': alert['level'],
                'threshold': alert['threshold'],
                'action': alert['action']
            })
    
    alert_df = pd.DataFrame(alerts)
    
    if len(alert_df) > 0:
        logger.info(f"⚠️  {len(alert_df)} alerts triggered!")
        logger.info(f"   Alert levels: {alert_df['alert_level'].value_counts().to_dict()}")
    else:
        logger.info("✅ No alerts triggered (all predictions below 100 AQI)")
    
    if output_path:
        alert_df.to_csv(output_path, index=False)
        logger.info(f"✅ Saved alert report: {output_path}")
    
    return alert_df


def plot_alert_distribution(predictions, y_test, output_path=None):
    """
    Plot distribution of predictions with alert thresholds.
    
    Args:
        predictions: Model predictions
        y_test: True values
        output_path: Path to save plot
    """
    logger.info("Creating alert distribution plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Predictions with thresholds
    ax1 = axes[0]
    ax1.hist(predictions, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add threshold lines
    for threshold, info in ALERT_THRESHOLDS.items():
        ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax1.text(threshold + 5, ax1.get_ylim()[1] * 0.9, f"{info['level']}\n({threshold})", 
                fontsize=9, color='red', fontweight='bold')
    
    ax1.set_xlabel('Predicted AQI', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Prediction Distribution with Alert Thresholds', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Plot 2: Actual vs Predicted with categories
    ax2 = axes[1]
    scatter = ax2.scatter(y_test, predictions, alpha=0.6, c=predictions, cmap='RdYlGn_r', s=30)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Add threshold lines
    for threshold in ALERT_THRESHOLDS.keys():
        ax2.axhline(threshold, color='red', linestyle=':', linewidth=1, alpha=0.5)
        ax2.axvline(threshold, color='red', linestyle=':', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('True AQI', fontsize=12)
    ax2.set_ylabel('Predicted AQI', fontsize=12)
    ax2.set_title('Actual vs Predicted AQI', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.colorbar(scatter, ax=ax2, label='Predicted AQI')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {output_path}")
    
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main Analysis Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_shap_analysis():
    """
    Run complete SHAP analysis pipeline.
    """
    logger.info("=" * 70)
    logger.info("SHAP ANALYSIS STARTED")
    logger.info("=" * 70)
    
    # ── Step 1: Load Model and Data ───────────────────────────────────────────
    model, scaler, metadata = load_latest_model()
    X_test, y_test, feature_names = load_test_data()
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Get predictions
    predictions = model.predict(X_test_scaled)
    
    # ── Step 2: Create SHAP Explainer ─────────────────────────────────────────
    # Use subset of data as background (for speed)
    background_size = min(100, len(X_test_scaled))
    X_background = X_test_scaled[:background_size]
    
    explainer = create_shap_explainer(model, X_background, model_type='linear')
    
    # ── Step 3: Compute SHAP Values ───────────────────────────────────────────
    # Use subset for SHAP computation (for speed)
    max_samples = min(200, len(X_test_scaled))
    shap_values, X_sample = compute_shap_values(explainer, X_test_scaled, max_samples=max_samples)
    
    # ── Step 4: Generate Visualizations ───────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 70)
    
    # 1. SHAP Summary Plot (Global Importance)
    plot_shap_summary(
        shap_values, X_sample, feature_names,
        output_path=OUTPUT_DIR / "shap_01_summary_plot.png"
    )
    
    # 2. SHAP Bar Plot (Mean Absolute SHAP)
    importance_df = plot_shap_bar(
        shap_values, feature_names,
        output_path=OUTPUT_DIR / "shap_02_bar_plot.png"
    )
    
    # 3. SHAP Waterfall Plot (Individual Predictions)
    # Plot for 3 different samples (low, medium, high AQI)
    sample_indices = [0, len(X_sample)//2, len(X_sample)-1]
    for i, idx in enumerate(sample_indices):
        plot_shap_waterfall(
            explainer, shap_values, X_sample, feature_names,
            sample_idx=idx,
            output_path=OUTPUT_DIR / f"shap_03_waterfall_sample_{i+1}.png"
        )
    
    # 4. SHAP Dependence Plots (Top 5 Features)
    top_features = importance_df.tail(5)['Feature'].tolist()
    for i, feature in enumerate(top_features):
        feature_idx = feature_names.index(feature)
        plot_shap_dependence(
            shap_values, X_sample, feature_names,
            feature_idx=feature_idx,
            output_path=OUTPUT_DIR / f"shap_04_dependence_{feature}.png"
        )
    
    # ── Step 5: Alert System Analysis ─────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("ALERT SYSTEM ANALYSIS")
    logger.info("=" * 70)
    
    # Generate alert report
    alert_df = generate_alert_report(
        predictions,
        output_path=OUTPUT_DIR / "shap_alert_report.csv"
    )
    
    # Plot alert distribution
    plot_alert_distribution(
        predictions, y_test,
        output_path=OUTPUT_DIR / "shap_05_alert_distribution.png"
    )
    
    # ── Step 6: Generate Summary Report ───────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING SUMMARY REPORT")
    logger.info("=" * 70)
    
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("SHAP ANALYSIS SUMMARY REPORT")
    summary_lines.append("=" * 70)
    summary_lines.append("")
    
    # Model Info
    summary_lines.append("MODEL INFORMATION")
    summary_lines.append("-" * 70)
    summary_lines.append(f"Model: {metadata['model_name']}")
    summary_lines.append(f"Test R²: {metadata['metrics']['test_r2']:.4f}")
    summary_lines.append(f"Test RMSE: {metadata['metrics']['test_rmse']:.2f}")
    summary_lines.append(f"Test MAE: {metadata['metrics']['test_mae']:.2f}")
    summary_lines.append("")
    
    # Feature Importance
    summary_lines.append("TOP 10 FEATURES (by Mean |SHAP|)")
    summary_lines.append("-" * 70)
    for i, row in importance_df.tail(10).iloc[::-1].iterrows():
        summary_lines.append(f"{row['Feature']:30s} {row['Mean |SHAP|']:8.4f}")
    summary_lines.append("")
    
    # Alert Statistics
    summary_lines.append("ALERT STATISTICS")
    summary_lines.append("-" * 70)
    summary_lines.append(f"Total Predictions: {len(predictions)}")
    summary_lines.append(f"Alerts Triggered: {len(alert_df)}")
    summary_lines.append(f"Alert Rate: {len(alert_df)/len(predictions)*100:.1f}%")
    summary_lines.append("")
    
    if len(alert_df) > 0:
        summary_lines.append("Alert Level Distribution:")
        for level, count in alert_df['alert_level'].value_counts().items():
            summary_lines.append(f"  {level}: {count} ({count/len(alert_df)*100:.1f}%)")
        summary_lines.append("")
        
        summary_lines.append("AQI Category Distribution (Alerts Only):")
        for category, count in alert_df['category'].value_counts().items():
            summary_lines.append(f"  {category}: {count}")
    else:
        summary_lines.append("No alerts triggered - all predictions below 100 AQI")
    
    summary_lines.append("")
    summary_lines.append("=" * 70)
    summary_lines.append("GENERATED FILES")
    summary_lines.append("=" * 70)
    summary_lines.append("1. shap_01_summary_plot.png - Global feature importance")
    summary_lines.append("2. shap_02_bar_plot.png - Mean absolute SHAP values")
    summary_lines.append("3. shap_03_waterfall_sample_*.png - Individual predictions (3 samples)")
    summary_lines.append("4. shap_04_dependence_*.png - Feature interactions (top 5 features)")
    summary_lines.append("5. shap_05_alert_distribution.png - Alert threshold analysis")
    summary_lines.append("6. shap_06_cross_model_comparison.png - Cross-model SHAP comparison")
    summary_lines.append("7. shap_alert_report.csv - Detailed alert information")
    summary_lines.append("8. shap_summary_report.txt - This report")
    summary_lines.append("")
    summary_lines.append("=" * 70)
    
    # Save summary report
    summary_text = "\n".join(summary_lines)
    summary_path = OUTPUT_DIR / "shap_summary_report.txt"
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    logger.info(f"✅ Saved summary report: {summary_path}")
    
    # Print summary
    print("\n" + summary_text)
    
    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("SHAP ANALYSIS COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Visualizations: {len(list(OUTPUT_DIR.glob('shap_*.png')))} PNG files")
    logger.info(f"Reports: 2 files (CSV + TXT)")
    logger.info(f"Top Feature: {importance_df.iloc[-1]['Feature']}")
    logger.info(f"Alerts: {len(alert_df)} / {len(predictions)} predictions")
    logger.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Model SHAP Comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_model_shap_comparison():
    """
    Compute SHAP values for all available models and compare feature importance.
    """
    logger.info("=" * 70)
    logger.info("CROSS-MODEL SHAP COMPARISON")
    logger.info("=" * 70)

    model_dir = Path("models")
    metadata_files = sorted(model_dir.glob("*_metadata.json"))
    if not metadata_files:
        logger.warning("No model metadata found for cross-model comparison")
        return

    # Load data once
    X_test, y_test, feature_names = load_test_data()

    # Find models with different prefixes
    model_prefixes = set()
    for mf in metadata_files:
        name = mf.name.replace("_metadata.json", "")
        prefix = name.rsplit("_", 1)[0]  # Remove timestamp suffix
        model_prefixes.add(prefix)

    all_importance = {}
    loaded_models = 0

    for prefix in sorted(model_prefixes):
        # Find the latest metadata for this model prefix
        candidates = [mf for mf in metadata_files if mf.name.startswith(prefix)]
        if not candidates:
            continue
        latest_md = candidates[-1]

        with open(latest_md, 'r') as f:
            metadata = json.load(f)

        model = joblib.load(metadata['model_file'])
        scaler = joblib.load(metadata['scaler_file'])

        X_test_scaled = scaler.transform(X_test)

        # Determine explainer type
        model_name = metadata['model_name']
        if 'ridge' in model_name or 'linear' in model_name:
            explainer = shap.LinearExplainer(model, X_test_scaled[:100])
        else:
            explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X_test_scaled[:100])
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        all_importance[model_name] = dict(zip(feature_names, mean_abs_shap))
        loaded_models += 1

    if loaded_models < 2:
        logger.info(f"Only {loaded_models} model(s) found — skipping cross-model comparison plot")
        return

    # Build comparison DataFrame (top 10 features across models)
    top_features = set()
    for model_name, imp in all_importance.items():
        sorted_feats = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        for feat, _ in sorted_feats[:10]:
            top_features.add(feat)

    comparison_data = []
    for feat in top_features:
        row = {'Feature': feat}
        for model_name in all_importance:
            row[model_name] = all_importance[model_name].get(feat, 0)
        comparison_data.append(row)

    comp_df = pd.DataFrame(comparison_data)
    comp_df['mean'] = comp_df[[m for m in all_importance]].mean(axis=1)
    comp_df = comp_df.sort_values('mean', ascending=True).tail(10)

    # Plot
    plt.figure(figsize=(12, 8))
    y_pos = range(len(comp_df))
    bar_width = 0.8 / max(1, len(all_importance))

    for i, model_name in enumerate(all_importance):
        values = comp_df[model_name].values
        offset = (i - (len(all_importance) - 1) / 2) * bar_width
        plt.barh([y + offset for y in y_pos], values, height=bar_width,
                 label=model_name, alpha=0.8)

    plt.yticks(y_pos, comp_df['Feature'].values)
    plt.xlabel('Mean |SHAP Value|')
    plt.title('Cross-Model Feature Importance Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "shap_06_cross_model_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved cross-model comparison: {output_path}")
    plt.close()

    # Log comparison table
    print("\nCross-Model Feature Importance (Top 10):")
    print(comp_df.to_string(index=False))


if __name__ == "__main__":
    try:
        run_shap_analysis()
        run_cross_model_shap_comparison()
    except Exception as e:
        logger.error(f"SHAP analysis failed: {e}", exc_info=True)
        raise
