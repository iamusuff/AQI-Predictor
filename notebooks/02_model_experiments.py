"""
Pearls AQI Predictor — Model Experiments & Hyperparameter Tuning
Hyperparameter tuning, learning curves, residual analysis, and model comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

OUTPUT_DIR = "notebooks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("AQI PREDICTOR - MODEL EXPERIMENTS & HYPERPARAMETER TUNING")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & Prepare Data
# ─────────────────────────────────────────────────────────────────────────────

print("\n[1/6] Loading and preparing data...")
df = pd.read_csv('../data/features.csv', parse_dates=['timestamp'])
print(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")

exclude_cols = ['timestamp', 'aqi', 'dominentpol', 'weather_main', 'weather_description']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols].values
y = df['aqi'].values

# Temporal split
test_size = 0.15
val_size = 0.15
n_samples = len(X)
test_idx = int(n_samples * (1 - test_size))
val_idx = int(test_idx * (1 - val_size / (1 - test_size)))

X_train = X[:val_idx]
X_val = X[val_idx:test_idx]
X_test = X[test_idx:]
y_train = y[:val_idx]
y_val = y[val_idx:test_idx]
y_test = y[test_idx:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

print(f"✅ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Hyperparameter Tuning — Ridge
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2/6] Hyperparameter tuning — Ridge Regression...")

alpha_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
ridge_cv_scores = []

for alpha in alpha_values:
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train_s, y_train)
    y_val_pred = model.predict(X_val_s)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2 = r2_score(y_val, y_val_pred)
    ridge_cv_scores.append({'alpha': alpha, 'val_rmse': rmse, 'val_r2': r2})

ridge_cv_df = pd.DataFrame(ridge_cv_scores)
best_alpha = ridge_cv_df.loc[ridge_cv_df['val_rmse'].idxmin(), 'alpha']
print(f"✅ Best alpha: {best_alpha} (Val RMSE: {ridge_cv_df['val_rmse'].min():.4f})")

# Plot Ridge tuning
plt.figure()
plt.semilogx(ridge_cv_df['alpha'], ridge_cv_df['val_rmse'], marker='o', linewidth=2)
plt.axvline(best_alpha, color='red', linestyle='--', label=f'Best α={best_alpha}')
plt.xlabel('Alpha (regularization strength)')
plt.ylabel('Validation RMSE')
plt.title('Ridge Regression — Alpha Tuning')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'{OUTPUT_DIR}/experiment_01_ridge_tuning.png', dpi=150, bbox_inches='tight')
print("✅ Saved: experiment_01_ridge_tuning.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Hyperparameter Tuning — Random Forest
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3/6] Hyperparameter tuning — Random Forest...")

n_estimators_list = [10, 25, 50, 100, 200]
max_depth_list = [5, 10, 15, 20, 30]

rf_results = []
for n_est in n_estimators_list:
    for depth in max_depth_list:
        model = RandomForestRegressor(
            n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1
        )
        model.fit(X_train_s, y_train)
        y_val_pred = model.predict(X_val_s)
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        r2 = r2_score(y_val, y_val_pred)
        rf_results.append({'n_estimators': n_est, 'max_depth': depth, 'val_rmse': rmse, 'val_r2': r2})

rf_results_df = pd.DataFrame(rf_results)
best_rf = rf_results_df.loc[rf_results_df['val_rmse'].idxmin()]
print(f"✅ Best RF: n_estimators={int(best_rf['n_estimators'])}, max_depth={int(best_rf['max_depth'])}, Val RMSE={best_rf['val_rmse']:.4f}")

# Plot RF tuning heatmap
pivot = rf_results_df.pivot_table(values='val_rmse', index='max_depth', columns='n_estimators')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis_r')
plt.title('Random Forest — Validation RMSE by Hyperparameters')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/experiment_02_rf_tuning.png', dpi=150, bbox_inches='tight')
print("✅ Saved: experiment_02_rf_tuning.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Hyperparameter Tuning — XGBoost
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4/6] Hyperparameter tuning — XGBoost...")

learning_rates = [0.01, 0.05, 0.1, 0.2]
max_depths = [3, 6, 9]
n_estimators_xgb = [50, 100, 200]

xgb_results = []
for lr in learning_rates:
    for depth in max_depths:
        for n_est in n_estimators_xgb:
            model = xgb.XGBRegressor(
                n_estimators=n_est, max_depth=depth, learning_rate=lr,
                random_state=42, n_jobs=-1
            )
            model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
            y_val_pred = model.predict(X_val_s)
            rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            xgb_results.append({
                'learning_rate': lr, 'max_depth': depth,
                'n_estimators': n_est, 'val_rmse': rmse
            })

xgb_results_df = pd.DataFrame(xgb_results)
best_xgb = xgb_results_df.loc[xgb_results_df['val_rmse'].idxmin()]
print(f"✅ Best XGBoost: lr={best_xgb['learning_rate']}, depth={int(best_xgb['max_depth'])}, n_est={int(best_xgb['n_estimators'])}, Val RMSE={best_xgb['val_rmse']:.4f}")

# Plot XGBoost tuning (LR vs RMSE by depth)
plt.figure(figsize=(12, 5))
for depth in max_depths:
    subset = xgb_results_df[xgb_results_df['max_depth'] == depth].groupby('learning_rate')['val_rmse'].min().reset_index()
    plt.plot(subset['learning_rate'], subset['val_rmse'], marker='o', label=f'max_depth={depth}')
plt.xlabel('Learning Rate')
plt.ylabel('Validation RMSE')
plt.title('XGBoost — Validation RMSE by Learning Rate and Depth')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'{OUTPUT_DIR}/experiment_03_xgboost_tuning.png', dpi=150, bbox_inches='tight')
print("✅ Saved: experiment_03_xgboost_tuning.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Learning Curves
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5/6] Generating learning curves...")

# Train with varying training set sizes
train_sizes = np.linspace(0.1, 1.0, 10)
lc_ridge_rmse = []
lc_rf_rmse = []
lc_xgb_rmse = []

for frac in train_sizes:
    n_train = int(len(X_train_s) * frac)

    # Ridge
    m = Ridge(alpha=float(best_alpha), random_state=42)
    m.fit(X_train_s[:n_train], y_train[:n_train])
    p = m.predict(X_val_s)
    lc_ridge_rmse.append(np.sqrt(mean_squared_error(y_val, p)))

    # RF
    m = RandomForestRegressor(
        n_estimators=int(best_rf['n_estimators']),
        max_depth=int(best_rf['max_depth']),
        random_state=42, n_jobs=-1
    )
    m.fit(X_train_s[:n_train], y_train[:n_train])
    p = m.predict(X_val_s)
    lc_rf_rmse.append(np.sqrt(mean_squared_error(y_val, p)))

    # XGBoost
    m = xgb.XGBRegressor(
        n_estimators=int(best_xgb['n_estimators']),
        max_depth=int(best_xgb['max_depth']),
        learning_rate=best_xgb['learning_rate'],
        random_state=42, n_jobs=-1
    )
    m.fit(X_train_s[:n_train], y_train[:n_train])
    p = m.predict(X_val_s)
    lc_xgb_rmse.append(np.sqrt(mean_squared_error(y_val, p)))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes * 100, lc_ridge_rmse, marker='o', label='Ridge Regression', linewidth=2)
plt.plot(train_sizes * 100, lc_rf_rmse, marker='s', label='Random Forest', linewidth=2)
plt.plot(train_sizes * 100, lc_xgb_rmse, marker='^', label='XGBoost', linewidth=2)
plt.xlabel('Training Set Size (%)')
plt.ylabel('Validation RMSE')
plt.title('Learning Curves — Model Performance vs Training Data Size')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'{OUTPUT_DIR}/experiment_04_learning_curves.png', dpi=150, bbox_inches='tight')
print("✅ Saved: experiment_04_learning_curves.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 6. Residual Analysis
# ─────────────────────────────────────────────────────────────────────────────

print("\n[6/6] Performing residual analysis...")

# Train best models
ridge_best = Ridge(alpha=float(best_alpha), random_state=42)
ridge_best.fit(X_train_s, y_train)

rf_best = RandomForestRegressor(
    n_estimators=int(best_rf['n_estimators']),
    max_depth=int(best_rf['max_depth']),
    random_state=42, n_jobs=-1
)
rf_best.fit(X_train_s, y_train)

xgb_best = xgb.XGBRegressor(
    n_estimators=int(best_xgb['n_estimators']),
    max_depth=int(best_xgb['max_depth']),
    learning_rate=best_xgb['learning_rate'],
    random_state=42, n_jobs=-1
)
xgb_best.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)

# Test predictions
ridge_pred = ridge_best.predict(X_test_s)
rf_pred = rf_best.predict(X_test_s)
xgb_pred = xgb_best.predict(X_test_s)

# Residuals
ridge_resid = y_test - ridge_pred
rf_resid = y_test - rf_pred
xgb_resid = y_test - xgb_pred

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Residual distributions
models_resid = [
    ('Ridge Regression', ridge_resid, 'steelblue'),
    ('Random Forest', rf_resid, 'green'),
    ('XGBoost', xgb_resid, 'orange'),
]

for i, (name, resid, color) in enumerate(models_resid):
    axes[0, i].hist(resid, bins=30, edgecolor='black', alpha=0.7, color=color)
    axes[0, i].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[0, i].set_xlabel('Residual (True - Predicted)')
    axes[0, i].set_ylabel('Frequency')
    axes[0, i].set_title(f'{name}\nResidual Distribution')
    axes[0, i].grid(True, alpha=0.3)

# Actual vs Predicted scatter
for i, (name, pred, color) in enumerate([
    ('Ridge Regression', ridge_pred, 'steelblue'),
    ('Random Forest', rf_pred, 'green'),
    ('XGBoost', xgb_pred, 'orange'),
]):
    axes[1, i].scatter(y_test, pred, alpha=0.5, s=20, color=color)
    axes[1, i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                    'r--', linewidth=2, label='Perfect')
    axes[1, i].set_xlabel('True AQI')
    axes[1, i].set_ylabel('Predicted AQI')
    axes[1, i].set_title(f'{name}\nActual vs Predicted')
    axes[1, i].legend()
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/experiment_05_residual_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Saved: experiment_05_residual_analysis.png")
plt.close()

# Print summary metrics
models_summary = [
    ('Ridge Regression', ridge_pred),
    ('Random Forest', rf_pred),
    ('XGBoost', xgb_pred),
]

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON (Test Set)")
print("=" * 70)
print(f"{'Model':25s} {'RMSE':>8s} {'MAE':>8s} {'R²':>8s}")
print("-" * 55)
for name, pred in models_summary:
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"{name:25s} {rmse:8.3f} {mae:8.3f} {r2:8.4f}")

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING SUMMARY")
print("=" * 70)
print(f"Ridge Regression  — Best alpha: {best_alpha}")
print(f"Random Forest     — Best n_estimators: {int(best_rf['n_estimators'])}, max_depth: {int(best_rf['max_depth'])}")
print(f"XGBoost           — Best lr: {best_xgb['learning_rate']}, depth: {int(best_xgb['max_depth'])}, n_est: {int(best_xgb['n_estimators'])}")

print("\n" + "=" * 70)
print("GENERATED FILES")
print("=" * 70)
print("1. experiment_01_ridge_tuning.png       — Ridge alpha tuning curve")
print("2. experiment_02_rf_tuning.png          — RF hyperparameter heatmap")
print("3. experiment_03_xgboost_tuning.png     — XGBoost learning rate sweep")
print("4. experiment_04_learning_curves.png    — Learning curves for all models")
print("5. experiment_05_residual_analysis.png  — Residual analysis (3 models)")
print("=" * 70)
