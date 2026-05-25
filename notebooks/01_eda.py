"""
Pearls AQI Predictor — Exploratory Data Analysis (EDA)
Comprehensive analysis of AQI features and patterns.

Run this script to generate EDA visualizations and insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("=" * 70)
print("AQI PREDICTOR - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────────────────────────────────────

print("\n[1/10] Loading data...")
df = pd.read_csv('../data/features.csv', parse_dates=['timestamp'])
print(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")
print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"   Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Basic Statistics
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2/10] Computing basic statistics...")
print("\nDataset Info:")
print(f"  Total rows: {len(df)}")
print(f"  Total columns: {len(df.columns)}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nMissing Values:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing': missing[missing > 0],
    'Percentage': missing_pct[missing > 0]
})
if len(missing_df) > 0:
    print(missing_df)
else:
    print("  ✅ No missing values!")

print("\nAQI Statistics:")
print(df['aqi'].describe())


# ─────────────────────────────────────────────────────────────────────────────
# 3. AQI Distribution
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3/10] Analyzing AQI distribution...")

# AQI categories
aqi_categories = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy for Sensitive", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]

# Count by category
print("\nAQI Category Distribution:")
for min_val, max_val, label, color in aqi_categories:
    count = len(df[(df['aqi'] >= min_val) & (df['aqi'] <= max_val)])
    pct = (count / len(df)) * 100
    print(f"  {label:25s}: {count:4d} ({pct:5.1f}%)")

# Plot AQI distribution
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.hist(df['aqi'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('AQI')
plt.ylabel('Frequency')
plt.title('AQI Distribution')
plt.axvline(df['aqi'].mean(), color='red', linestyle='--', label=f'Mean: {df["aqi"].mean():.1f}')
plt.axvline(df['aqi'].median(), color='green', linestyle='--', label=f'Median: {df["aqi"].median():.1f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(df['aqi'], vert=True)
plt.ylabel('AQI')
plt.title('AQI Box Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_01_aqi_distribution.png', dpi=150, bbox_inches='tight')
print("✅ Saved: eda_01_aqi_distribution.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Time Series Analysis
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4/10] Analyzing time series patterns...")

# Plot AQI over time
plt.figure(figsize=(14, 6))
plt.plot(df['timestamp'], df['aqi'], linewidth=0.8, alpha=0.7)
plt.xlabel('Date')
plt.ylabel('AQI')
plt.title('AQI Time Series')
plt.grid(True, alpha=0.3)

# Add horizontal lines for AQI categories
for min_val, max_val, label, color in aqi_categories[:4]:  # First 4 categories
    plt.axhline(max_val, color=color, linestyle='--', alpha=0.3, linewidth=1)

plt.tight_layout()
plt.savefig('eda_02_aqi_timeseries.png', dpi=150, bbox_inches='tight')
print("✅ Saved: eda_02_aqi_timeseries.png")
plt.close()

# Rolling averages
plt.figure(figsize=(14, 6))
plt.plot(df['timestamp'], df['aqi'], label='Hourly AQI', alpha=0.3, linewidth=0.5)
plt.plot(df['timestamp'], df['aqi_rolling_24h'], label='24h Average', linewidth=1.5)
plt.plot(df['timestamp'], df['aqi_rolling_12h'], label='12h Average', linewidth=1.2, alpha=0.8)
plt.xlabel('Date')
plt.ylabel('AQI')
plt.title('AQI with Rolling Averages')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('eda_03_aqi_rolling.png', dpi=150, bbox_inches='tight')
print("✅ Saved: eda_03_aqi_rolling.png")
plt.close()

# Seasonal decomposition (formal STL)
print("\n[Seasonal Decomposition] Performing STL decomposition...")
decomposition = seasonal_decompose(
    df.set_index('timestamp')['aqi'].resample('D').mean().dropna(),
    model='additive',
    period=7
)

fig, axes = plt.subplots(4, 1, figsize=(14, 10))
decomposition.observed.plot(ax=axes[0], title='Observed (Daily Avg AQI)', linewidth=0.8)
decomposition.trend.plot(ax=axes[1], title='Trend', linewidth=0.8, color='orange')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal (7-day)', linewidth=0.8, color='green')
decomposition.resid.plot(ax=axes[3], title='Residual', linewidth=0.8, color='red', marker='o', markersize=2)
plt.tight_layout()
plt.savefig('eda_12_seasonal_decomposition.png', dpi=150, bbox_inches='tight')
print("✅ Saved: eda_12_seasonal_decomposition.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Seasonal and Daily Patterns
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5/10] Analyzing seasonal and daily patterns...")

# By hour
hourly_aqi = df.groupby('hour')['aqi'].agg(['mean', 'std', 'min', 'max'])
print("\nAQI by Hour of Day:")
print(hourly_aqi)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(hourly_aqi.index, hourly_aqi['mean'], marker='o', linewidth=2)
plt.fill_between(hourly_aqi.index, 
                 hourly_aqi['mean'] - hourly_aqi['std'],
                 hourly_aqi['mean'] + hourly_aqi['std'],
                 alpha=0.3)
plt.xlabel('Hour of Day')
plt.ylabel('AQI')
plt.title('Average AQI by Hour')
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 24, 2))

# By day of week
dow_aqi = df.groupby('day_of_week')['aqi'].agg(['mean', 'std'])
dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

plt.subplot(1, 2, 2)
plt.bar(range(7), dow_aqi['mean'], yerr=dow_aqi['std'], capsize=5, alpha=0.7)
plt.xlabel('Day of Week')
plt.ylabel('AQI')
plt.title('Average AQI by Day of Week')
plt.xticks(range(7), dow_labels)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('eda_04_temporal_patterns.png', dpi=150, bbox_inches='tight')
print("✅ Saved: eda_04_temporal_patterns.png")
plt.close()

# By season
season_aqi = df.groupby('season')['aqi'].agg(['mean', 'std', 'count'])
season_labels = ['Winter', 'Spring', 'Summer', 'Fall']
print("\nAQI by Season:")
for i, label in enumerate(season_labels):
    if i in season_aqi.index:
        print(f"  {label:10s}: {season_aqi.loc[i, 'mean']:6.1f} ± {season_aqi.loc[i, 'std']:5.1f} (n={season_aqi.loc[i, 'count']:.0f})")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Pollutant Analysis
# ─────────────────────────────────────────────────────────────────────────────

print("\n[6/10] Analyzing pollutants...")

pollutants = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']

print("\nPollutant Statistics:")
print(df[pollutants].describe())

# Pollutant distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, pollutant in enumerate(pollutants):
    axes[i].hist(df[pollutant].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[i].set_xlabel(pollutant.upper())
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{pollutant.upper()} Distribution')
    axes[i].axvline(df[pollutant].mean(), color='red', linestyle='--', 
                    label=f'Mean: {df[pollutant].mean():.1f}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_05_pollutant_distributions.png', dpi=150, bbox_inches='tight')
print("✅ Saved: eda_05_pollutant_distributions.png")
plt.close()

# Pollutants over time
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

for i, pollutant in enumerate(pollutants):
    axes[i].plot(df['timestamp'], df[pollutant], linewidth=0.5, alpha=0.7)
    axes[i].set_xlabel('Date')
    axes[i].set_ylabel(pollutant.upper())
    axes[i].set_title(f'{pollutant.upper()} Time Series')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_06_pollutant_timeseries.png', dpi=150, bbox_inches='tight')
print("✅ Saved: eda_06_pollutant_timeseries.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Weather Analysis
# ─────────────────────────────────────────────────────────────────────────────

print("\n[7/10] Analyzing weather features...")

weather_features = ['temperature', 'humidity', 'wind_speed', 'pressure']

print("\nWeather Statistics:")
print(df[weather_features].describe())

# Weather distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, feature in enumerate(weather_features):
    axes[i].hist(df[feature].dropna(), bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[i].set_xlabel(feature.replace('_', ' ').title())
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{feature.replace("_", " ").title()} Distribution')
    axes[i].axvline(df[feature].mean(), color='red', linestyle='--',
                    label=f'Mean: {df[feature].mean():.1f}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_07_weather_distributions.png', dpi=150, bbox_inches='tight')
print("✅ Saved: eda_07_weather_distributions.png")
plt.close()

# Temperature vs AQI
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.scatter(df['temperature'], df['aqi'], alpha=0.3, s=10)
plt.xlabel('Temperature (°C)')
plt.ylabel('AQI')
plt.title('Temperature vs AQI')
plt.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(df['temperature'].dropna(), df['aqi'][df['temperature'].notna()], 1)
p = np.poly1d(z)
plt.plot(df['temperature'].sort_values(), p(df['temperature'].sort_values()), 
         "r--", linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(df['humidity'], df['aqi'], alpha=0.3, s=10, color='green')
plt.xlabel('Humidity (%)')
plt.ylabel('AQI')
plt.title('Humidity vs AQI')
plt.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(df['humidity'].dropna(), df['aqi'][df['humidity'].notna()], 1)
p = np.poly1d(z)
plt.plot(df['humidity'].sort_values(), p(df['humidity'].sort_values()),
         "r--", linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
plt.legend()

plt.tight_layout()
plt.savefig('eda_08_weather_vs_aqi.png', dpi=150, bbox_inches='tight')
print("✅ Saved: eda_08_weather_vs_aqi.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Correlation Analysis
# ─────────────────────────────────────────────────────────────────────────────

print("\n[8/10] Computing correlations...")

# Select numeric features for correlation
numeric_features = ['aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co',
                   'temperature', 'humidity', 'wind_speed', 'pressure',
                   'hour', 'day_of_week', 'month', 'is_weekend', 'season']

correlation_matrix = df[numeric_features].corr()

# Top correlations with AQI
aqi_corr = correlation_matrix['aqi'].sort_values(ascending=False)
print("\nTop 10 Features Correlated with AQI:")
print(aqi_corr.head(10))

# Correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('eda_09_correlation_heatmap.png', dpi=150, bbox_inches='tight')
print("✅ Saved: eda_09_correlation_heatmap.png")
plt.close()

# Pollutants vs AQI correlation
pollutant_corr = correlation_matrix.loc[pollutants, 'aqi'].sort_values(ascending=False)
print("\nPollutant Correlations with AQI:")
print(pollutant_corr)

plt.figure(figsize=(10, 6))
plt.barh(range(len(pollutant_corr)), pollutant_corr.values)
plt.yticks(range(len(pollutant_corr)), [p.upper() for p in pollutant_corr.index])
plt.xlabel('Correlation with AQI')
plt.title('Pollutant Correlations with AQI')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('eda_10_pollutant_correlations.png', dpi=150, bbox_inches='tight')
print("✅ Saved: eda_10_pollutant_correlations.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Feature Importance (Preliminary)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[9/10] Analyzing feature importance...")

# Scatter plots: Top correlated features vs AQI
top_features = aqi_corr.head(6).index[1:]  # Exclude AQI itself

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, feature in enumerate(top_features):
    if i < len(axes):
        axes[i].scatter(df[feature], df['aqi'], alpha=0.3, s=10)
        axes[i].set_xlabel(feature.replace('_', ' ').title())
        axes[i].set_ylabel('AQI')
        axes[i].set_title(f'{feature.replace("_", " ").title()} vs AQI\n(r={aqi_corr[feature]:.3f})')
        axes[i].grid(True, alpha=0.3)
        
        # Add trend line
        if df[feature].notna().sum() > 0:
            z = np.polyfit(df[feature].dropna(), df['aqi'][df[feature].notna()], 1)
            p = np.poly1d(z)
            x_sorted = df[feature].sort_values()
            axes[i].plot(x_sorted, p(x_sorted), "r--", linewidth=2, alpha=0.7)

plt.tight_layout()
plt.savefig('eda_11_top_features_vs_aqi.png', dpi=150, bbox_inches='tight')
print("✅ Saved: eda_11_top_features_vs_aqi.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 10. Summary Report
# ─────────────────────────────────────────────────────────────────────────────

print("\n[10/10] Generating summary report...")

summary_report = f"""
{'=' * 70}
AQI PREDICTOR - EDA SUMMARY REPORT
{'=' * 70}

DATASET OVERVIEW
{'─' * 70}
Total Rows:              {len(df):,}
Total Columns:           {len(df.columns)}
Date Range:              {df['timestamp'].min()} to {df['timestamp'].max()}
Duration:                {(df['timestamp'].max() - df['timestamp'].min()).days} days
Memory Usage:            {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

AQI STATISTICS
{'─' * 70}
Mean AQI:                {df['aqi'].mean():.2f}
Median AQI:              {df['aqi'].median():.2f}
Std Dev:                 {df['aqi'].std():.2f}
Min AQI:                 {df['aqi'].min():.2f}
Max AQI:                 {df['aqi'].max():.2f}

AQI CATEGORY DISTRIBUTION
{'─' * 70}
"""

for min_val, max_val, label, color in aqi_categories:
    count = len(df[(df['aqi'] >= min_val) & (df['aqi'] <= max_val)])
    pct = (count / len(df)) * 100
    summary_report += f"{label:25s}: {count:5d} ({pct:5.1f}%)\n"

summary_report += f"""
TOP 5 FEATURES CORRELATED WITH AQI
{'─' * 70}
"""

for feature, corr in aqi_corr.head(6).items():
    if feature != 'aqi':
        summary_report += f"{feature:25s}: {corr:6.3f}\n"

summary_report += f"""
POLLUTANT STATISTICS
{'─' * 70}
"""

for pollutant in pollutants:
    summary_report += f"{pollutant.upper():10s}: Mean={df[pollutant].mean():6.2f}, Std={df[pollutant].std():6.2f}, Max={df[pollutant].max():6.2f}\n"

summary_report += f"""
WEATHER STATISTICS
{'─' * 70}
Temperature:             {df['temperature'].mean():.2f}°C (±{df['temperature'].std():.2f})
Humidity:                {df['humidity'].mean():.2f}% (±{df['humidity'].std():.2f})
Wind Speed:              {df['wind_speed'].mean():.2f} m/s (±{df['wind_speed'].std():.2f})
Pressure:                {df['pressure'].mean():.2f} hPa (±{df['pressure'].std():.2f})

TEMPORAL PATTERNS
{'─' * 70}
Peak AQI Hour:           {hourly_aqi['mean'].idxmax()}:00 (AQI={hourly_aqi['mean'].max():.1f})
Lowest AQI Hour:         {hourly_aqi['mean'].idxmin()}:00 (AQI={hourly_aqi['mean'].min():.1f})

GENERATED VISUALIZATIONS
{'─' * 70}
1.  eda_01_aqi_distribution.png
2.  eda_02_aqi_timeseries.png
3.  eda_03_aqi_rolling.png
4.  eda_04_temporal_patterns.png
5.  eda_05_pollutant_distributions.png
6.  eda_06_pollutant_timeseries.png
7.  eda_07_weather_distributions.png
8.  eda_08_weather_vs_aqi.png
9.  eda_09_correlation_heatmap.png
10. eda_10_pollutant_correlations.png
11. eda_11_top_features_vs_aqi.png
12. eda_12_seasonal_decomposition.png

{'=' * 70}
EDA COMPLETED SUCCESSFULLY
{'=' * 70}
"""

print(summary_report)

# Save report
with open('eda_summary_report.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)
print("\n✅ Saved: eda_summary_report.txt")

print("\n" + "=" * 70)
print("All EDA visualizations and reports have been generated!")
print("Check the 'notebooks/' directory for output files.")
print("=" * 70)
