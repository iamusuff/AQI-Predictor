# 🌍 Hawa Alert: AQI Predictor

> Predict the Air Quality Index (AQI) for **Karachi** (and other cities) for the next 3 days using a fully serverless ML pipeline — built entirely on free-tier cloud services.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hawa-alert-aqi-predictor.streamlit.app/)

**Live App:** [https://hawa-alert-aqi-predictor.streamlit.app/](https://hawa-alert-aqi-predictor.streamlit.app/)  
**Source Code:** [https://github.com/iamusuff/AQI-Predictor](https://github.com/iamusuff/AQI-Predictor)

---

## Overview

Air pollution is a critical public health challenge in Karachi — one of the world's most polluted cities. Hawa Alert provides citizens, schools, and administrators with accurate 3-day AQI forecasts so they can make informed decisions about outdoor activity and health precautions.

The system is 100% serverless and operates entirely on free-tier cloud services, making it sustainable for long-term public deployment.

**Features:**
- ⏰ Automated hourly data collection from AQICN & OpenMeteo APIs (free!)
- 🔄 Daily model retraining with multiple ML algorithms (LightGBM, CatBoost, XGBoost, Random Forest)
- 📊 Interactive Streamlit dashboard with 3-day AQI forecasts (24h, 48h, 72h)
- 🎯 SHAP-based explainability for predictions
- 🚨 Real-time health alerts and recommendations by AQI category
- 🔁 Full CI/CD automation via GitHub Actions — zero manual intervention after deployment

---

## Model Performance

The production model (XGBoost) was trained on 90 days of hourly data (2,160 records) with a temporal 70%/15%/15% train/validation/test split.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Test R² | 0.8908 | 89.08% of AQI variance explained |
| Test RMSE | 15.41 AQI pts | Typical error within one health category |
| Test MAE | 10.23 AQI pts | Most predictions within ±10 AQI points |
| Training Time | 4.2s | Fast enough for daily automated retraining |
| Inference Latency | ~800ms | Suitable for real-time forecasting |
| Cross-Val RMSE | 15.8 ± 0.4 | Stable across time folds |

**Comparison to baseline:** 56% RMSE improvement over a naive persistence model (yesterday's AQI).

### Forecast Accuracy by Horizon

| Horizon | RMSE | MAE | R² | Within ±1 Category |
|---------|------|-----|----|---------------------|
| Current (t+0) | 15.41 | 10.23 | 0.8908 | 94.3% |
| 24 hours | 18.67 | 13.45 | 0.8512 | 89.7% |
| 48 hours | 21.34 | 15.89 | 0.8134 | 84.2% |
| 72 hours | 24.12 | 18.21 | 0.7756 | 78.9% |

### Model Comparison

| Model | Val RMSE | Val R² | Test RMSE | Test R² | Training Time |
|-------|----------|--------|-----------|---------|---------------|
| **XGBoost** ✅ | **15.98** | **0.8854** | **15.41** | **0.8908** | 4.2s |
| LightGBM | 16.12 | 0.8831 | 15.63 | 0.8895 | 2.8s |
| CatBoost | 16.34 | 0.8797 | 15.89 | 0.8869 | 12.1s |
| Random Forest | 18.56 | 0.8421 | 18.03 | 0.8456 | 8.9s |

XGBoost was selected for production based on best validation RMSE.

---

## Architecture

```
┌─────────────────┐
│  AQICN API      │  Pollutants: PM2.5, PM10, O3, NO2, SO2, CO (real-time)
│  OpenMeteo API  │  Weather: temp, humidity, wind, pressure (current + 7-day forecast)
└────────┬────────┘
         │ raw data (hourly via GitHub Actions)
         ▼
┌─────────────────┐      ┌──────────────────┐
│     Feature     │─────▶│  Hopsworks       │
│   Engineering   │      │  Feature Store   │
│  (21 features)  │      │  (versioned)     │
└─────────────────┘      └────────┬─────────┘
                                  │ 90-day rolling window
                                  ▼
                         ┌──────────────────┐
                         │  Training        │
                         │  Pipeline        │
                         │  (daily)         │
                         └────────┬─────────┘
                                  │ best model
                                  ▼
                         ┌──────────────────┐
                         │  Hopsworks       │
                         │  Model Registry  │
                         └────────┬─────────┘
                                  │ model + scaler + feature names
                                  ▼
                         ┌──────────────────┐
                         │   Streamlit      │
                         │   Dashboard      │
                         └──────────────────┘
```

---

## Data Sources

### AQICN API (Pollutants)
- **Station:** Karachi, Pakistan — Station ID `@11348`
- **Data:** AQI, PM2.5, PM10, O3, NO2, SO2, CO, dominant pollutant
- **Update frequency:** Hourly
- **Cost:** Free (1,000 requests/day)

### OpenMeteo API (Weather + Forecasts)
- **Location:** Karachi (24.8607°N, 67.0011°E)
- **Data:** Temperature, humidity, wind speed, pressure, visibility, cloud cover
- **Forecast horizon:** 7 days (168 hours)
- **Cost:** Completely free — no API key required (10,000 requests/day)

> OpenMeteo replaced the originally planned OpenWeather API because it provides enterprise-grade weather data and historical pollutant records (PM2.5, PM10, O3, NO2, SO2, CO for the past 5 years) at zero cost.

---

## Feature Engineering

21 features are computed per hourly record across four categories:

### Raw Pollutant Features (6)
`pm25`, `pm10`, `o3`, `no2`, `so2`, `co`

### Weather Features (6)
`temperature`, `humidity`, `wind_speed`, `pressure`, `visibility`, `cloud_cover`

### Temporal Features (6)
`hour`, `day_of_week`, `day_of_month`, `month`, `season`, `is_weekend`

Temporal features capture strong diurnal patterns (rush-hour AQI spikes at 8–10 AM and 6–8 PM) and seasonal variation (winter smog episodes).

### Derived Features (3)
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `pm25_pm10_ratio` | PM2.5 / PM10 | Distinguishes combustion (vehicles) vs. dust sources |
| `temp_humidity` | Temp × Humidity | Captures heat-index effect on pollutant accumulation |
| `wind_pm25` | Wind Speed × PM2.5 | Models wind dispersion of particulate matter |

Derived features rank in the top 10 by SHAP importance, confirming the value of domain-informed feature engineering.

### Top Features by SHAP Importance

| Rank | Feature | SHAP Value | Insight |
|------|---------|-----------|---------|
| 1 | PM2.5 | 0.3421 | Dominant AQI driver (60% of high-AQI days) |
| 2 | PM10 | 0.2134 | Significant during dust storms |
| 3 | Humidity | 0.1876 | High humidity traps pollutants near ground level |
| 4 | Hour | 0.1543 | Rush-hour AQI spikes clearly captured |
| 5 | PM2.5/PM10 Ratio | 0.1298 | Separates combustion vs. dust events |
| 6 | Wind Speed | 0.1187 | Negative correlation — disperses pollutants |
| 7 | Temperature | 0.0987 | Warmer temps increase photochemical O3 |
| 8 | O3 | 0.0854 | Contributes to AQI in summer months |
| 9 | Wind × PM2.5 | 0.0743 | Interaction captures dispersion dynamics |
| 10 | Month | 0.0621 | Winter smog > summer |

---

## Inference & Multi-Horizon Forecasting

For future horizons (t+24h, t+48h, t+72h), future pollutant concentrations are unknown since sensors cannot predict future emissions. The system applies a **meteorological decay model** based on atmospheric physics:

- **PM2.5 / PM10:** Base decay 2%/day + wind dispersion (up to 15% reduction above 10 m/s) + humidity trapping (up to 5% increase above 70% humidity)
- **O3:** Base decay 2%/day + temperature amplification (+1% per 10°C above 25°C, for photochemical reactions)
- **NO2, SO2, CO:** Slow decay of 5%/day (reduced nighttime/off-peak emissions assumed)

This approach improved 24h-ahead RMSE from 22.3 (simple persistence) to 18.7 — a 16% improvement.

**Confidence intervals** are computed as ±1.96 × RMSE, widening with horizon:

| Horizon | 95% CI Width |
|---------|-------------|
| Current | ±30.2 |
| 24h | ±34.7 |
| 48h | ±37.7 |
| 72h | ±42.1 |

---

## Project Structure

```
AQI-Predictor/
├── src/
│   ├── config.py              # Configuration & constants
│   ├── utils.py               # API helpers, feature engineering, shared utilities
│   ├── feature_pipeline.py    # Fetch → compute features → store in Hopsworks
│   ├── backfill.py            # One-time historical data backfill (90 days)
│   ├── training_pipeline.py   # Train → evaluate → register best model
│   └── inference.py           # Load model → apply decay model → predict next 3 days
├── app/
│   └── streamlit_app.py       # Interactive 4-page dashboard
├── notebooks/
│   ├── 01_eda.py
│   └── 02_model_experiments.ipynb
├── .github/workflows/
│   ├── feature_pipeline.yml   # Runs every hour (cron: 0 * * * *)
│   └── training_pipeline.yml  # Runs every day at midnight UTC
├── requirements.txt
├── .env.example
└── main.py                    # CLI entry point
```

---

## Quick Start

### 1. Clone & create virtual environment
```bash
git clone https://github.com/iamusuff/AQI-Predictor.git
cd AQI-Predictor
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API keys
```bash
copy .env.example .env
# Edit .env and fill in your API keys
```

You'll need to sign up for:
- **AQICN token** → [https://aqicn.org/data-platform/token/](https://aqicn.org/data-platform/token/)
- **OpenMeteo** → FREE, no API key required — [https://open-meteo.com/](https://open-meteo.com/)
- **Hopsworks account** → [https://app.hopsworks.ai/](https://app.hopsworks.ai/) (free tier)

### 4. Validate configuration
```bash
python main.py --check-config
```

### 5. Run pipelines

Run these in order on first setup:

```bash
python main.py --pipeline backfill    # One-time: backfill 90 days of historical data (~45 min)
python main.py --pipeline feature     # Fetch & store latest features
python main.py --pipeline train       # Train all 4 models & register best to Hopsworks
python main.py --pipeline predict     # Predict AQI for next 3 days
```

### 6. Launch dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## Dashboard

The Streamlit dashboard has four pages:

**Dashboard (Main)**
- 4 forecast cards: Current, +24h, +48h, +72h — each with AQI value, health category badge, 95% confidence interval, and personalized health advice
- Current conditions panel: temperature, humidity, wind speed, pressure, visibility
- Live pollutant readings: PM2.5, PM10, O3, NO2, SO2, CO
- Model performance metrics (R², RMSE, MAE)
- 5-minute cached refresh

**Historical Trends**
- Interactive time-series chart for 7, 30, or 90-day windows
- 24-hour rolling average overlay
- EPA health category background shading
- Summary statistics (min, max, mean, median)

**Feature Importance**
- SHAP summary bar chart (top 15 features)
- Radar chart (top 8 features)
- Cumulative contribution chart (top 10 features)
- Feature descriptions with hover tooltips

**About**
- Project overview, data source links, model comparison table, EPA AQI guide

---

## CI/CD & MLOps

### GitHub Actions Pipelines

| Pipeline | Trigger | Runtime | Monthly Runs |
|----------|---------|---------|--------------|
| Feature Pipeline | Every hour (`0 * * * *`) | ~2–3 min | 720 |
| Training Pipeline | Daily at midnight UTC | ~8–12 min | 30 |

Combined monthly compute: ~1,080 minutes — within GitHub Actions free tier (2,000 min/month).

### Feature Pipeline (Hourly)
1. Fetch pollutant data from AQICN API
2. Fetch weather data from OpenMeteo API
3. Compute 21 features + validate ranges
4. Insert into Hopsworks Feature Store
5. Upload `features.csv` as artifact (7-day retention)

### Training Pipeline (Daily)
1. Load 90-day rolling window from Hopsworks
2. Train LightGBM, CatBoost, XGBoost, Random Forest with temporal CV (3 folds)
3. Select best model by validation RMSE
4. Compute SHAP importance
5. Register model to Hopsworks Model Registry with full metadata (R², RMSE, MAE, hyperparameters, feature names)
6. Upload `shap_importance.csv` to GitHub Releases

### Model Versioning & Rollback

Every daily run creates a versioned model in Hopsworks Model Registry. The dashboard always loads the best model by `test_r2`. To roll back to a previous version, update the version number in `inference.py` and redeploy.

---

## Technology Stack

| Layer | Tool | Notes |
|-------|------|-------|
| Data APIs | AQICN + OpenMeteo | OpenMeteo is free with no API key; provides historical data too |
| Feature Store | Hopsworks (Free Tier) | Versioning, time-travel queries, model registry |
| ML Models | LightGBM, CatBoost, XGBoost, Random Forest | All gradient boosting; XGBoost selected for production |
| Explainability | SHAP 0.49.1 | TreeExplainer for fast, accurate feature importance |
| CI/CD | GitHub Actions | Cron-scheduled hourly + daily pipelines |
| Dashboard | Streamlit 1.28+ | `@st.cache_resource` for Hopsworks connection, `st.session_state` for inference results |
| Visualization | Plotly 5.15.0 | Interactive charts with dark theme + health-category color scales |
| Deployment | Streamlit Cloud (Free Tier) | Auto-deploys from GitHub main, HTTPS, auto-scaling |

---

## Cost Analysis

| Service | Tier | Cost |
|---------|------|------|
| AQICN API | Free (1,000 req/day) | $0 |
| OpenMeteo API | Free (10,000 req/day) | $0 |
| Hopsworks | Serverless Free Tier | $0 |
| GitHub Actions | Free (2,000 min/month) | $0 |
| Streamlit Cloud | Free Tier | $0 |
| **Total** | | **$0/month** |

Projected cost at 10 cities: still $0/month (within free tier limits).

---

## AQI Scale

| AQI Range | Category | Color | Health Guidance |
|-----------|----------|-------|----------------|
| 0–50 | Good | 🟢 Green | Air quality is satisfactory. Enjoy outdoor activities. |
| 51–100 | Moderate | 🟡 Yellow | Acceptable; unusually sensitive people should limit prolonged outdoor exertion. |
| 101–150 | Unhealthy for Sensitive Groups | 🟠 Orange | Children, elderly, asthmatics should reduce prolonged outdoor exertion. |
| 151–200 | Unhealthy | 🔴 Red | Everyone may experience health effects. Limit outdoor exertion. |
| 201–300 | Very Unhealthy | 🟣 Purple | Health alert — everyone may experience serious effects. Avoid outdoor activities. |
| 301–500 | Hazardous | 🟤 Maroon | Health emergency. Everyone should avoid all outdoor exertion. |

Categories follow US EPA AQI standards (EPA-454/B-24-002).

---

## Development Status

- [x] **Phase 1**: Project setup & configuration ✅
- [x] **Phase 2**: Feature pipeline development ✅
- [x] **Phase 3**: Historical data backfill (90 days, 2,160 records) ✅
- [x] **Phase 4**: Exploratory Data Analysis ✅
- [x] **Phase 5**: Training pipeline (4 models, temporal CV, recency weighting) ✅
- [x] **Phase 6**: Advanced analytics & explainability (SHAP) ✅
- [x] **Phase 7**: Web application dashboard (4 pages) ✅
- [x] **Phase 8**: CI/CD automation (GitHub Actions — hourly + daily) ✅

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Acknowledgments

- [AQICN](https://aqicn.org/) for real-time air quality data
- [OpenMeteo](https://open-meteo.com/) for free weather data, forecasts, and historical pollutant records
- [Hopsworks](https://www.hopsworks.ai/) for feature store & model registry
- US EPA for AQI calculation standards and health category definitions