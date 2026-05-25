# 🌍 Pearls AQI Predictor

> Predict the Air Quality Index (AQI) for **Karachi** (and other cities) for the next 3 days using a fully serverless ML pipeline.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Features:**
- ⏰ Automated hourly data collection from AQICN & OpenWeather APIs
- 🔄 Daily model retraining with multiple ML algorithms (Ridge, Random Forest, XGBoost, LSTM, GRU)
- 📊 Interactive Streamlit dashboard with 3-day forecasts
- 🎯 SHAP-based explainability for predictions
- 🚨 Real-time health alerts for hazardous AQI levels

## Architecture

```
┌─────────────────┐
│  AQICN API      │  Pollutants: PM2.5, PM10, O3, NO2, SO2, CO
│  OpenWeather API│  Weather: temp, humidity, wind, pressure
└────────┬────────┘
         │ raw data (hourly via GitHub Actions)
         ▼
┌─────────────────┐      ┌──────────────────┐
│     Feature     │─────▶│  Hopsworks       │
│   Generation    │      │  Feature Store   │
│  (time, derived)│      │  (versioned)     │
└─────────────────┘      └────────┬─────────┘
                                  │ features + targets
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
                                  │ model + features
                                  ▼
                         ┌──────────────────┐
                         │   Streamlit      │
                         │   Dashboard      │
                         │   + Flask API    │
                         └──────────────────┘
```

## Project Structure

```
AQI_Predictor/
├── src/
│   ├── config.py              # Configuration & constants
│   ├── utils.py               # API helpers & feature engineering
│   ├── feature_pipeline.py    # Fetch → compute → store features
│   ├── backfill.py            # Historical data backfill
│   ├── training_pipeline.py   # Train → evaluate → register model
│   └── inference.py           # Load model → predict next 3 days
├── app/
│   ├── streamlit_app.py       # Interactive dashboard
│   └── flask_api.py           # REST API
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_experiments.ipynb
│   └── 03_shap_analysis.ipynb
├── .github/workflows/
│   ├── feature_pipeline.yml   # Runs every hour
│   └── training_pipeline.yml  # Runs every day
├── tests/
├── requirements.txt
├── .env.example
└── main.py                    # CLI entry point
```

## Quick Start

### 1. Clone & create virtual environment
```bash
git clone <your-repo-url>
cd AQI_Predictor
python -m venv venv
venv\Scripts\activate      # Windows
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
Sign up for:
- **AQICN token** → https://aqicn.org/data-platform/token/
- **OpenWeatherMap key** → https://openweathermap.org/api
- **Hopsworks account** → https://app.hopsworks.ai/

### 4. Validate configuration
```bash
python main.py --check-config
```

### 5. Run pipelines
```bash
python main.py --pipeline feature     # Fetch & store latest features
python main.py --pipeline backfill    # Backfill historical data
python main.py --pipeline train       # Train & register model
python main.py --pipeline predict     # Predict next 3 days
```

### 6. Launch dashboard
```bash
streamlit run app/streamlit_app.py
```

## Technology Stack

| Layer | Tool |
|-------|------|
| Data APIs | AQICN, OpenWeatherMap |
| Feature Store | Hopsworks |
| ML Models | Scikit-learn, XGBoost, TensorFlow |
| Explainability | SHAP |
| CI/CD | GitHub Actions |
| Dashboard | Streamlit |
| API | Flask |

## AQI Scale

| AQI Range | Category | Color | Health Impact |
|-----------|----------|-------|---------------|
| 0 – 50 | Good | 🟢 Green | Air quality is satisfactory |
| 51 – 100 | Moderate | 🟡 Yellow | Acceptable; some pollutants may affect sensitive people |
| 101 – 150 | Unhealthy for Sensitive Groups | 🟠 Orange | Sensitive individuals should limit outdoor activity |
| 151 – 200 | Unhealthy | 🔴 Red | Everyone may experience health effects |
| 201 – 300 | Very Unhealthy | 🟣 Purple | Health alert — everyone should avoid outdoor activity |
| 301 – 500 | Hazardous | 🟤 Maroon | Health emergency |

---

## 📝 Development Status

- [x] **Phase 1**: Project setup & configuration ✅
- [x] **Phase 2**: Feature pipeline development ✅
- [x] **Phase 3**: Historical data backfill ✅
- [x] **Phase 4**: Exploratory Data Analysis ✅
- [x] **Phase 5**: Training pipeline (multiple models) ✅
- [ ] **Phase 6**: Advanced analytics & explainability (SHAP)
- [ ] **Phase 7**: Web application dashboard
- [ ] **Phase 8**: CI/CD automation (GitHub Actions)

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_feature_pipeline.py -v
```

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is part of the Pearls AQI Predictor initiative.

---

## 🙏 Acknowledgments

- [AQICN](https://aqicn.org/) for real-time air quality data
- [OpenWeatherMap](https://openweathermap.org/) for weather data
- [Hopsworks](https://www.hopsworks.ai/) for feature store & model registry
- US EPA for AQI calculation standards
