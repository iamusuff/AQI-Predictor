# 🚀 Quick Start Guide - AQI Predictor

## Current Status: Phase 2 Complete ✅

You now have a **working feature pipeline** that can fetch real-time air quality and weather data!

---

## ⚡ Quick Test (5 minutes)

### Step 1: Get API Keys (Free)

1. **AQICN API Key** (Required)
   - Visit: https://aqicn.org/data-platform/token/
   - Sign up with email
   - Copy your token

2. **OpenWeather API Key** (Required)
   - Visit: https://openweathermap.org/api
   - Sign up for free account
   - Go to API Keys section
   - Copy your key

3. **Hopsworks** (Optional - skip for now)
   - Can use local CSV storage instead

### Step 2: Configure Environment

```bash
# Copy template
copy .env.example .env

# Edit .env file and add your keys:
AQICN_API_KEY=your_aqicn_token_here
OPENWEATHER_API_KEY=your_openweather_key_here
CITY=karachi
```

### Step 3: Test the Pipeline

```bash
# Test API connections
python src/utils.py

# Run feature pipeline (saves to CSV)
python src/feature_pipeline.py --no-hopsworks

# Check the output
type data\features.csv
```

### Expected Output

```
============================================================
FEATURE PIPELINE STARTED
City: Karachi, Pakistan
============================================================

[1/5] Fetching air quality data from AQICN...
✅ AQI: 156, PM2.5: 65.5, Dominant: pm25

[2/5] Fetching weather data from OpenWeather...
✅ Temp: 28.5°C, Humidity: 65%, Wind: 3.5 m/s

[3/5] Computing features...
✅ Generated 31 features

[5/5] Saving features locally (backup)...
✅ Saved features to data/features.csv

============================================================
FEATURE PIPELINE COMPLETED
  Local Storage: ✅ Success
============================================================
```

---

## 📂 What You Have Now

### Working Components ✅

1. **API Integration**
   - ✅ Fetch real-time AQI data (PM2.5, PM10, O3, NO2, SO2, CO)
   - ✅ Fetch weather data (temperature, humidity, wind, pressure)
   - ✅ Error handling and logging

2. **Feature Engineering**
   - ✅ 30+ features per timestamp
   - ✅ Time-based features (hour, day, month, season)
   - ✅ Rolling averages (3h, 6h, 12h, 24h)
   - ✅ Pollutant ratios and interactions

3. **Data Storage**
   - ✅ Local CSV storage (`data/features.csv`)
   - ✅ Hopsworks integration (optional)
   - ✅ Duplicate handling

4. **CLI Interface**
   - ✅ `python main.py --pipeline feature`
   - ✅ `python main.py --check-config`

### What's Next ⏳

1. **Phase 3**: Historical data backfill (90-180 days)
2. **Phase 4**: Exploratory data analysis
3. **Phase 5**: Train ML models
4. **Phase 6**: SHAP explainability
5. **Phase 7**: Streamlit dashboard
6. **Phase 8**: GitHub Actions automation

---

## 🎯 Common Commands

### Configuration
```bash
# Validate configuration
python main.py --check-config

# Test API connections
python src/utils.py
```

### Feature Pipeline
```bash
# Run once (local storage only)
python src/feature_pipeline.py --no-hopsworks

# Run with Hopsworks (if configured)
python src/feature_pipeline.py

# Run via main CLI
python main.py --pipeline feature
```

### Data Inspection
```bash
# View CSV data
type data\features.csv

# Count rows
python -c "import pandas as pd; df = pd.read_csv('data/features.csv'); print(f'Rows: {len(df)}')"

# View columns
python -c "import pandas as pd; df = pd.read_csv('data/features.csv'); print(df.columns.tolist())"
```

---

## 🏙️ City Configuration

Currently configured for **Karachi**. To switch cities:

### Option 1: Environment Variable
```bash
# Edit .env file
CITY=lahore    # or islamabad
```

### Option 2: Command Line
```bash
python main.py --city lahore --pipeline feature
```

### Supported Cities
- `karachi` - Karachi, Pakistan (default)
- `lahore` - Lahore, Pakistan
- `islamabad` - Islamabad, Pakistan

---

## 📊 Generated Features

### Raw Features (13)
- `timestamp` - Event timestamp
- `aqi` - Overall Air Quality Index
- `pm25`, `pm10`, `o3`, `no2`, `so2`, `co` - Pollutants
- `dominentpol` - Dominant pollutant
- `temperature`, `humidity`, `wind_speed`, `pressure`, `visibility`, `clouds`
- `weather_main` - Weather condition

### Time Features (6)
- `hour` (0-23)
- `day_of_week` (0=Monday, 6=Sunday)
- `day_of_month` (1-31)
- `month` (1-12)
- `is_weekend` (0/1)
- `season` (0=Winter, 1=Spring, 2=Summer, 3=Fall)

### Derived Features (12+)
- Rolling averages: `aqi_rolling_3h`, `aqi_rolling_6h`, `aqi_rolling_12h`, `aqi_rolling_24h`
- Rolling PM2.5: `pm25_rolling_3h`, `pm25_rolling_6h`, `pm25_rolling_12h`, `pm25_rolling_24h`
- Rolling PM10: `pm10_rolling_3h`, `pm10_rolling_6h`, `pm10_rolling_12h`, `pm10_rolling_24h`
- Change rates: `aqi_change_1h`, `aqi_change_3h`
- Ratios: `pm25_pm10_ratio`, `no2_o3_ratio`
- Interactions: `temp_humidity_interaction`, `wind_pm25_interaction`

---

## 🔧 Troubleshooting

### Issue: "Missing API key"
```bash
# Solution: Check .env file exists and has keys
python main.py --check-config
```

### Issue: "Failed to fetch AQICN data"
```bash
# Solution: Verify API key is correct
# Test manually: https://api.waqi.info/feed/@11348/?token=YOUR_TOKEN
```

### Issue: "Failed to fetch OpenWeather data"
```bash
# Solution: Verify API key is correct
# Test manually: https://api.openweathermap.org/data/2.5/weather?lat=24.8607&lon=67.0011&appid=YOUR_KEY
```

### Issue: "Hopsworks import error"
```bash
# Solution: Use local storage instead
python src/feature_pipeline.py --no-hopsworks
```

### Issue: "No data in CSV"
```bash
# Solution: Check if data directory exists
mkdir data
python src/feature_pipeline.py --no-hopsworks
```

---

## 📚 Documentation

- **[README.md](README.md)** - Project overview
- **[implementation_plan.md](implementation_plan.md)** - Full 8-phase plan
- **[PHASE1_COMPLETE.md](PHASE1_COMPLETE.md)** - Phase 1 summary
- **[PHASE2_COMPLETE.md](PHASE2_COMPLETE.md)** - Phase 2 summary (current)
- **[PROGRESS.md](PROGRESS.md)** - Overall progress tracker

---

## 🎓 Learning Resources

### AQI Standards
- US EPA AQI Guide: https://www.airnow.gov/aqi/aqi-basics/
- AQICN Documentation: https://aqicn.org/api/

### APIs
- AQICN API Docs: https://aqicn.org/json-api/doc/
- OpenWeather API Docs: https://openweathermap.org/api

### Hopsworks
- Hopsworks Docs: https://docs.hopsworks.ai/
- Feature Store Guide: https://docs.hopsworks.ai/feature-store-api/latest/

---

## ❓ FAQ

**Q: Do I need Hopsworks to test the pipeline?**  
A: No! You can use local CSV storage with `--no-hopsworks` flag.

**Q: How often should I run the feature pipeline?**  
A: For production, hourly (via GitHub Actions in Phase 8). For testing, run manually.

**Q: Can I use a different city?**  
A: Yes! Edit `CITY` in `.env` or use `--city` flag. Supported: karachi, lahore, islamabad.

**Q: What if I don't have API keys yet?**  
A: You can't test the pipeline without AQICN and OpenWeather keys (both free).

**Q: How much data do I need for training?**  
A: Minimum 90 days recommended (Phase 3 will handle this).

**Q: Can I run this on Linux/Mac?**  
A: Yes! Code is cross-platform. Just use forward slashes for paths.

---

## 🚀 Ready for Phase 3?

Once you've successfully tested Phase 2, you're ready to proceed to **Phase 3: Historical Data Backfill**.

Phase 3 will:
- Fetch 90-180 days of historical data
- Generate a complete training dataset
- Prepare for model training in Phase 5

**To proceed**: Let me know when you're ready, and I'll implement Phase 3!

---

**Need Help?** Check the documentation files or ask questions!
