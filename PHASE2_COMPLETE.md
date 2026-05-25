# Phase 2: Feature Pipeline Development ✅

## Completed Tasks

### 1. Utility Functions (`src/utils.py`)

Created comprehensive utility module with:

#### API Wrapper Functions
- ✅ **`fetch_aqicn_data()`** - Fetches real-time air quality data from AQICN API
  - Returns: AQI, PM2.5, PM10, O3, NO2, SO2, CO, dominant pollutant
  - Includes error handling and logging
  - Uses station ID from config

- ✅ **`fetch_openweather_data()`** - Fetches weather data from OpenWeatherMap API
  - Returns: Temperature, humidity, wind speed, pressure, visibility, clouds, weather description
  - Uses metric units (Celsius, m/s)
  - Includes error handling and logging

#### Feature Engineering Functions
- ✅ **`compute_time_features()`** - Extracts time-based features
  - Hour, day of week, day of month, month
  - Is weekend flag (0/1)
  - Season (0=Winter, 1=Spring, 2=Summer, 3=Fall)

- ✅ **`compute_derived_features()`** - Computes advanced features from raw data
  - **Rolling averages**: 3h, 6h, 12h, 24h for AQI, PM2.5, PM10
  - **AQI change rate**: 1-hour and 3-hour differences
  - **Pollutant ratios**: PM2.5/PM10, NO2/O3
  - **Weather interactions**: Temperature × Humidity, Wind × PM2.5

- ✅ **`compute_features()`** - Main feature computation orchestrator
  - Merges AQI and weather data
  - Adds time features
  - Returns complete feature dictionary

#### AQI Calculation Functions
- ✅ **`calculate_aqi_from_pollutant()`** - EPA standard AQI calculation
  - Implements EPA breakpoints for PM2.5 and PM10
  - Linear interpolation between breakpoints
  - Returns AQI value (0-500)

- ✅ **`compute_aqi_target()`** - Overall AQI from multiple pollutants
  - Takes maximum AQI across all pollutants
  - Fallback to API-provided AQI

#### Data Validation
- ✅ **`validate_feature_data()`** - Validates feature completeness and ranges
  - Checks required fields (timestamp, AQI, PM2.5, temperature, humidity)
  - Validates AQI range (0-500)
  - Validates temperature range (-50 to 60°C)
  - Validates humidity range (0-100%)

### 2. Feature Pipeline (`src/feature_pipeline.py`)

Created main pipeline orchestration with:

#### Hopsworks Integration
- ✅ **`connect_to_hopsworks()`** - Establishes connection to Hopsworks Feature Store
  - Uses API key from config
  - Graceful fallback if Hopsworks unavailable

- ✅ **`get_or_create_feature_group()`** - Manages feature group lifecycle
  - Gets existing feature group or creates new one
  - Defines schema with timestamp as primary key
  - Enables online feature serving

- ✅ **`insert_features_to_hopsworks()`** - Stores features in Feature Store
  - Batch insert with job waiting
  - Error handling and logging

#### Local Storage (Fallback)
- ✅ **`save_features_locally()`** - CSV-based local storage
  - Saves to `data/features.csv`
  - Appends to existing file
  - Removes duplicates (keeps latest)
  - Useful for development and backup

#### Main Pipeline Functions
- ✅ **`run()`** - Main pipeline execution
  - **Step 1**: Fetch AQI data from AQICN
  - **Step 2**: Fetch weather data from OpenWeather
  - **Step 3**: Compute and validate features
  - **Step 4**: Store in Hopsworks (optional)
  - **Step 5**: Save locally (optional)
  - Comprehensive logging at each step
  - Returns success/failure status

- ✅ **`run_batch()`** - Batch processing for multiple hours
  - Useful for catching up missed runs
  - Rate limiting to avoid API throttling

#### CLI Interface
- ✅ Command-line arguments:
  - `--no-hopsworks` - Skip Hopsworks storage
  - `--no-local` - Skip local storage
  - `--batch N` - Run for N hours back

### 3. Features Generated

The pipeline generates **30+ features** per timestamp:

#### Raw Features (13)
- `timestamp` - Event timestamp
- `aqi` - Overall Air Quality Index
- `pm25`, `pm10`, `o3`, `no2`, `so2`, `co` - Pollutant concentrations
- `dominentpol` - Dominant pollutant
- `temperature`, `humidity`, `wind_speed`, `pressure`, `visibility`, `clouds` - Weather features
- `weather_main` - Weather condition

#### Time Features (6)
- `hour` (0-23)
- `day_of_week` (0=Monday, 6=Sunday)
- `day_of_month` (1-31)
- `month` (1-12)
- `is_weekend` (0/1)
- `season` (0-3)

#### Derived Features (12+)
- `aqi_rolling_3h`, `aqi_rolling_6h`, `aqi_rolling_12h`, `aqi_rolling_24h`
- `pm25_rolling_3h`, `pm25_rolling_6h`, `pm25_rolling_12h`, `pm25_rolling_24h`
- `pm10_rolling_3h`, `pm10_rolling_6h`, `pm10_rolling_12h`, `pm10_rolling_24h`
- `aqi_change_1h`, `aqi_change_3h`
- `pm25_pm10_ratio`, `no2_o3_ratio`
- `temp_humidity_interaction`, `wind_pm25_interaction`

## Testing the Pipeline

### Prerequisites
Before running the pipeline, ensure you have:

1. **API Keys** - Create `.env` file with:
   ```bash
   AQICN_API_KEY=your_aqicn_key
   OPENWEATHER_API_KEY=your_openweather_key
   HOPSWORKS_API_KEY=your_hopsworks_key  # Optional for now
   CITY=karachi
   ```

2. **Dependencies** - Install required packages:
   ```bash
   pip install requests pandas numpy python-dotenv
   # Hopsworks optional for now
   ```

### Test Commands

#### 1. Test Utility Functions
```bash
python src/utils.py
```
This will:
- Fetch live AQI data
- Fetch live weather data
- Compute features
- Validate features

#### 2. Run Feature Pipeline (Local Only)
```bash
python src/feature_pipeline.py --no-hopsworks
```
This will:
- Fetch data from both APIs
- Compute all features
- Save to `data/features.csv`

#### 3. Run Feature Pipeline (With Hopsworks)
```bash
python src/feature_pipeline.py
```
This will:
- Fetch data from both APIs
- Compute all features
- Store in Hopsworks Feature Store
- Save local backup to `data/features.csv`

#### 4. Run Batch Mode (Catch Up)
```bash
python src/feature_pipeline.py --batch 24 --no-hopsworks
```
This will run the pipeline 24 times (useful for testing, but note: current APIs only provide real-time data).

#### 5. Use Main CLI
```bash
python main.py --pipeline feature
```
This runs the feature pipeline through the main CLI entry point.

## Expected Output

When running successfully, you should see:

```
============================================================
FEATURE PIPELINE STARTED
City: Karachi, Pakistan
Timestamp: 2024-01-15T10:00:00.000000
============================================================

[1/5] Fetching air quality data from AQICN...
✅ AQI: 156, PM2.5: 65.5, Dominant: pm25

[2/5] Fetching weather data from OpenWeather...
✅ Temp: 28.5°C, Humidity: 65%, Wind: 3.5 m/s

[3/5] Computing features...
✅ Generated 31 features

Sample features:
  timestamp: 2024-01-15 10:00:00
  aqi: 156
  pm25: 65.5
  temperature: 28.5
  humidity: 65
  hour: 10
  day_of_week: 0
  season: 0

[4/5] Storing features in Hopsworks Feature Store...
⚠️  Skipping Hopsworks storage (connection failed)

[5/5] Saving features locally (backup)...
✅ Saved features to data/features.csv

============================================================
FEATURE PIPELINE COMPLETED
  Hopsworks: ⚠️  Skipped/Failed
  Local Storage: ✅ Success
============================================================
```

## Data Storage

### Local Storage
- **Location**: `data/features.csv`
- **Format**: CSV with headers
- **Behavior**: Appends new data, removes duplicates by timestamp
- **Use Case**: Development, testing, backup

### Hopsworks Storage
- **Feature Group**: `aqi_features` (version 1)
- **Primary Key**: `timestamp`
- **Event Time**: `timestamp`
- **Online Enabled**: Yes (for real-time serving)
- **Use Case**: Production, model training, feature serving

## Known Limitations

1. **Historical Data**: Current implementation only fetches real-time data
   - For historical backfill, we need Phase 3 (backfill.py)
   - OpenWeather historical API requires paid plan or alternative sources

2. **Derived Features**: Rolling averages require historical data
   - First few rows will have limited rolling window data
   - Will improve as more data accumulates

3. **Hopsworks**: May not work on Windows without C++ Build Tools
   - Recommended to use cloud environment for Hopsworks
   - Local CSV storage works as fallback

4. **API Rate Limits**:
   - AQICN: ~1000 calls/day (free tier)
   - OpenWeather: 1000 calls/day (free tier)
   - Hourly pipeline = 24 calls/day (well within limits)

## Next Steps (Phase 3)

Before proceeding to Phase 3, you should:

1. **Test the Feature Pipeline**:
   ```bash
   python src/utils.py
   python src/feature_pipeline.py --no-hopsworks
   ```

2. **Verify Data**:
   - Check `data/features.csv` exists and has data
   - Verify all expected columns are present
   - Check data quality (no nulls in critical fields)

3. **Get API Keys** (if not already done):
   - AQICN: https://aqicn.org/data-platform/token/
   - OpenWeather: https://openweathermap.org/api

### Phase 3 Preview: Historical Data Backfill

Next phase will focus on:
1. **Historical Weather Data** (`src/backfill.py`)
   - Use OpenWeather historical API or alternative sources
   - Fetch 90-180 days of past data
   - Generate training dataset

2. **Data Sources**:
   - OpenWeather One Call API 3.0 (timemachine)
   - OpenAQ for historical air quality data
   - AQICN historical data (if available)

3. **Backfill Strategy**:
   - Loop through past dates
   - Fetch data for each timestamp
   - Compute features
   - Bulk insert to Feature Store
   - Save local backup

## Questions to Address

1. **API Keys**: Do you have AQICN and OpenWeather API keys ready?
2. **Hopsworks**: Should we set up Hopsworks account now, or continue with local storage?
3. **Target City**: Confirm Karachi is the target city, or switch to Lahore/Islamabad?
4. **Testing**: Should we test the pipeline now before proceeding to Phase 3?

---

**Status**: Phase 2 Complete ✅  
**Ready for**: Phase 3 - Historical Data Backfill  
**Blockers**: None (can proceed with local storage)
