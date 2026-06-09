"""
Tests for utils.py — AQI Predictor Utility Functions
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import (
    compute_time_features,
    get_season,
    compute_derived_features,
    compute_features,
    calculate_aqi_from_pollutant,
    compute_aqi_target,
    validate_feature_data,
)


# ─────────────────────────────────────────────────────────────────────────────
# get_season() Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGetSeason:

    def test_winter_months(self):
        for month in [12, 1, 2]:
            assert get_season(month) == 0, f"Month {month} should be Winter (0)"

    def test_spring_months(self):
        for month in [3, 4, 5]:
            assert get_season(month) == 1, f"Month {month} should be Spring (1)"

    def test_summer_months(self):
        for month in [6, 7, 8]:
            assert get_season(month) == 2, f"Month {month} should be Summer (2)"

    def test_fall_months(self):
        for month in [9, 10, 11]:
            assert get_season(month) == 3, f"Month {month} should be Fall (3)"

    def test_returns_int(self):
        assert isinstance(get_season(6), int)


# ─────────────────────────────────────────────────────────────────────────────
# compute_time_features() Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeTimeFeatures:

    def setup_method(self):
        # Monday 2024-01-15, 10:30 AM
        self.ts = datetime(2024, 1, 15, 10, 30, 0)

    def test_returns_dict(self):
        result = compute_time_features(self.ts)
        assert isinstance(result, dict)

    def test_hour_is_correct(self):
        result = compute_time_features(self.ts)
        assert result["hour"] == 10

    def test_day_of_week_monday(self):
        result = compute_time_features(self.ts)
        assert result["day_of_week"] == 0  # Monday = 0

    def test_day_of_month(self):
        result = compute_time_features(self.ts)
        assert result["day_of_month"] == 15

    def test_month(self):
        result = compute_time_features(self.ts)
        assert result["month"] == 1

    def test_is_weekend_false_on_monday(self):
        result = compute_time_features(self.ts)
        assert result["is_weekend"] == 0

    def test_is_weekend_true_on_saturday(self):
        saturday = datetime(2024, 1, 20, 10, 0, 0)  # Saturday
        result = compute_time_features(saturday)
        assert result["is_weekend"] == 1

    def test_is_weekend_true_on_sunday(self):
        sunday = datetime(2024, 1, 21, 10, 0, 0)  # Sunday
        result = compute_time_features(sunday)
        assert result["is_weekend"] == 1

    def test_season_is_winter_in_january(self):
        result = compute_time_features(self.ts)
        assert result["season"] == 0  # Winter

    def test_all_expected_keys_present(self):
        result = compute_time_features(self.ts)
        for key in ["hour", "day_of_week", "day_of_month", "month", "is_weekend", "season"]:
            assert key in result, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# compute_derived_features() Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeDerivedFeatures:

    def make_df(self, n=30):
        """Make a simple DataFrame with enough rows for rolling windows."""
        return pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "aqi":         np.random.randint(50, 200, n).astype(float),
            "pm25":        np.random.uniform(10, 80, n),
            "pm10":        np.random.uniform(20, 150, n),
            "o3":          np.random.uniform(10, 60, n),
            "no2":         np.random.uniform(5, 40, n),
            "temperature": np.random.uniform(20, 40, n),
            "humidity":    np.random.uniform(30, 90, n),
            "wind_speed":  np.random.uniform(0, 20, n),
        })

    def test_returns_dataframe(self):
        df = self.make_df()
        result = compute_derived_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_original_columns_preserved(self):
        df = self.make_df()
        result = compute_derived_features(df)
        for col in ["aqi", "pm25", "pm10"]:
            assert col in result.columns

    def test_rolling_aqi_columns_created(self):
        df = self.make_df()
        result = compute_derived_features(df)
        assert "aqi_rolling_3h"  in result.columns
        assert "aqi_rolling_24h" in result.columns

    def test_rolling_pm25_columns_created(self):
        df = self.make_df()
        result = compute_derived_features(df)
        assert "pm25_rolling_3h" in result.columns

    def test_aqi_change_columns_created(self):
        df = self.make_df()
        result = compute_derived_features(df)
        assert "aqi_change_1h" in result.columns
        assert "aqi_change_3h" in result.columns

    def test_pm25_pm10_ratio_created(self):
        df = self.make_df()
        result = compute_derived_features(df)
        assert "pm25_pm10_ratio" in result.columns

    def test_temp_humidity_interaction_created(self):
        df = self.make_df()
        result = compute_derived_features(df)
        assert "temp_humidity_interaction" in result.columns

    def test_no_extra_rows_added(self):
        df = self.make_df(30)
        result = compute_derived_features(df)
        assert len(result) == 30

    def test_pm25_pm10_ratio_no_division_by_zero(self):
        """pm10=0 should not crash — we add 1e-6."""
        df = self.make_df(10)
        df["pm10"] = 0.0
        result = compute_derived_features(df)
        assert result["pm25_pm10_ratio"].isnull().sum() == 0


# ─────────────────────────────────────────────────────────────────────────────
# calculate_aqi_from_pollutant() Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateAQIFromPollutant:

    def test_pm25_good_range(self):
        """PM2.5 = 5 µg/m³ → AQI should be 0–50 (Good)."""
        aqi = calculate_aqi_from_pollutant("pm25", 5.0)
        assert aqi is not None
        assert 0 <= aqi <= 50

    def test_pm25_moderate_range(self):
        """PM2.5 = 20 µg/m³ → AQI should be 51–100 (Moderate)."""
        aqi = calculate_aqi_from_pollutant("pm25", 20.0)
        assert aqi is not None
        assert 51 <= aqi <= 100

    def test_pm25_unhealthy_range(self):
        """PM2.5 = 100 µg/m³ → AQI should be 151–200 (Unhealthy)."""
        aqi = calculate_aqi_from_pollutant("pm25", 100.0)
        assert aqi is not None
        assert 151 <= aqi <= 200

    def test_pm10_good_range(self):
        aqi = calculate_aqi_from_pollutant("pm10", 20.0)
        assert aqi is not None
        assert 0 <= aqi <= 50

    def test_unsupported_pollutant_returns_none(self):
        """Unsupported pollutant should return None, not crash."""
        result = calculate_aqi_from_pollutant("methane", 50.0)
        assert result is None

    def test_returns_numeric(self):
        aqi = calculate_aqi_from_pollutant("pm25", 10.0)
        assert isinstance(aqi, (int, float))

    def test_very_high_concentration_returns_500(self):
        """Extremely high PM2.5 should cap at 500."""
        aqi = calculate_aqi_from_pollutant("pm25", 9999.0)
        assert aqi == 500


# ─────────────────────────────────────────────────────────────────────────────
# compute_aqi_target() Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeAQITarget:

    def test_returns_max_of_pollutants(self):
        """Overall AQI = max of individual pollutant AQIs."""
        data = {"pm25": 100.0, "pm10": 20.0}  # pm25 should dominate
        aqi = compute_aqi_target(data)
        pm25_aqi = calculate_aqi_from_pollutant("pm25", 100.0)
        assert aqi == pm25_aqi

    def test_fallback_to_api_aqi(self):
        """No pm25/pm10 → should fall back to aqi field."""
        data = {"pm25": None, "pm10": None, "aqi": 123}
        result = compute_aqi_target(data)
        assert result == 123

    def test_returns_float_or_int(self):
        data = {"pm25": 30.0, "pm10": 50.0}
        result = compute_aqi_target(data)
        assert isinstance(result, (int, float))

    def test_empty_dict_returns_zero(self):
        result = compute_aqi_target({})
        assert result == 0


# ─────────────────────────────────────────────────────────────────────────────
# validate_feature_data() Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateFeatureData:

    def valid_features(self):
        return {
            "timestamp":   datetime(2024, 1, 15, 10, 0),
            "aqi":         120,
            "pm25":        45.0,
            "temperature": 28.0,
            "humidity":    60,
        }

    def test_valid_data_passes(self):
        is_valid, msg = validate_feature_data(self.valid_features())
        assert is_valid is True
        assert msg == ""

    def test_missing_aqi_fails(self):
        f = self.valid_features()
        f["aqi"] = None
        is_valid, msg = validate_feature_data(f)
        assert is_valid is False
        assert "aqi" in msg

    def test_missing_pm25_fails(self):
        f = self.valid_features()
        f["pm25"] = None
        is_valid, msg = validate_feature_data(f)
        assert is_valid is False

    def test_aqi_out_of_range_fails(self):
        f = self.valid_features()
        f["aqi"] = 600  # > 500
        is_valid, msg = validate_feature_data(f)
        assert is_valid is False
        assert "AQI" in msg

    def test_temperature_too_low_fails(self):
        f = self.valid_features()
        f["temperature"] = -100  # below -50
        is_valid, msg = validate_feature_data(f)
        assert is_valid is False

    def test_temperature_too_high_fails(self):
        f = self.valid_features()
        f["temperature"] = 100  # above 60
        is_valid, msg = validate_feature_data(f)
        assert is_valid is False

    def test_humidity_out_of_range_fails(self):
        f = self.valid_features()
        f["humidity"] = 150  # > 100
        is_valid, msg = validate_feature_data(f)
        assert is_valid is False

    def test_returns_tuple(self):
        result = validate_feature_data(self.valid_features())
        assert isinstance(result, tuple)
        assert len(result) == 2


# ─────────────────────────────────────────────────────────────────────────────
# compute_features() Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeFeatures:

    def sample_aqi_data(self):
        return {
            "aqi":         156,
            "pm25":        65.5,
            "pm10":        120.3,
            "o3":          45.2,
            "no2":         32.1,
            "so2":         12.5,
            "co":          0.8,
            "timestamp":   "2024-01-15T10:00:00",
            "dominentpol": "pm25",
        }

    def sample_weather_data(self):
        return {
            "temperature": 28.0,
            "humidity":    65,
            "wind_speed":  10.5,
            "pressure":    1012,
            "visibility":  8000,
            "clouds":      40,
        }

    def test_returns_dict(self):
        result = compute_features(self.sample_aqi_data(), self.sample_weather_data())
        assert isinstance(result, dict)

    def test_contains_aqi(self):
        result = compute_features(self.sample_aqi_data(), self.sample_weather_data())
        assert "aqi" in result

    def test_contains_time_features(self):
        result = compute_features(self.sample_aqi_data(), self.sample_weather_data())
        for key in ["hour", "day_of_week", "month", "is_weekend", "season"]:
            assert key in result, f"Missing time feature: {key}"

    def test_contains_weather_features(self):
        result = compute_features(self.sample_aqi_data(), self.sample_weather_data())
        assert "temperature" in result
        assert "humidity"    in result

    def test_contains_pollutants(self):
        result = compute_features(self.sample_aqi_data(), self.sample_weather_data())
        for pol in ["pm25", "pm10", "o3", "no2", "so2", "co"]:
            assert pol in result

    def test_pm25_is_float(self):
        result = compute_features(self.sample_aqi_data(), self.sample_weather_data())
        assert isinstance(result["pm25"], float)

    def test_aqi_is_int(self):
        result = compute_features(self.sample_aqi_data(), self.sample_weather_data())
        assert isinstance(result["aqi"], int)

    def test_handles_none_weather_gracefully(self):
        """Missing weather fields should not crash — should get 0 or NaN."""
        result = compute_features(self.sample_aqi_data(), {})
        assert isinstance(result, dict)