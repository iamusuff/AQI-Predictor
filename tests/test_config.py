"""
Tests for config.py — AQI Predictor Configuration
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import (
    CITIES,
    CITY_CONFIG,
    POLLUTANT_COLUMNS,
    WEATHER_COLUMNS,
    ROLLING_WINDOWS,
    FORECAST_HORIZONS,
    AQI_CATEGORIES,
    FEATURE_GROUP_NAME,
    FEATURE_VIEW_NAME,
    MODEL_NAME,
    get_aqi_category,
    validate_config,
)


# ─────────────────────────────────────────────────────────────────────────────
# City Configuration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCityConfig:

    def test_cities_dict_has_required_keys(self):
        """All cities must have required config keys."""
        required_keys = ["name", "country", "lat", "lon", "aqicn_station", "timezone"]
        for city_name, city_data in CITIES.items():
            for key in required_keys:
                assert key in city_data, f"City '{city_name}' missing key: '{key}'"

    def test_karachi_exists(self):
        """Karachi must always be present as the default city."""
        assert "karachi" in CITIES

    def test_city_coordinates_are_valid(self):
        """Latitude must be -90 to 90, longitude -180 to 180."""
        for city_name, city_data in CITIES.items():
            assert -90 <= city_data["lat"] <= 90, f"{city_name} has invalid latitude"
            assert -180 <= city_data["lon"] <= 180, f"{city_name} has invalid longitude"

    def test_city_config_is_dict(self):
        """Active CITY_CONFIG must be a dict."""
        assert isinstance(CITY_CONFIG, dict)

    def test_city_config_has_lat_lon(self):
        """Active CITY_CONFIG must have lat and lon."""
        assert "lat" in CITY_CONFIG
        assert "lon" in CITY_CONFIG

    def test_karachi_coordinates_approximately_correct(self):
        """Karachi coordinates should be in the right ballpark."""
        karachi = CITIES["karachi"]
        assert 24 <= karachi["lat"] <= 26, "Karachi latitude looks wrong"
        assert 66 <= karachi["lon"] <= 68, "Karachi longitude looks wrong"


# ─────────────────────────────────────────────────────────────────────────────
# Feature Column Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureColumns:

    def test_pollutant_columns_not_empty(self):
        assert len(POLLUTANT_COLUMNS) > 0

    def test_pollutant_columns_contain_pm25(self):
        assert "pm25" in POLLUTANT_COLUMNS

    def test_pollutant_columns_contain_pm10(self):
        assert "pm10" in POLLUTANT_COLUMNS

    def test_weather_columns_not_empty(self):
        assert len(WEATHER_COLUMNS) > 0

    def test_weather_columns_contain_temperature(self):
        assert "temperature" in WEATHER_COLUMNS

    def test_weather_columns_contain_humidity(self):
        assert "humidity" in WEATHER_COLUMNS

    def test_rolling_windows_are_positive_ints(self):
        for w in ROLLING_WINDOWS:
            assert isinstance(w, int) and w > 0, f"Rolling window {w} is invalid"

    def test_rolling_windows_are_sorted(self):
        assert ROLLING_WINDOWS == sorted(ROLLING_WINDOWS), "Windows should be sorted ascending"

    def test_forecast_horizons_are_multiples_of_24(self):
        """Horizons should be 24, 48, 72 — daily steps."""
        for h in FORECAST_HORIZONS:
            assert h % 24 == 0, f"Horizon {h} is not a multiple of 24"


# ─────────────────────────────────────────────────────────────────────────────
# AQI Categories Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAQICategories:

    def test_aqi_categories_not_empty(self):
        assert len(AQI_CATEGORIES) > 0

    def test_aqi_categories_have_required_fields(self):
        for cat in AQI_CATEGORIES:
            assert "min" in cat
            assert "max" in cat
            assert "label" in cat
            assert "color" in cat

    def test_aqi_categories_start_at_zero(self):
        assert AQI_CATEGORIES[0]["min"] == 0

    def test_aqi_categories_no_gaps(self):
        """Each category's min should follow previous category's max."""
        for i in range(1, len(AQI_CATEGORIES)):
            prev_max = AQI_CATEGORIES[i - 1]["max"]
            curr_min = AQI_CATEGORIES[i]["min"]
            assert curr_min == prev_max + 1, (
                f"Gap between categories at index {i}: "
                f"prev max={prev_max}, curr min={curr_min}"
            )

    def test_aqi_colors_are_hex(self):
        """All category colors must be valid hex strings."""
        for cat in AQI_CATEGORIES:
            color = cat["color"]
            assert color.startswith("#"), f"Color '{color}' is not a hex string"
            assert len(color) == 7, f"Color '{color}' is not a 6-digit hex"


# ─────────────────────────────────────────────────────────────────────────────
# get_aqi_category() Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAQICategory:

    def test_good_aqi(self):
        result = get_aqi_category(25)
        assert result["label"] == "Good"

    def test_moderate_aqi(self):
        result = get_aqi_category(75)
        assert result["label"] == "Moderate"

    def test_unhealthy_for_sensitive(self):
        result = get_aqi_category(125)
        assert "Sensitive" in result["label"]

    def test_unhealthy_aqi(self):
        result = get_aqi_category(175)
        assert result["label"] == "Unhealthy"

    def test_very_unhealthy_aqi(self):
        result = get_aqi_category(250)
        assert result["label"] == "Very Unhealthy"

    def test_hazardous_aqi(self):
        result = get_aqi_category(400)
        assert result["label"] == "Hazardous"

    def test_boundary_value_50(self):
        """AQI=50 is the upper bound of 'Good'."""
        result = get_aqi_category(50)
        assert result["label"] == "Good"

    def test_boundary_value_51(self):
        """AQI=51 is the lower bound of 'Moderate'."""
        result = get_aqi_category(51)
        assert result["label"] == "Moderate"

    def test_returns_dict(self):
        result = get_aqi_category(100)
        assert isinstance(result, dict)

    def test_extreme_value_returns_something(self):
        """Extremely high AQI should not crash — fallback to Hazardous."""
        result = get_aqi_category(9999)
        assert result is not None
        assert "label" in result


# ─────────────────────────────────────────────────────────────────────────────
# validate_config() Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateConfig:

    def test_validate_config_raises_when_keys_missing(self, monkeypatch):
        """validate_config() must raise ValueError when API keys are absent."""
        import config
        monkeypatch.setattr(config, "AQICN_API_KEY", "")
        monkeypatch.setattr(config, "HOPSWORKS_API_KEY", "")

        with pytest.raises(ValueError) as exc_info:
            config.validate_config()

        assert "AQICN_API_KEY" in str(exc_info.value)

    def test_validate_config_returns_true_when_keys_set(self, monkeypatch):
        """validate_config() must return True when all keys are present."""
        import config
        monkeypatch.setattr(config, "AQICN_API_KEY", "dummy_key_123")
        monkeypatch.setattr(config, "HOPSWORKS_API_KEY", "dummy_hw_key_456")

        result = config.validate_config()
        assert result is True


# ─────────────────────────────────────────────────────────────────────────────
# Constants Sanity Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConstants:

    def test_feature_group_name_is_string(self):
        assert isinstance(FEATURE_GROUP_NAME, str) and len(FEATURE_GROUP_NAME) > 0

    def test_feature_view_name_is_string(self):
        assert isinstance(FEATURE_VIEW_NAME, str) and len(FEATURE_VIEW_NAME) > 0

    def test_model_name_is_string(self):
        assert isinstance(MODEL_NAME, str) and len(MODEL_NAME) > 0