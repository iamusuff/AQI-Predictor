"""
Tests for inference.py — AQI Predictor Inference Module
"""

import pytest
import sys
import os
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import (
    get_aqi_category,
    prepare_inference_features,
    apply_pollutant_persistence_with_decay,
    predict_next_3_days,
)


# ─────────────────────────────────────────────────────────────────────────────
# get_aqi_category() Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAQICategory:

    def test_good_range(self):
        result = get_aqi_category(25)
        assert result["category"] == "Good"

    def test_moderate_range(self):
        result = get_aqi_category(75)
        assert result["category"] == "Moderate"

    def test_unhealthy_for_sensitive(self):
        result = get_aqi_category(130)
        assert "Sensitive" in result["category"]

    def test_unhealthy(self):
        result = get_aqi_category(175)
        assert result["category"] == "Unhealthy"

    def test_very_unhealthy(self):
        result = get_aqi_category(250)
        assert result["category"] == "Very Unhealthy"

    def test_hazardous(self):
        result = get_aqi_category(400)
        assert result["category"] == "Hazardous"

    def test_returns_dict_with_required_keys(self):
        result = get_aqi_category(100)
        for key in ["category", "color", "level", "advice"]:
            assert key in result, f"Missing key: {key}"

    def test_negative_aqi_clamped_to_zero(self):
        """Negative AQI should be treated as 0 (Good)."""
        result = get_aqi_category(-10)
        assert result["category"] == "Good"

    def test_level_increases_with_aqi(self):
        """Health level should increase as AQI worsens."""
        levels = [get_aqi_category(v)["level"] for v in [25, 75, 130, 175, 250, 400]]
        assert levels == sorted(levels), "Levels should be non-decreasing"

    def test_color_is_hex_string(self):
        result = get_aqi_category(50)
        assert result["color"].startswith("#")
        assert len(result["color"]) == 7

    def test_advice_is_non_empty_string(self):
        result = get_aqi_category(100)
        assert isinstance(result["advice"], str)
        assert len(result["advice"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# prepare_inference_features() Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPrepareInferenceFeatures:

    def test_returns_numpy_array(self):
        data = {"pm25": 50.0, "aqi": 120, "temperature": 30.0}
        feature_names = ["pm25", "aqi", "temperature"]
        result = prepare_inference_features(data, feature_names)
        assert isinstance(result, np.ndarray)

    def test_shape_is_1_by_n_features(self):
        feature_names = ["pm25", "pm10", "o3", "temperature"]
        data = {"pm25": 50.0, "pm10": 80.0, "o3": 30.0, "temperature": 28.0}
        result = prepare_inference_features(data, feature_names)
        assert result.shape == (1, 4)

    def test_feature_order_matches_feature_names(self):
        """Values must be placed in the exact order of feature_names."""
        feature_names = ["aqi", "pm25", "temperature"]
        data = {"aqi": 100, "pm25": 40.0, "temperature": 25.0}
        result = prepare_inference_features(data, feature_names)
        assert result[0, 0] == 100.0   # aqi
        assert result[0, 1] == 40.0    # pm25
        assert result[0, 2] == 25.0    # temperature

    def test_missing_feature_defaults_to_zero(self):
        """If a feature is not in data, should default to 0."""
        feature_names = ["pm25", "some_missing_feature"]
        data = {"pm25": 50.0}
        result = prepare_inference_features(data, feature_names)
        assert result[0, 1] == 0.0

    def test_none_value_defaults_to_zero(self):
        feature_names = ["pm25", "pm10"]
        data = {"pm25": None, "pm10": 80.0}
        result = prepare_inference_features(data, feature_names)
        assert result[0, 0] == 0.0

    def test_dtype_is_float64(self):
        feature_names = ["aqi"]
        data = {"aqi": 100}
        result = prepare_inference_features(data, feature_names)
        assert result.dtype == np.float64


# ─────────────────────────────────────────────────────────────────────────────
# apply_pollutant_persistence_with_decay() Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyPollutantPersistenceWithDecay:

    def current_pollutants(self):
        return {
            "pm25": 100.0,
            "pm10": 150.0,
            "o3":   50.0,
            "no2":  40.0,
            "so2":  20.0,
            "co":   10.0,
        }

    def sample_weather(self):
        return {
            "temperature": 25.0,
            "humidity":    50,
            "wind_speed":  5.0,
        }

    def test_returns_dict(self):
        result = apply_pollutant_persistence_with_decay(
            self.current_pollutants(), self.sample_weather(), 24
        )
        assert isinstance(result, dict)

    def test_all_pollutants_in_result(self):
        result = apply_pollutant_persistence_with_decay(
            self.current_pollutants(), self.sample_weather(), 24
        )
        for pol in ["pm25", "pm10", "o3", "no2", "so2", "co"]:
            assert pol in result

    def test_all_values_non_negative(self):
        """Decay should never produce negative pollutant levels."""
        result = apply_pollutant_persistence_with_decay(
            self.current_pollutants(), self.sample_weather(), 72
        )
        for pol, val in result.items():
            assert val >= 0, f"{pol} went negative: {val}"

    def test_pm25_decays_over_time(self):
        """PM2.5 at t+72 should be less than at t+24 (decay over time)."""
        r24 = apply_pollutant_persistence_with_decay(
            self.current_pollutants(), self.sample_weather(), 24
        )
        r72 = apply_pollutant_persistence_with_decay(
            self.current_pollutants(), self.sample_weather(), 72
        )
        assert r72["pm25"] <= r24["pm25"]

    def test_high_wind_reduces_pm25(self):
        """High wind speed should reduce PM2.5 more than calm conditions."""
        calm_weather  = {**self.sample_weather(), "wind_speed": 0.0}
        windy_weather = {**self.sample_weather(), "wind_speed": 15.0}

        r_calm  = apply_pollutant_persistence_with_decay(self.current_pollutants(), calm_weather,  24)
        r_windy = apply_pollutant_persistence_with_decay(self.current_pollutants(), windy_weather, 24)

        assert r_windy["pm25"] < r_calm["pm25"]

    def test_zero_pollutants_stay_zero(self):
        """Zero input → zero output (no negative or NaN)."""
        zero_pol = {k: 0.0 for k in ["pm25", "pm10", "o3", "no2", "so2", "co"]}
        result = apply_pollutant_persistence_with_decay(zero_pol, self.sample_weather(), 24)
        for pol, val in result.items():
            assert val == 0.0, f"{pol} should stay 0"


# ─────────────────────────────────────────────────────────────────────────────
# predict_next_3_days() Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictNext3Days:
    """
    Tests use a mocked model and scaler — no real ML model needed.
    """

    def make_mock_model(self, return_value=120.0):
        model = MagicMock()
        model.predict.return_value = np.array([return_value])
        return model

    def make_mock_scaler(self):
        scaler = MagicMock()
        scaler.transform.side_effect = lambda x: x  # Pass-through
        return scaler

    def feature_names(self):
        return ["pm25", "pm10", "o3", "no2", "so2", "co",
                "temperature", "humidity", "wind_speed",
                "hour", "day_of_week", "month", "is_weekend", "season"]

    def current_data(self):
        return {
            "pm25": 60.0, "pm10": 100.0, "o3": 40.0,
            "no2": 30.0, "so2": 15.0, "co": 5.0,
            "temperature": 28.0, "humidity": 65, "wind_speed": 8.0,
            "hour": 10, "day_of_week": 0, "month": 1,
            "is_weekend": 0, "season": 0, "aqi": 150,
        }

    def test_returns_dict(self):
        result = predict_next_3_days(
            self.make_mock_model(), self.make_mock_scaler(),
            self.feature_names(), self.current_data()
        )
        assert isinstance(result, dict)

    def test_has_all_horizon_keys(self):
        result = predict_next_3_days(
            self.make_mock_model(), self.make_mock_scaler(),
            self.feature_names(), self.current_data()
        )
        for key in ["current", "24h", "48h", "72h"]:
            assert key in result, f"Missing horizon key: {key}"

    def test_each_prediction_has_required_fields(self):
        result = predict_next_3_days(
            self.make_mock_model(), self.make_mock_scaler(),
            self.feature_names(), self.current_data()
        )
        required = ["aqi", "timestamp", "label", "confidence", "ci_lower", "ci_upper", "health"]
        for horizon, pred in result.items():
            for field in required:
                assert field in pred, f"'{horizon}' missing field: '{field}'"

    def test_aqi_values_are_non_negative(self):
        result = predict_next_3_days(
            self.make_mock_model(return_value=80.0), self.make_mock_scaler(),
            self.feature_names(), self.current_data()
        )
        for horizon, pred in result.items():
            assert pred["aqi"] >= 0, f"{horizon} AQI is negative"

    def test_confidence_interval_is_valid(self):
        """ci_lower must always be <= aqi <= ci_upper."""
        result = predict_next_3_days(
            self.make_mock_model(return_value=120.0), self.make_mock_scaler(),
            self.feature_names(), self.current_data()
        )
        for horizon, pred in result.items():
            assert pred["ci_lower"] <= pred["aqi"] <= pred["ci_upper"], (
                f"{horizon}: CI [{pred['ci_lower']}, {pred['ci_upper']}] "
                f"doesn't contain AQI {pred['aqi']}"
            )

    def test_model_predict_called_four_times(self):
        """Model should be called once per horizon: current + 24h + 48h + 72h."""
        mock_model = self.make_mock_model()
        predict_next_3_days(
            mock_model, self.make_mock_scaler(),
            self.feature_names(), self.current_data()
        )
        assert mock_model.predict.call_count == 4

    def test_works_without_weather_forecasts(self):
        """Should not crash when weather_forecasts=None (uses persistence model)."""
        result = predict_next_3_days(
            self.make_mock_model(), self.make_mock_scaler(),
            self.feature_names(), self.current_data(),
            weather_forecasts=None
        )
        assert "current" in result
        assert "72h"     in result

    def test_health_category_present_in_each_prediction(self):
        result = predict_next_3_days(
            self.make_mock_model(), self.make_mock_scaler(),
            self.feature_names(), self.current_data()
        )
        for horizon, pred in result.items():
            assert "category" in pred["health"]