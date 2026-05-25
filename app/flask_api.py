"""
Pearls AQI Predictor — Flask REST API

Endpoints:
  GET /api/predict        → 3-day AQI forecast as JSON
  GET /api/current        → current AQI & conditions
  GET /api/history        → historical AQI data
  GET /api/health         → health check
"""

import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'features.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')

AQI_CATEGORIES = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy for Sensitive", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]


def get_aqi_category(aqi: float) -> dict:
    for lo, hi, label, color in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return {"category": label, "color": color, "min": lo, "max": hi}
    return {"category": "Hazardous", "color": "#7e0023", "min": 301, "max": 500}


def load_history(days: int = 30) -> pd.DataFrame:
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    cutoff = datetime.now() - timedelta(days=days)
    df = df[df["timestamp"] >= cutoff].sort_values("timestamp")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "data_file_exists": os.path.exists(DATA_FILE),
        "models_dir_exists": os.path.exists(MODELS_DIR),
    })


@app.route("/api/current", methods=["GET"])
def current():
    """Return current AQI and weather conditions."""
    try:
        from inference import run as run_inference
        result = run_inference(models_dir=MODELS_DIR)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["GET"])
def predict():
    """Return 3-day AQI forecast."""
    try:
        from inference import run as run_inference
        result = run_inference(models_dir=MODELS_DIR)
        return jsonify({
            "predictions": result["predictions"],
            "model_info": result["model_info"],
            "generated_at": result["generated_at"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def history():
    """Return historical AQI data. Query param: ?days=7 (default 30)."""
    try:
        days = request.args.get("days", 30, type=int)
        days = min(max(days, 1), 365)
        df = load_history(days=days)
        if df.empty:
            return jsonify({"error": "No historical data found"}), 404

        records = []
        for _, row in df.iterrows():
            aqi_val = row.get("aqi")
            records.append({
                "timestamp": str(row["timestamp"]),
                "aqi": aqi_val,
                "category": get_aqi_category(aqi_val)["category"] if pd.notna(aqi_val) else None,
                "pm25": row.get("pm25"),
                "pm10": row.get("pm10"),
                "temperature": row.get("temperature"),
                "humidity": row.get("humidity"),
                "wind_speed": row.get("wind_speed"),
            })

        return jsonify({
            "days": days,
            "count": len(records),
            "records": records,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/shap/<filename>", methods=["GET"])
def shap_plot(filename):
    """Serve SHAP visualization PNGs from notebooks/ directory."""
    filepath = os.path.join(NOTEBOOKS_DIR, filename)
    if os.path.exists(filepath) and filename.endswith(".png"):
        return send_file(filepath, mimetype="image/png")
    return jsonify({"error": "File not found"}), 404


@app.route("/api/shap-list", methods=["GET"])
def shap_list():
    """List available SHAP visualization files."""
    p = Path(NOTEBOOKS_DIR)
    files = sorted(f.name for f in p.glob("shap_*.png"))
    return jsonify({"files": files})


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print(f"Starting AQI Predictor API on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=debug)
