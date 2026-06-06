"""
Pearls AQI Predictor — Interactive Streamlit Dashboard

Features:
- Real-time AQI with color-coded badge
- 3-day forecast with confidence intervals
- Current weather & pollutant conditions
- Historical trend charts (7 / 30 / 90 days)
- SHAP feature importance visualisation
- Health alert banners
- Model performance metrics
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'features.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Predictor — Karachi",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

AQI_CATEGORIES = [
    (0, 50, "Good", "#00e400", "Air quality is satisfactory"),
    (51, 100, "Moderate", "#ffff00", "Acceptable; some pollutants may affect sensitive people"),
    (101, 150, "Unhealthy for Sensitive", "#ff7e00", "Sensitive groups should reduce outdoor activity"),
    (151, 200, "Unhealthy", "#ff0000", "Everyone may experience health effects"),
    (201, 300, "Very Unhealthy", "#8f3f97", "Health alert — avoid outdoor activity"),
    (301, 500, "Hazardous", "#7e0023", "Health emergency — avoid all outdoor activity"),
]


def aqi_category(aqi):
    for lo, hi, label, color, desc in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return label, color, desc
    return "Hazardous", "#7e0023", "Health emergency"


def aqi_color(aqi):
    return aqi_category(aqi)[1]


def load_history(days=30):
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    cutoff = datetime.now() - timedelta(days=days)
    return df[df["timestamp"] >= cutoff].sort_values("timestamp").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────
st.sidebar.title("🌍 AQI Predictor")
st.sidebar.markdown("**Karachi, Pakistan**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Forecast Details", "Historical Trends", "Feature Importance", "Model Info"],
)

history_days = st.sidebar.selectbox("History window", [7, 30, 90], index=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Refresh data**")
if st.sidebar.button("🔄 Refresh Now"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ─────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_inference():
    from inference import run as run_inference
    return run_inference(models_dir=MODELS_DIR)


@st.cache_data(ttl=300)
def get_history_data(days):
    return load_history(days)


# ─────────────────────────────────────────────────────────────────────
# Dashboard Page
# ─────────────────────────────────────────────────────────────────────
def render_dashboard():
    try:
        result = get_inference()
    except Exception as e:
        st.error(f"Failed to fetch predictions: {e}")
        st.info("Make sure a model is trained and API keys are configured.")
        return

    predictions = result["predictions"]
    conditions  = result["current_conditions"]
    model_info  = result["model_info"]

    current_aqi = predictions["current"]["aqi"]
    cat, color, desc = aqi_category(current_aqi)

    # ── Alert banner ─────────────────────────────────────────────
    if current_aqi > 150:
        level = "🔴 RED ALERT" if current_aqi > 200 else "🟠 ORANGE ALERT"
        st.error(f"### {level} — AQI is {current_aqi:.0f} ({cat})")
        st.warning(desc)
    elif current_aqi > 100:
        st.warning(f"### 🟡 CAUTION — AQI is {current_aqi:.0f} ({cat})")
        st.caption(desc)
    else:
        st.success(f"### ✅ AQI is {current_aqi:.0f} ({cat}) — Air quality is satisfactory")

    # ── Header row ───────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown(
            f"<div style='background:{color}; padding:20px; border-radius:10px; text-align:center'>"
            f"<h1 style='color:white; margin:0'>{current_aqi:.0f}</h1>"
            f"<p style='color:white; margin:0; font-size:18px'>{cat}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.metric("Temperature", f"{conditions.get('temperature', '—'):.1f}°C")
        st.metric("Humidity",    f"{conditions.get('humidity',    '—'):.0f}%")
    with col3:
        st.metric("Wind Speed", f"{conditions.get('wind_speed', '—'):.1f} m/s")
        st.metric("PM2.5",      f"{conditions.get('pm25',       '—'):.1f} µg/m³")

    # ── 3-day forecast mini chart ────────────────────────────────
    st.subheader("📈 3-Day Forecast")
    forecast_data = []
    for label in ["current", "24h", "48h", "72h"]:
        p = predictions[label]
        forecast_data.append({
            "label":    label.upper(),
            "aqi":      p["aqi"],
            "ci_lower": p.get("ci_lower", p["aqi"] * 0.95),
            "ci_upper": p.get("ci_upper", p["aqi"] * 1.05),
        })

    fdf = pd.DataFrame(forecast_data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fdf["label"], y=fdf["aqi"], mode="lines+markers",
        name="Forecast", line=dict(color="#ff7e00", width=3),
        marker=dict(size=10, color=fdf["aqi"].apply(aqi_color)),
    ))
    fig.add_trace(go.Scatter(
        x=fdf["label"], y=fdf["ci_upper"], mode="lines",
        name="Upper CI", line=dict(dash="dot", width=1, color="gray"),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=fdf["label"], y=fdf["ci_lower"], mode="lines",
        name="Lower CI", line=dict(dash="dot", width=1, color="gray"),
        fill="tonexty", fillcolor="rgba(128,128,128,0.15)",
        showlegend=False,
    ))
    fig.update_layout(
        yaxis_title="AQI",
        hovermode="x unified",
        height=300, margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Current conditions details ───────────────────────────────
    with st.expander("🌤️ Current Conditions", expanded=False):
        cols = st.columns(4)
        labels = [
            ("PM2.5",      "pm25",       "µg/m³"),
            ("PM10",       "pm10",       "µg/m³"),
            ("Pressure",   "pressure",   "hPa"),
            ("Visibility", "visibility", "m"),
        ]
        for i, (label, key, unit) in enumerate(labels):
            val = conditions.get(key, "—")
            cols[i % 4].metric(label, f"{val} {unit}" if val != "—" else "—")

    # ── Model Performance ─────────────────────────────────────────
    st.subheader("🤖 Model Performance")
    metrics = model_info.get("metrics", {})
    if metrics:
        mc1, mc2, mc3, mc4 = st.columns(4)
        # KEY FIX: use "name" not "model_name" — matches inference.py response structure
        mc1.metric("Model", model_info.get("name", "—").replace("_", " ").title())
        mc2.metric("RMSE", f"{metrics.get('test_rmse', '—'):.4f}" if isinstance(metrics.get('test_rmse'), (int, float)) else "—")
        mc3.metric("MAE",  f"{metrics.get('test_mae',  '—'):.4f}" if isinstance(metrics.get('test_mae'),  (int, float)) else "—")
        mc4.metric("R²",   f"{metrics.get('test_r2',   '—'):.4f}" if isinstance(metrics.get('test_r2'),   (int, float)) else "—")
    else:
        st.caption("Model metrics not available. Train a model first.")

    # ── SHAP Feature Importance (Dashboard Summary) ────────────────
    st.subheader("🎯 Feature Importance (SHAP)")
    shap_dir      = Path(NOTEBOOKS_DIR)
    bar_png       = shap_dir / "shap_02_bar_plot.png"
    waterfall_files = sorted(shap_dir.glob("shap_03_waterfall_*.png"))
    if bar_png.exists():
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.image(str(bar_png), caption="Global Feature Importance", use_container_width=True)
        with col_right:
            if waterfall_files:
                st.image(str(waterfall_files[0]), caption="Sample Waterfall Explanation", use_container_width=True)
            st.markdown("**Top 3 Features:**")
            st.markdown("- **PM2.5** — dominant pollutant")
            st.markdown("- **Humidity** — weather factor")
            st.markdown("- **PM10** — coarser particulate")
    else:
        st.info("Run the SHAP analysis notebook to see feature importance (→ navigate to 'Feature Importance' page).")


# ─────────────────────────────────────────────────────────────────────
# Forecast Details Page
# ─────────────────────────────────────────────────────────────────────
def render_forecast():
    try:
        result = get_inference()
    except Exception as e:
        st.error(f"Failed to fetch predictions: {e}")
        return

    predictions = result["predictions"]
    model_info  = result["model_info"]
    st.subheader("📊 Detailed 3-Day Forecast")

    rows = []
    for label in ["current", "24h", "48h", "72h"]:
        p = predictions[label]
        cat, color, _ = aqi_category(p["aqi"])
        rows.append({
            "Period":     label.upper(),
            "AQI":        f"{p['aqi']:.1f}",
            "Category":   cat,
            "Confidence": p["confidence"].title(),
            "CI Lower":   f"{p.get('ci_lower', p['aqi']*0.95):.1f}",
            "CI Upper":   f"{p.get('ci_upper', p['aqi']*1.05):.1f}",
            "":           "🟢" if p["aqi"] <= 100 else ("🟡" if p["aqi"] <= 150 else ("🟠" if p["aqi"] <= 200 else "🔴")),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Bar chart
    fdf = pd.DataFrame([
        {"Period": label.upper(), "AQI": predictions[label]["aqi"], **predictions[label]}
        for label in ["current", "24h", "48h", "72h"]
    ])
    fig = px.bar(
        fdf, x="Period", y="AQI",
        color="AQI", color_continuous_scale=["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97"],
        text="AQI", text_auto=".0f",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ About the Forecast"):
        forecast_method = model_info.get('forecast_method', 'Weather-informed')
        st.markdown(f"""
        **Forecasting Method:** {forecast_method}

        - **Current (t+0)**: Real-time prediction using latest AQICN pollutants + OpenMeteo weather
        - **24h / 48h / 72h**: TRUE multi-horizon predictions using:
          - OpenMeteo weather forecasts at target time
          - Physics-based pollutant persistence (wind dispersion, humidity trapping, temperature effects)
          - Model prediction with complete feature engineering for each horizon
        - **Confidence intervals**: 95% prediction intervals based on model test RMSE
          - Wider CIs for farther horizons (accounts for increasing uncertainty)
        - **Model**: {model_info.get('name', 'Best model from Hopsworks Model Registry')}

        **NOT simple trend scaling** — each prediction uses the ML model independently!
        """)


# ─────────────────────────────────────────────────────────────────────
# Historical Trends Page
# ─────────────────────────────────────────────────────────────────────
def render_history():
    df = get_history_data(history_days)
    if df.empty:
        st.warning("No historical data available. Run backfill first.")
        return

    st.subheader(f"📈 Historical AQI — Last {history_days} Days")
    st.caption(f"{len(df)} data points | {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")

    # Time series
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["aqi"], mode="lines",
        name="Hourly AQI", line=dict(color="steelblue", width=1),
    ))
    if "aqi_rolling_24h" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["aqi_rolling_24h"], mode="lines",
            name="24h Rolling Avg", line=dict(color="darkorange", width=2),
        ))

    for lo, hi, label, color, _ in AQI_CATEGORIES:
        if hi <= df["aqi"].max() or lo <= df["aqi"].max():
            fig.add_hline(y=hi, line_dash="dot", line_color=color, opacity=0.3,
                          annotation_text=label if hi <= df["aqi"].max() else None)

    fig.update_layout(
        xaxis_title="Date", yaxis_title="AQI",
        hovermode="x unified", height=400,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Pollutant & weather subplots
    pollutant_cols = ["pm25", "pm10", "o3", "no2", "so2", "co"]
    weather_cols   = ["temperature", "humidity", "wind_speed", "pressure"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧪 Pollutants")
        fig2 = go.Figure()
        for col in pollutant_cols:
            if col in df.columns:
                fig2.add_trace(go.Scatter(
                    x=df["timestamp"], y=df[col], mode="lines",
                    name=col.upper(), line=dict(width=1),
                ))
        fig2.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("🌡️ Weather")
        fig3 = go.Figure()
        for col in weather_cols:
            if col in df.columns:
                fig3.add_trace(go.Scatter(
                    x=df["timestamp"], y=df[col], mode="lines",
                    name=col.replace("_", " ").title(), line=dict(width=1),
                ))
        fig3.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig3, use_container_width=True)

    # Summary stats
    with st.expander("📊 Summary Statistics", expanded=False):
        stats = df["aqi"].describe()
        st.json({
            "Mean AQI":   f"{stats['mean']:.1f}",
            "Median AQI": f"{stats['50%']:.1f}",
            "Min AQI":    f"{stats['min']:.0f}",
            "Max AQI":    f"{stats['max']:.0f}",
            "Std Dev":    f"{stats['std']:.1f}",
            "Data Points": int(stats["count"]),
        })


# ─────────────────────────────────────────────────────────────────────
# Feature Importance Page
# ─────────────────────────────────────────────────────────────────────
def render_shap():
    st.subheader("🎯 Feature Importance (SHAP)")
    st.markdown("Global feature importance from SHAP analysis of the trained model.")

    shap_dir  = Path(NOTEBOOKS_DIR)
    png_files = sorted(shap_dir.glob("shap_*.png"))

    if not png_files:
        st.warning("No SHAP visualizations found. Run the SHAP analysis notebook first.")
        return

    col1, col2   = st.columns(2)
    summary_png  = shap_dir / "shap_01_summary_plot.png"
    bar_png      = shap_dir / "shap_02_bar_plot.png"

    with col1:
        if summary_png.exists():
            st.image(str(summary_png), caption="SHAP Summary (Beeswarm)", use_container_width=True)
    with col2:
        if bar_png.exists():
            st.image(str(bar_png), caption="Mean |SHAP| (Bar Chart)", use_container_width=True)

    st.subheader("🔬 Feature Interactions")
    dep_files = sorted(shap_dir.glob("shap_04_dependence_*.png"))
    if dep_files:
        cols = st.columns(min(3, len(dep_files)))
        for i, f in enumerate(dep_files):
            with cols[i % 3]:
                st.image(str(f), caption=f.stem.replace("shap_04_dependence_", ""), use_container_width=True)
    else:
        st.info("Dependence plots not yet available.")

    st.subheader("💧 Individual Prediction Explanations")
    waterfall_files = sorted(shap_dir.glob("shap_03_waterfall_*.png"))
    if waterfall_files:
        cols = st.columns(len(waterfall_files))
        for i, f in enumerate(waterfall_files):
            with cols[i]:
                st.image(str(f), caption=f"Sample {i+1}", use_container_width=True)

    alert_png = shap_dir / "shap_05_alert_distribution.png"
    if alert_png.exists():
        st.subheader("🚨 Alert Distribution")
        st.image(str(alert_png), use_container_width=True)

    cross_png = shap_dir / "shap_06_cross_model_comparison.png"
    if cross_png.exists():
        st.subheader("📊 Cross-Model Feature Importance")
        st.image(str(cross_png), use_container_width=True)

    alert_csv = shap_dir / "shap_alert_report.csv"
    if alert_csv.exists():
        with st.expander("📋 Alert Report", expanded=False):
            alert_df = pd.read_csv(alert_csv)
            st.dataframe(alert_df, use_container_width=True)
            alert_summary = alert_df["alert_level"].value_counts().to_frame("Count")
            alert_summary["Percentage"] = (alert_summary["Count"] / len(alert_df) * 100).round(1).astype(str) + "%"
            st.subheader("Alert Level Summary")
            st.dataframe(alert_summary, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# Model Info Page  — reads from Hopsworks via get_inference()
# ─────────────────────────────────────────────────────────────────────
def render_model_info():
    st.subheader("🤖 Model Information")

    try:
        result = get_inference()
    except Exception as e:
        st.error(f"Failed to load model info: {e}")
        return

    model_info = result["model_info"]
    metrics    = model_info["metrics"]   # nested dict from inference.py

    # ── Summary JSON ─────────────────────────────────────────────
    st.json({
        "Model":           model_info.get("name",            "N/A"),
        "Forecast Method": model_info.get("forecast_method", "N/A"),
        "Generated At":    result.get("generated_at",        "N/A"),
        "Test R²":         metrics.get("test_r2",            "N/A"),
        "Test RMSE":       metrics.get("test_rmse",          "N/A"),
        "Test MAE":        metrics.get("test_mae",           "N/A"),
        "Val R²":          metrics.get("val_r2",             "N/A"),
        "Val RMSE":        metrics.get("val_rmse",           "N/A"),
    })

    # ── Metric cards ─────────────────────────────────────────────
    st.subheader("📊 Model Performance")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Model",     model_info.get("name", "—").replace("_", " ").title())
    mc2.metric("Test R²",   f"{metrics['test_r2']:.4f}"   if isinstance(metrics.get("test_r2"),   (int, float)) else "—")
    mc3.metric("Test RMSE", f"{metrics['test_rmse']:.4f}" if isinstance(metrics.get("test_rmse"), (int, float)) else "—")
    mc4.metric("Test MAE",  f"{metrics['test_mae']:.4f}"  if isinstance(metrics.get("test_mae"),  (int, float)) else "—")

    v1, v2 = st.columns(2)
    v1.metric("Val R²",   f"{metrics['val_r2']:.4f}"   if isinstance(metrics.get("val_r2"),   (int, float)) else "—")
    v2.metric("Val RMSE", f"{metrics['val_rmse']:.4f}" if isinstance(metrics.get("val_rmse"), (int, float)) else "—")

    # ── Forecast method banner ────────────────────────────────────
    st.subheader("🔮 Forecast Method")
    method = model_info.get("forecast_method", "N/A")
    if method == "Weather-informed":
        st.success(f"✅ **{method}** — Using OpenMeteo weather forecasts for 24h / 48h / 72h predictions")
    else:
        st.warning(f"⚠️ **{method}** — Weather forecast unavailable; using current conditions for all horizons")

    # ── SHAP report (optional) ────────────────────────────────────
    shap_report = os.path.join(NOTEBOOKS_DIR, "shap_summary_report.txt")
    if os.path.exists(shap_report):
        with st.expander("📝 SHAP Summary Report", expanded=False):
            with open(shap_report, encoding="utf-8", errors="replace") as f:
                st.text(f.read())


# ─────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────
if page == "Dashboard":
    render_dashboard()
elif page == "Forecast Details":
    render_forecast()
elif page == "Historical Trends":
    render_history()
elif page == "Feature Importance":
    render_shap()
elif page == "Model Info":
    render_model_info()

# ── Footer ─────────────────────────────────────────────────────────
st.markdown("---")
st.caption("🌍 Pearls AQI Predictor | Data sources: AQICN + OpenMeteo | Powered by Hopsworks")