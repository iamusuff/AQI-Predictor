"""
Pearls AQI Predictor — Premium Dashboard
Merged: Complete logic + refined glassmorphism design
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

DATA_FILE     = os.path.join(os.path.dirname(__file__), '..', 'data', 'features.csv')
MODELS_DIR    = os.path.join(os.path.dirname(__file__), '..', 'models')
NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')

# ─────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hawa Alert — AQI Karachi",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# Premium CSS Theme
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1280px;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    padding: 1.4rem !important;
    box-shadow: 0 2px 20px rgba(0,0,0,0.12) !important;
    transition: box-shadow 0.2s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 4px 28px rgba(0,0,0,0.18) !important;
}
div[data-testid="stMetricLabel"] p {
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    color: #888 !important;
    font-weight: 500 !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 22px !important;
}
section[data-testid="stSidebar"] {
    background: rgba(10,10,15,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
h2, h3, h4 {
    font-weight: 600 !important;
    letter-spacing: -0.3px !important;
}
div[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
}
hr {
    border-color: rgba(255,255,255,0.07) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
AQI_CATEGORIES = [
    (0,   50,  "Good",                    "#00e400", "Air quality is satisfactory"),
    (51,  100, "Moderate",                "#ffff00", "Acceptable; some pollutants may affect sensitive people"),
    (101, 150, "Unhealthy for Sensitive", "#ff7e00", "Sensitive groups should reduce outdoor activity"),
    (151, 200, "Unhealthy",               "#ff0000", "Everyone may experience health effects"),
    (201, 300, "Very Unhealthy",          "#8f3f97", "Health alert — avoid outdoor activity"),
    (301, 500, "Hazardous",               "#7e0023", "Health emergency — avoid all outdoor activity"),
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

def apply_chart_theme(fig, height=350):
    """Consistent dark transparent theme for all charts."""
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="DM Sans, sans-serif", size=12),
        hovermode="x unified",
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(255,255,255,0.1)',
            borderwidth=1,
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False, showline=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False, showline=False)
    return fig

# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='padding: 0.5rem 0 1rem 0;'>
    <div style='font-size:22px; font-weight:700; letter-spacing:-0.5px;'>🌍 Hawa Alert</div>
    <div style='font-size:13px; color:#666; margin-top:2px;'>Karachi, Pakistan · AQI Monitor</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Forecast Details", "Historical Trends", "Feature Importance", "Model Info"],
)

st.sidebar.markdown("---")
history_days = st.sidebar.selectbox("History Window", [7, 30, 90], index=1)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ─────────────────────────────────────────────────────────────────────
# Cache Helpers
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

    # ── Alert Banner ──────────────────────────────────────────────
    if current_aqi > 150:
        level = "🔴 RED ALERT" if current_aqi > 200 else "🟠 ORANGE ALERT"
        st.error(f"**{level} — AQI {current_aqi:.0f} ({cat}):** {desc}")
    elif current_aqi > 100:
        st.warning(f"**🟡 CAUTION — AQI {current_aqi:.0f} ({cat}):** {desc}")
    else:
        st.success(f"**✅ GOOD — AQI {current_aqi:.0f} ({cat}):** Air quality is satisfactory.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: AQI Hero + Forecast Chart ─────────────────────────
    col_left, col_right = st.columns([1, 2])

    with col_left:
        with st.container(border=True):
            st.markdown("#### Current Air Quality")
            st.markdown(
                f"<div style='background:{color}; padding:28px 20px; border-radius:12px; "
                f"text-align:center; box-shadow:0 4px 20px rgba(0,0,0,0.2); margin-bottom:18px;'>"
                f"<div style='color:white; font-size:52px; font-weight:700; line-height:1;'>{current_aqi:.0f}</div>"
                f"<div style='color:rgba(255,255,255,0.85); font-size:13px; font-weight:600; "
                f"text-transform:uppercase; letter-spacing:1.5px; margin-top:6px;'>{cat}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            m1, m2 = st.columns(2)
            with m1:
                temp = conditions.get('temperature')
                st.metric("Temperature", f"{temp:.1f}°C" if temp is not None else "—")
                wind = conditions.get('wind_speed')
                st.metric("Wind Speed", f"{wind:.1f} m/s" if wind is not None else "—")
            with m2:
                hum = conditions.get('humidity')
                st.metric("Humidity", f"{hum:.0f}%" if hum is not None else "—")
                pm25 = conditions.get('pm25')
                st.metric("PM2.5", f"{pm25:.1f} µg/m³" if pm25 is not None else "—")

    with col_right:
        with st.container(border=True):
            st.markdown("#### 📈 3-Day Forecast Horizon")
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
                x=fdf["label"], y=fdf["ci_upper"], mode="lines",
                line=dict(dash="dot", width=1, color="rgba(180,180,180,0.4)"),
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=fdf["label"], y=fdf["ci_lower"], mode="lines",
                line=dict(dash="dot", width=1, color="rgba(180,180,180,0.4)"),
                fill="tonexty", fillcolor="rgba(255,126,0,0.08)",
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=fdf["label"], y=fdf["aqi"], mode="lines+markers",
                name="AQI Forecast",
                line=dict(color="#ff7e00", width=3),
                marker=dict(size=10, color=fdf["aqi"].apply(aqi_color),
                            line=dict(width=2, color="white")),
            ))
            apply_chart_theme(fig, height=245)
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Environmental Details ─────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("#### 🌤️ Environmental Details")
        c1, c2, c3, c4 = st.columns(4)
        details = [
            (c1, "PM2.5",      conditions.get('pm25'),       "µg/m³"),
            (c2, "PM10",       conditions.get('pm10'),       "µg/m³"),
            (c3, "Pressure",   conditions.get('pressure'),   "hPa"),
            (c4, "Visibility", conditions.get('visibility'), "m"),
        ]
        for col, label, val, unit in details:
            col.metric(label, f"{val} {unit}" if val is not None else "—")

    # ── Row 3: Model Performance + SHAP ──────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_perf, col_shap = st.columns([1, 1])

    with col_perf:
        with st.container(border=True):
            st.markdown("#### 🤖 ML Model Performance")
            metrics = model_info.get("metrics", {})
            if metrics:
                mc1, mc2 = st.columns(2)
                mc1.metric("Model", model_info.get("name", "—").replace("_", " ").title())
                r2 = metrics.get('test_r2')
                mc2.metric("R² Score", f"{r2:.4f}" if isinstance(r2, (int, float)) else "—")
                st.markdown("---")
                mc3, mc4 = st.columns(2)
                rmse = metrics.get('test_rmse')
                mae  = metrics.get('test_mae')
                mc3.metric("RMSE", f"{rmse:.2f}" if isinstance(rmse, (int, float)) else "—")
                mc4.metric("MAE",  f"{mae:.2f}"  if isinstance(mae,  (int, float)) else "—")
            else:
                st.caption("Model metrics not available. Train a model first.")

    with col_shap:
        with st.container(border=True):
            st.markdown("#### 🎯 Feature Importance (SHAP)")
            shap_dir = Path(NOTEBOOKS_DIR)
            bar_png  = shap_dir / "shap_02_bar_plot.png"
            if bar_png.exists():
                st.image(str(bar_png), caption="Global Feature Attribution Weights", use_container_width=True)
            else:
                st.info("Run the SHAP analysis notebook to populate this panel.")
                st.markdown("**Top features (typical):**")
                st.markdown("- **PM2.5** — dominant pollutant\n- **Humidity** — weather factor\n- **PM10** — coarser particulate")


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

    st.markdown("## 📊 Detailed 3-Day Forecast")

    with st.container(border=True):
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
                "Status":     "🟢" if p["aqi"] <= 100 else ("🟡" if p["aqi"] <= 150 else ("🟠" if p["aqi"] <= 200 else "🔴")),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("#### AQI Distribution Per Horizon")
        fdf = pd.DataFrame([
            {"Period": label.upper(), "AQI": predictions[label]["aqi"], **predictions[label]}
            for label in ["current", "24h", "48h", "72h"]
        ])
        fig = px.bar(
            fdf, x="Period", y="AQI",
            color="AQI",
            color_continuous_scale=["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97"],
            text="AQI", text_auto=".0f",
        )
        fig.update_traces(textposition="outside")
        apply_chart_theme(fig, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ About This Forecast", expanded=False):
        forecast_method = model_info.get('forecast_method', 'Weather-informed')
        model_name      = model_info.get('name', 'Best model from Hopsworks Model Registry')
        st.markdown(f"""
**Forecasting Method:** {forecast_method}

- **Current (t+0)**: Real-time prediction using latest AQICN pollutants + OpenMeteo weather
- **24h / 48h / 72h**: TRUE multi-horizon predictions using:
  - OpenMeteo weather forecasts at target time
  - Physics-based pollutant persistence (wind dispersion, humidity trapping, temperature effects)
  - Model prediction with complete feature engineering for each horizon
- **Confidence Intervals**: 95% prediction intervals based on model test RMSE
  - Wider CIs for farther horizons (accounts for increasing uncertainty)
- **Model**: {model_name}

**NOT simple trend scaling** — each horizon gets an independent ML model prediction.
        """)


# ─────────────────────────────────────────────────────────────────────
# Historical Trends Page
# ─────────────────────────────────────────────────────────────────────
def render_history():
    df = get_history_data(history_days)
    if df.empty:
        st.warning("No historical data available. Run backfill first.")
        return

    st.markdown(f"## 📈 Historical AQI — Last {history_days} Days")
    st.caption(f"{len(df)} data points · {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    with st.container(border=True):
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["aqi"], mode="lines",
            name="Hourly AQI", line=dict(color="steelblue", width=1.5),
        ))
        if "aqi_rolling_24h" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["aqi_rolling_24h"], mode="lines",
                name="24h Rolling Avg", line=dict(color="darkorange", width=2.5),
            ))
        for lo, hi, label, color, _ in AQI_CATEGORIES:
            if hi <= df["aqi"].max() or lo <= df["aqi"].max():
                fig.add_hline(
                    y=hi, line_dash="dot", line_color=color, opacity=0.25,
                    annotation_text=label if hi <= df["aqi"].max() else None,
                    annotation_font_size=10,
                )
        apply_chart_theme(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_p, col_w = st.columns(2)
    with col_p:
        with st.container(border=True):
            st.markdown("#### 🧪 Pollutants Over Time")
            fig2 = go.Figure()
            for col in ["pm25", "pm10", "o3", "no2", "so2", "co"]:
                if col in df.columns:
                    fig2.add_trace(go.Scatter(
                        x=df["timestamp"], y=df[col], mode="lines",
                        name=col.upper(), line=dict(width=1.2),
                    ))
            apply_chart_theme(fig2, height=290)
            st.plotly_chart(fig2, use_container_width=True)

    with col_w:
        with st.container(border=True):
            st.markdown("#### 🌡️ Weather Conditions")
            fig3 = go.Figure()
            for col in ["temperature", "humidity", "wind_speed", "pressure"]:
                if col in df.columns:
                    fig3.add_trace(go.Scatter(
                        x=df["timestamp"], y=df[col], mode="lines",
                        name=col.replace("_", " ").title(), line=dict(width=1.2),
                    ))
            apply_chart_theme(fig3, height=290)
            st.plotly_chart(fig3, use_container_width=True)

    with st.expander("📊 Summary Statistics", expanded=False):
        stats = df["aqi"].describe()
        st.json({
            "Mean AQI":    f"{stats['mean']:.1f}",
            "Median AQI":  f"{stats['50%']:.1f}",
            "Min AQI":     f"{stats['min']:.0f}",
            "Max AQI":     f"{stats['max']:.0f}",
            "Std Dev":     f"{stats['std']:.1f}",
            "Data Points": int(stats["count"]),
        })


# ─────────────────────────────────────────────────────────────────────
# Feature Importance Page
# ─────────────────────────────────────────────────────────────────────
def render_shap():
    st.markdown("## 🎯 Feature Importance (SHAP Analysis)")
    st.markdown("Global feature importance from post-hoc SHAP analysis of the trained model.")

    shap_dir  = Path(NOTEBOOKS_DIR)
    png_files = sorted(shap_dir.glob("shap_*.png"))

    if not png_files:
        st.warning("No SHAP visualizations found. Run the SHAP analysis notebook first.")
        return

    # ── Summary + Bar ─────────────────────────────────────────────
    col1, col2  = st.columns(2)
    summary_png = shap_dir / "shap_01_summary_plot.png"
    bar_png     = shap_dir / "shap_02_bar_plot.png"

    with col1:
        with st.container(border=True):
            if summary_png.exists():
                st.image(str(summary_png), caption="SHAP Summary — Beeswarm Plot", use_container_width=True)
    with col2:
        with st.container(border=True):
            if bar_png.exists():
                st.image(str(bar_png), caption="Mean |SHAP| — Feature Attribution Bar Chart", use_container_width=True)

    # ── Dependence Plots ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔬 Feature Interactions")
    dep_files = sorted(shap_dir.glob("shap_04_dependence_*.png"))
    if dep_files:
        cols = st.columns(min(3, len(dep_files)))
        for i, f in enumerate(dep_files):
            with cols[i % 3]:
                st.image(str(f), caption=f.stem.replace("shap_04_dependence_", ""), use_container_width=True)
    else:
        st.info("Dependence plots not yet available.")

    # ── Waterfall Plots ───────────────────────────────────────────
    st.markdown("### 💧 Individual Prediction Explanations")
    waterfall_files = sorted(shap_dir.glob("shap_03_waterfall_*.png"))
    if waterfall_files:
        cols = st.columns(len(waterfall_files))
        for i, f in enumerate(waterfall_files):
            with cols[i]:
                st.image(str(f), caption=f"Sample {i+1}", use_container_width=True)

    # ── Alert Distribution ────────────────────────────────────────
    alert_png = shap_dir / "shap_05_alert_distribution.png"
    if alert_png.exists():
        st.markdown("### 🚨 Alert Distribution")
        with st.container(border=True):
            st.image(str(alert_png), use_container_width=True)

    # ── Cross-Model Comparison ────────────────────────────────────
    cross_png = shap_dir / "shap_06_cross_model_comparison.png"
    if cross_png.exists():
        st.markdown("### 📊 Cross-Model Feature Importance")
        with st.container(border=True):
            st.image(str(cross_png), use_container_width=True)

    # ── Alert Report CSV ──────────────────────────────────────────
    alert_csv = shap_dir / "shap_alert_report.csv"
    if alert_csv.exists():
        with st.expander("📋 Alert Report", expanded=False):
            alert_df = pd.read_csv(alert_csv)
            st.dataframe(alert_df, use_container_width=True)
            alert_summary = alert_df["alert_level"].value_counts().to_frame("Count")
            alert_summary["Percentage"] = (
                alert_summary["Count"] / len(alert_df) * 100
            ).round(1).astype(str) + "%"
            st.markdown("**Alert Level Summary**")
            st.dataframe(alert_summary, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# Model Info Page
# ─────────────────────────────────────────────────────────────────────
def render_model_info():
    st.markdown("## 🤖 Model Registry & Performance")

    try:
        result = get_inference()
    except Exception as e:
        st.error(f"Failed to load model info: {e}")
        return

    model_info = result["model_info"]
    metrics    = model_info["metrics"]   # nested: test_r2, test_rmse, test_mae, val_r2, val_rmse

    # ── Registry Snapshot JSON ────────────────────────────────────
    with st.container(border=True):
        st.markdown("#### Registry Snapshot")
        st.json({
            "Model Name":      model_info.get("name",            "N/A"),
            "Forecast Method": model_info.get("forecast_method", "N/A"),
            "Generated At":    result.get("generated_at",        "N/A"),
            "Test R²":         metrics.get("test_r2",            "N/A"),
            "Test RMSE":       metrics.get("test_rmse",          "N/A"),
            "Test MAE":        metrics.get("test_mae",           "N/A"),
            "Val R²":          metrics.get("val_r2",             "N/A"),
            "Val RMSE":        metrics.get("val_rmse",           "N/A"),
        })

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metric Cards ──────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("#### 📊 Performance Metrics")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Model",     model_info.get("name", "—").replace("_", " ").title())
        r2   = metrics.get("test_r2")
        rmse = metrics.get("test_rmse")
        mae  = metrics.get("test_mae")
        mc2.metric("Test R²",   f"{r2:.4f}"   if isinstance(r2,   (int, float)) else "—")
        mc3.metric("Test RMSE", f"{rmse:.4f}" if isinstance(rmse, (int, float)) else "—")
        mc4.metric("Test MAE",  f"{mae:.4f}"  if isinstance(mae,  (int, float)) else "—")

        st.markdown("---")
        v1, v2, v3 = st.columns(3)
        val_r2   = metrics.get("val_r2")
        val_rmse = metrics.get("val_rmse")
        v1.metric("Val R²",          f"{val_r2:.4f}"   if isinstance(val_r2,   (int, float)) else "—")
        v2.metric("Val RMSE",        f"{val_rmse:.4f}" if isinstance(val_rmse, (int, float)) else "—")
        v3.metric("Forecast Method", model_info.get("forecast_method", "—"))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Forecast Method Banner ────────────────────────────────────
    method = model_info.get("forecast_method", "N/A")
    if method == "Weather-informed":
        st.success(f"✅ **{method}** — Using OpenMeteo weather forecasts for 24h / 48h / 72h predictions")
    else:
        st.warning(f"⚠️ **{method}** — Weather forecast unavailable; using current conditions for all horizons")

    # ── SHAP Summary Report ───────────────────────────────────────
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

# ── Footer ────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.caption("🌍 Hawa Alert AQI Engine · Data: AQICN + OpenMeteo · Model Registry: Hopsworks Feature Store")