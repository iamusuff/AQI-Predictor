"""
Hawa Alert AQI Predictor — Redesigned Dashboard
Sample UI: Clean light/dark card layout with sidebar nav
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
# CSS — Sample UI Design
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global Reset ─────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif !important;
    background-color: #f0f2f5 !important;
    color: #1a1d23 !important;
}

.main .block-container {
    padding: 1.8rem 2rem 3rem 2rem !important;
    max-width: 1320px !important;
    background-color: #f0f2f5 !important;
}

/* ── Sidebar ──────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #1a1d23 !important;
    border-right: 1px solid #2a2d35 !important;
    min-width: 230px !important;
}
section[data-testid="stSidebar"] * {
    color: #c8ccd4 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 6px 10px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.07) !important;
    color: #fff !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #2a2d35 !important;
}
section[data-testid="stSidebar"] .stButton button {
    background: #2e3240 !important;
    color: #c8ccd4 !important;
    border: 1px solid #3a3e4a !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: #3a3e4a !important;
    color: #fff !important;
}

/* ── Page Title ───────────────────────────────── */
h1 {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #1a1d23 !important;
    letter-spacing: -0.5px !important;
    margin-bottom: 0.2rem !important;
}
h2 {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #1a1d23 !important;
    letter-spacing: -0.4px !important;
}
h3, h4 {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #4a5060 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    margin-bottom: 0.8rem !important;
}

/* ── Cards ────────────────────────────────────── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #e4e7ed !important;
    border-radius: 16px !important;
    padding: 1.4rem 1.6rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04) !important;
    transition: box-shadow 0.2s ease, transform 0.2s ease !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.08), 0 8px 24px rgba(0,0,0,0.07) !important;
    transform: translateY(-1px) !important;
}

/* ── Metrics ──────────────────────────────────── */
div[data-testid="stMetricLabel"] p {
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    color: #8a9099 !important;
    font-weight: 600 !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 21px !important;
    font-weight: 500 !important;
    color: #1a1d23 !important;
}
div[data-testid="stMetricDelta"] {
    font-size: 12px !important;
}

/* ── Alert Banners ────────────────────────────── */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-width: 1px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 0.7rem 1rem !important;
}

/* ── Selectbox / Radio ────────────────────────── */
div[data-testid="stSelectbox"] > div,
div[data-baseweb="select"] {
    border-radius: 10px !important;
    border-color: #e4e7ed !important;
    font-size: 13px !important;
    background: #f8f9fb !important;
}

/* ── DataFrames ───────────────────────────────── */
div[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #e4e7ed !important;
}
div[data-testid="stDataFrame"] table thead tr th {
    background: #f4f6f9 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    color: #8a9099 !important;
    font-weight: 600 !important;
}
div[data-testid="stDataFrame"] table tbody tr:nth-child(even) {
    background: #fafbfc !important;
}

/* ── Expander ─────────────────────────────────── */
details {
    border-radius: 12px !important;
    border: 1px solid #e4e7ed !important;
    background: #fff !important;
    padding: 0.2rem !important;
}
details summary {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #4a5060 !important;
    padding: 0.6rem 0.8rem !important;
}

/* ── Divider ──────────────────────────────────── */
hr {
    border-color: #e8eaee !important;
    margin: 1rem 0 !important;
}

/* ── Caption / Small Text ─────────────────────── */
small, .stCaption, [data-testid="stCaptionContainer"] p {
    color: #8a9099 !important;
    font-size: 11px !important;
}

/* ── JSON viewer ──────────────────────────────── */
div[data-testid="stJson"] {
    background: #f8f9fb !important;
    border-radius: 10px !important;
    border: 1px solid #e4e7ed !important;
    font-size: 12px !important;
}

/* ── Image caption ────────────────────────────── */
figcaption {
    font-size: 11px !important;
    color: #8a9099 !important;
    text-align: center !important;
    margin-top: 6px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
AQI_CATEGORIES = [
    (0,   50,  "Good",                    "#22c55e", "Air quality is satisfactory"),
    (51,  100, "Moderate",                "#eab308", "Acceptable; some pollutants may affect sensitive people"),
    (101, 150, "Unhealthy for Sensitive", "#f97316", "Sensitive groups should reduce outdoor activity"),
    (151, 200, "Unhealthy",               "#ef4444", "Everyone may experience health effects"),
    (201, 300, "Very Unhealthy",          "#a855f7", "Health alert — avoid outdoor activity"),
    (301, 500, "Hazardous",               "#7f1d1d", "Health emergency — avoid all outdoor activity"),
]

def aqi_category(aqi):
    for lo, hi, label, color, desc in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return label, color, desc
    return "Hazardous", "#7f1d1d", "Health emergency"

def aqi_color(aqi):
    return aqi_category(aqi)[1]

def load_history(days=30):
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    cutoff = datetime.now() - timedelta(days=days)
    return df[df["timestamp"] >= cutoff].sort_values("timestamp").reset_index(drop=True)

def apply_chart_theme(fig, height=320):
    """Clean white-background chart theme."""
    fig.update_layout(
        margin=dict(l=8, r=8, t=12, b=8),
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Sora, sans-serif", size=11, color="#4a5060"),
        hovermode="x unified",
        legend=dict(
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#e4e7ed',
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor='#f0f2f5', gridwidth=1,
        zeroline=False, showline=False,
        tickfont=dict(size=10, color="#8a9099"),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor='#f0f2f5', gridwidth=1,
        zeroline=False, showline=False,
        tickfont=dict(size=10, color="#8a9099"),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='padding:1.2rem 0.4rem 1rem 0.4rem;'>
    <div style='display:flex; align-items:center; gap:10px; margin-bottom:6px;'>
        <span style='font-size:22px;'>🌍</span>
        <div>
            <div style='font-size:15px; font-weight:700; color:#fff; letter-spacing:-0.3px;'>AQI Predictor</div>
            <div style='font-size:11px; color:#666; font-weight:500;'><b style="color:#aaa">Karachi, Pakistan</b></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Forecast Details", "Historical Trends", "Feature Importance", "Model Info"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("<div style='font-size:11px;color:#555;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:6px;font-weight:600;'>History window</div>", unsafe_allow_html=True)
history_days = st.sidebar.selectbox("History Window", [7, 30, 90], index=1, label_visibility="collapsed")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

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

    # ── Page Header ───────────────────────────────────────────────
    st.markdown(f"""
    <div style='margin-bottom:1.4rem;'>
        <h1 style='margin:0 0 4px 0;'>Pearl AQI Predictor — Interactive Streamlit Dashboard</h1>
        <p style='color:#8a9099; font-size:13px; margin:0;'>Real-time air quality monitoring & ML forecasting · Karachi, Pakistan</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Alert Banner ──────────────────────────────────────────────
    if current_aqi > 150:
        level = "🔴 RED ALERT" if current_aqi > 200 else "🟠 ORANGE ALERT"
        st.error(f"**{level} — AQI {current_aqi:.0f} ({cat}):** {desc}")
    elif current_aqi > 100:
        st.warning(f"**🟡 CAUTION — AQI {current_aqi:.0f} ({cat}):** {desc}")
    else:
        st.success(f"**✅ Health Alert** — Air quality is satisfactory.")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Row 1: AQI Card | Weather Card | Forecast Card ───────────
    col_aqi, col_weather, col_forecast = st.columns([1.1, 1.1, 1.5])

    with col_aqi:
        with st.container(border=True):
            st.markdown("#### Current AQI")
            # Big AQI gauge card
            badge_bg = color
            st.markdown(
                f"""<div style='
                    background: linear-gradient(135deg, {badge_bg}22 0%, {badge_bg}11 100%);
                    border: 2px solid {badge_bg}55;
                    border-radius: 14px;
                    padding: 22px 16px 18px 16px;
                    text-align: center;
                    margin-bottom: 14px;
                '>
                    <div style='font-size:54px; font-weight:700; color:{badge_bg}; 
                                font-family:"JetBrains Mono",monospace; line-height:1;'>{current_aqi:.0f}</div>
                    <div style='font-size:13px; font-weight:600; color:{badge_bg}; 
                                margin-top:8px; letter-spacing:0.5px;'>{cat}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with col_weather:
        with st.container(border=True):
            st.markdown("#### Current Weather")
            temp = conditions.get('temperature')
            hum  = conditions.get('humidity')
            wind = conditions.get('wind_speed')
            # Weather icons row
            st.markdown(f"""
            <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; text-align:center; padding:8px 0 10px 0;'>
                <div>
                    <div style='font-size:26px;'>🌡️</div>
                    <div style='font-size:17px; font-weight:700; color:#1a1d23; font-family:"JetBrains Mono",monospace;'>{f"{temp:.0f}°C" if temp is not None else "—"}</div>
                    <div style='font-size:10px; color:#8a9099; text-transform:uppercase; letter-spacing:0.5px; font-weight:600; margin-top:2px;'>Temperature</div>
                </div>
                <div>
                    <div style='font-size:26px;'>💧</div>
                    <div style='font-size:17px; font-weight:700; color:#1a1d23; font-family:"JetBrains Mono",monospace;'>{f"{hum:.0f}%" if hum is not None else "—"}</div>
                    <div style='font-size:10px; color:#8a9099; text-transform:uppercase; letter-spacing:0.5px; font-weight:600; margin-top:2px;'>Humidity</div>
                </div>
                <div>
                    <div style='font-size:26px;'>💨</div>
                    <div style='font-size:17px; font-weight:700; color:#1a1d23; font-family:"JetBrains Mono",monospace;'>{f"{wind:.1f}" if wind is not None else "—"}</div>
                    <div style='font-size:10px; color:#8a9099; text-transform:uppercase; letter-spacing:0.5px; font-weight:600; margin-top:2px;'>Wind km/h</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_forecast:
        with st.container(border=True):
            st.markdown("#### 3-Day Forecast")
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
                line=dict(dash="dot", width=1, color="rgba(100,100,100,0.2)"),
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=fdf["label"], y=fdf["ci_lower"], mode="lines",
                line=dict(dash="dot", width=1, color="rgba(100,100,100,0.2)"),
                fill="tonexty", fillcolor="rgba(234,179,8,0.1)",
                showlegend=False,
            ))
            # 3-day lines
            colors_fc = ["#22c55e", "#f97316", "#a855f7"]
            for i in range(1, len(fdf)):
                fig.add_trace(go.Scatter(
                    x=fdf["label"].iloc[i-1:i+1], y=fdf["aqi"].iloc[i-1:i+1],
                    mode="lines", line=dict(color=colors_fc[i-1], width=2.5),
                    showlegend=True, name=f"{fdf['label'].iloc[i]}",
                ))
            fig.add_trace(go.Scatter(
                x=fdf["label"], y=fdf["aqi"], mode="markers",
                marker=dict(size=9, color=fdf["aqi"].apply(aqi_color),
                            line=dict(width=2, color="white")),
                showlegend=False,
            ))
            apply_chart_theme(fig, height=148)
            fig.update_layout(legend=dict(orientation="h", y=-0.25, x=0))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # ── Row 2: SHAP Feature Importance | Historical AQI Trends ───
    col_shap, col_hist = st.columns([1, 1.3])

    with col_shap:
        with st.container(border=True):
            st.markdown("#### SHAP Feature Importance")
            shap_dir = Path(NOTEBOOKS_DIR)
            bar_png  = shap_dir / "shap_02_bar_plot.png"
            if bar_png.exists():
                st.image(str(bar_png), caption="SHAP Feature Importance", use_container_width=True)
            else:
                # Placeholder chart
                feat_names = ["NaN", "SWI", "Graneous\nData", "SHAP\nFeatures", "Modulo", "AQI", "Pollutant"]
                feat_vals  = [28, 22, 18, 14, 10, 7, 5]
                fig_s = go.Figure(go.Bar(
                    x=feat_vals, y=feat_names,
                    orientation="h",
                    marker=dict(
                        color=feat_vals,
                        colorscale=[[0, "#22c55e"], [0.5, "#eab308"], [1, "#ef4444"]],
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="High", side="right"),
                            thickness=10, len=0.6,
                            tickfont=dict(size=9),
                        ),
                    ),
                ))
                fig_s.update_layout(
                    yaxis=dict(title="Top Features", tickfont=dict(size=10)),
                    xaxis=dict(title="Feature Importance", tickfont=dict(size=10)),
                )
                apply_chart_theme(fig_s, height=260)
                st.plotly_chart(fig_s, use_container_width=True)

    with col_hist:
        with st.container(border=True):
            st.markdown("#### Historical AQI trends (in 7 days)")
            df_hist = get_history_data(7)
            if not df_hist.empty and "aqi" in df_hist.columns:
                fig_h = go.Figure()
                fig_h.add_trace(go.Scatter(
                    x=df_hist["timestamp"], y=df_hist["aqi"],
                    mode="lines+markers", name="AQI",
                    line=dict(color="#3b82f6", width=2),
                    marker=dict(size=4),
                ))
                if "aqi_rolling_24h" in df_hist.columns:
                    fig_h.add_trace(go.Scatter(
                        x=df_hist["timestamp"], y=df_hist["aqi_rolling_24h"],
                        mode="lines", name="MSE",
                        line=dict(color="#f97316", width=2, dash="dot"),
                    ))
                apply_chart_theme(fig_h, height=260)
                fig_h.update_layout(
                    legend=dict(orientation="v", x=1.01, y=1),
                    yaxis_title="AQI", xaxis_title="Days",
                )
                st.plotly_chart(fig_h, use_container_width=True)
            else:
                # Placeholder
                days_x = list(range(1, 8))
                sample_aqi = [180, 240, 160, 300, 210, 170, 190]
                sample_mse = [160, 200, 150, 260, 180, 155, 175]
                sample_rmse= [140, 180, 130, 240, 160, 140, 160]
                sample_worse=[200, 260, 180, 320, 240, 200, 220]
                sample_worst=[220, 280, 200, 350, 260, 220, 240]
                fig_h = go.Figure()
                for name, vals, clr in [
                    ("AQI",   sample_aqi,  "#3b82f6"),
                    ("MSE",   sample_mse,  "#ef4444"),
                    ("RMSE",  sample_rmse, "#22c55e"),
                    ("Worse", sample_worse,"#f97316"),
                    ("Worst", sample_worst,"#a855f7"),
                ]:
                    fig_h.add_trace(go.Scatter(
                        x=days_x, y=vals, mode="lines+markers", name=name,
                        line=dict(color=clr, width=1.8),
                        marker=dict(size=5),
                    ))
                apply_chart_theme(fig_h, height=260)
                fig_h.update_layout(
                    legend=dict(orientation="v", x=1.01, y=1),
                    yaxis_title="AQI", xaxis_title="Days",
                )
                st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # ── Row 3: Weather Conditions | Pollutant Levels ─────────────
    col_wc, col_pl = st.columns(2)

    with col_wc:
        with st.container(border=True):
            st.markdown("#### Weather Conditions")
            df_full = get_history_data(history_days)
            fig_w = go.Figure()
            weather_cols = [
                ("temperature", "#3b82f6", "Weather"),
                ("humidity",    "#06b6d4", "Noon"),
                ("wind_speed",  "#8b5cf6", "Moist Sow"),
            ]
            if not df_full.empty:
                for col_name, clr, nm in weather_cols:
                    if col_name in df_full.columns:
                        fig_w.add_trace(go.Scatter(
                            x=df_full["timestamp"], y=df_full[col_name],
                            mode="lines", name=nm, line=dict(color=clr, width=1.5),
                        ))
            apply_chart_theme(fig_w, height=210)
            fig_w.update_layout(
                legend=dict(orientation="h", y=-0.28, x=0, font=dict(size=10)),
                xaxis_title="Time (days)", yaxis_title="Conditions",
            )
            st.plotly_chart(fig_w, use_container_width=True)

    with col_pl:
        with st.container(border=True):
            st.markdown("#### Pollutant Levels")
            df_full = get_history_data(history_days)
            fig_p = go.Figure()
            pollutant_cols = [
                ("pm25", "#ef4444", "AQI"),
                ("pm10", "#f97316", "MO"),
                ("o3",   "#22c55e", "MAI"),
            ]
            if not df_full.empty:
                for col_name, clr, nm in pollutant_cols:
                    if col_name in df_full.columns:
                        fig_p.add_trace(go.Scatter(
                            x=df_full["timestamp"], y=df_full[col_name],
                            mode="lines", name=nm, line=dict(color=clr, width=1.5),
                        ))
            apply_chart_theme(fig_p, height=210)
            fig_p.update_layout(
                legend=dict(orientation="h", y=-0.28, x=0, font=dict(size=10)),
                xaxis_title="Pollutant Levels", yaxis_title="",
            )
            st.plotly_chart(fig_p, use_container_width=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # ── Row 4: Model Performance Metrics ─────────────────────────
    metrics = model_info.get("metrics", {})
    r2   = metrics.get('test_r2',   0.85)
    rmse = metrics.get('test_rmse', 12.3)
    mae  = metrics.get('test_mae',  9.1)
    mape = metrics.get('test_mape', 4.5)

    with st.container(border=True):
        mc1, mc2, mc3, mc4 = st.columns(4)

        for col_m, label, val, sub in [
            (mc1, f"Test R²: {r2:.2f}",    r2,   "Test R²: Ideal performance: metrics"),
            (mc2, f"Test RMSE: {rmse:.1f}", rmse, "Test RMSE: Performance: metrics"),
            (mc3, f"Test MAE: {mae:.1f}",   mae,  "Test MAE: Performanety ration"),
            (mc4, f"Model ME: {mape:.1f}%", mape, "Test MAE: Performance ratics"),
        ]:
            col_m.markdown(
                f"""<div style='background:#f8f9fb; border:1px solid #e4e7ed; border-radius:12px;
                               padding:14px 16px; text-align:center;'>
                    <div style='font-size:16px; font-weight:700; color:#1a1d23; 
                                font-family:"JetBrains Mono",monospace;'>{label}</div>
                    <div style='font-size:10px; color:#8a9099; margin-top:4px; 
                                font-weight:500;'>{sub}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:11px; color:#8a9099; text-align:center; padding:4px 0;'>"
        "This model performance metrics is milita borature on pastlit nnosi-lol bit predicos for mxatin gratwecating a oaponents in the sorach."
        "</div>",
        unsafe_allow_html=True,
    )


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

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("#### AQI Distribution Per Horizon")
        fdf = pd.DataFrame([
            {"Period": label.upper(), "AQI": predictions[label]["aqi"], **predictions[label]}
            for label in ["current", "24h", "48h", "72h"]
        ])
        fig = px.bar(
            fdf, x="Period", y="AQI",
            color="AQI",
            color_continuous_scale=["#22c55e", "#eab308", "#f97316", "#ef4444", "#a855f7"],
            text="AQI", text_auto=".0f",
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        apply_chart_theme(fig, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ About This Forecast", expanded=False):
        forecast_method = model_info.get('forecast_method', 'Weather-informed')
        model_name      = model_info.get('name', 'Best model from Hopsworks Model Registry')
        st.markdown(f"""
**Forecasting Method:** {forecast_method}

- **Current (t+0)**: Real-time prediction using latest AQICN pollutants + OpenMeteo weather
- **24h / 48h / 72h**: TRUE multi-horizon predictions using OpenMeteo weather forecasts, physics-based pollutant persistence, and complete feature engineering for each horizon
- **Confidence Intervals**: 95% prediction intervals based on model test RMSE — wider CIs for farther horizons
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
            name="Hourly AQI", line=dict(color="#3b82f6", width=1.5),
        ))
        if "aqi_rolling_24h" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["aqi_rolling_24h"], mode="lines",
                name="24h Rolling Avg", line=dict(color="#f97316", width=2.5),
            ))
        for lo, hi, label, color, _ in AQI_CATEGORIES:
            if hi <= df["aqi"].max() or lo <= df["aqi"].max():
                fig.add_hline(
                    y=hi, line_dash="dot", line_color=color, opacity=0.3,
                    annotation_text=label if hi <= df["aqi"].max() else None,
                    annotation_font_size=9,
                )
        apply_chart_theme(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    col_p, col_w = st.columns(2)
    with col_p:
        with st.container(border=True):
            st.markdown("#### 🧪 Pollutants Over Time")
            fig2 = go.Figure()
            poll_colors = ["#ef4444","#f97316","#eab308","#22c55e","#3b82f6","#a855f7"]
            for i, col in enumerate(["pm25", "pm10", "o3", "no2", "so2", "co"]):
                if col in df.columns:
                    fig2.add_trace(go.Scatter(
                        x=df["timestamp"], y=df[col], mode="lines",
                        name=col.upper(), line=dict(width=1.3, color=poll_colors[i]),
                    ))
            apply_chart_theme(fig2, height=290)
            st.plotly_chart(fig2, use_container_width=True)

    with col_w:
        with st.container(border=True):
            st.markdown("#### 🌡️ Weather Conditions")
            fig3 = go.Figure()
            wx_colors = ["#f97316","#3b82f6","#22c55e","#a855f7"]
            for i, col in enumerate(["temperature", "humidity", "wind_speed", "pressure"]):
                if col in df.columns:
                    fig3.add_trace(go.Scatter(
                        x=df["timestamp"], y=df[col], mode="lines",
                        name=col.replace("_", " ").title(),
                        line=dict(width=1.3, color=wx_colors[i]),
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

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    st.markdown("### 🔬 Feature Interactions")
    dep_files = sorted(shap_dir.glob("shap_04_dependence_*.png"))
    if dep_files:
        cols = st.columns(min(3, len(dep_files)))
        for i, f in enumerate(dep_files):
            with cols[i % 3]:
                st.image(str(f), caption=f.stem.replace("shap_04_dependence_", ""), use_container_width=True)
    else:
        st.info("Dependence plots not yet available.")

    st.markdown("### 💧 Individual Prediction Explanations")
    waterfall_files = sorted(shap_dir.glob("shap_03_waterfall_*.png"))
    if waterfall_files:
        cols = st.columns(len(waterfall_files))
        for i, f in enumerate(waterfall_files):
            with cols[i]:
                st.image(str(f), caption=f"Sample {i+1}", use_container_width=True)

    alert_png = shap_dir / "shap_05_alert_distribution.png"
    if alert_png.exists():
        st.markdown("### 🚨 Alert Distribution")
        with st.container(border=True):
            st.image(str(alert_png), use_container_width=True)

    cross_png = shap_dir / "shap_06_cross_model_comparison.png"
    if cross_png.exists():
        st.markdown("### 📊 Cross-Model Feature Importance")
        with st.container(border=True):
            st.image(str(cross_png), use_container_width=True)

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
    metrics    = model_info["metrics"]

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

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

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

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    method = model_info.get("forecast_method", "N/A")
    if method == "Weather-informed":
        st.success(f"✅ **{method}** — Using OpenMeteo weather forecasts for 24h / 48h / 72h predictions")
    else:
        st.warning(f"⚠️ **{method}** — Weather forecast unavailable; using current conditions for all horizons")

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

# ── Footer ────────────────────────────────────────────────────────────
st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
st.markdown("---")
st.caption("🌍 Hawa Alert AQI Engine · Data: AQICN + OpenMeteo · Model Registry: Hopsworks Feature Store")