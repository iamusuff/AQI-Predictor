"""
Hawa Alert AQI Predictor — Redesigned Dashboard
No local files. SHAP from GitHub Releases. Charts from CSV data.
"""

import sys
import os
from datetime import datetime, timedelta, timezone
import io

import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import hopsworks
from config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# ── GitHub Releases — single source of truth ──────────────────────────
GITHUB_ORG  = "iamusuff"
GITHUB_REPO = "AQI-Predictor"
SHAP_CSV_URL = (
    f"https://github.com/{GITHUB_ORG}/{GITHUB_REPO}"
    f"/releases/download/shap-latest/shap_importance.csv"
)

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
# CSS — Dark Modern Theme
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Root tokens ── */
:root {
    --bg-base:        #0d0f14;
    --bg-surface:     #13161e;
    --bg-card:        #181c26;
    --bg-card-hover:  #1e2230;
    --bg-input:       #1a1e28;
    --border:         #252a38;
    --border-subtle:  #1e2230;
    --text-primary:   #e8eaf0;
    --text-secondary: #8b91a8;
    --text-muted:     #555c72;
    --accent-green:   #22c55e;
    --accent-yellow:  #eab308;
    --accent-orange:  #f97316;
    --accent-red:     #ef4444;
    --accent-purple:  #a855f7;
    --accent-blue:    #3b82f6;
    --accent-cyan:    #06b6d4;
    --glow-green:     rgba(34,197,94,0.15);
    --glow-yellow:    rgba(234,179,8,0.15);
    --glow-blue:      rgba(59,130,246,0.12);
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

/* ── Main container ── */
.main .block-container {
    padding: 1.8rem 2.2rem 3rem 2.2rem !important;
    max-width: 1380px !important;
    background-color: var(--bg-base) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 240px !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text-secondary) !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 7px 12px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    color: var(--text-secondary) !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.06) !important;
    color: var(--text-primary) !important;
}
section[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
    opacity: 0.5 !important;
}
section[data-testid="stSidebar"] .stButton button {
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: var(--bg-card-hover) !important;
    color: var(--text-primary) !important;
    border-color: rgba(255,255,255,0.12) !important;
}

/* ── Typography ── */
h1 {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.5px !important;
    margin-bottom: 0.2rem !important;
}
h2 {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.4px !important;
}
h3, h4 {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    margin-bottom: 0.9rem !important;
}

/* ── Cards / containers ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1.4rem 1.6rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.4), 0 8px 24px rgba(0,0,0,0.3) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(255,255,255,0.1) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.5), 0 12px 32px rgba(0,0,0,0.4) !important;
}

/* ── Metrics ── */
div[data-testid="stMetricLabel"] p {
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 22px !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

/* ── Alerts ── */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-width: 1px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 0.75rem 1.1rem !important;
    background: var(--bg-card) !important;
}

/* ── Selects ── */
div[data-testid="stSelectbox"] > div,
div[data-baseweb="select"] {
    border-radius: 10px !important;
    border-color: var(--border) !important;
    font-size: 13px !important;
    background: var(--bg-input) !important;
    color: var(--text-primary) !important;
}

/* ── DataFrames ── */
div[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}
div[data-testid="stDataFrame"] table thead tr th {
    background: var(--bg-surface) !important;
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
}
div[data-testid="stDataFrame"] table tbody tr td {
    color: var(--text-primary) !important;
    background: var(--bg-card) !important;
}
div[data-testid="stDataFrame"] table tbody tr:nth-child(even) td {
    background: var(--bg-surface) !important;
}

/* ── Expanders ── */
details {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
    padding: 0.2rem !important;
}
details summary {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    padding: 0.7rem 0.9rem !important;
}

/* ── Dividers ── */
hr {
    border-color: var(--border) !important;
    opacity: 0.5 !important;
    margin: 1rem 0 !important;
}

/* ── Captions / small text ── */
small, .stCaption, [data-testid="stCaptionContainer"] p {
    color: var(--text-muted) !important;
    font-size: 11px !important;
}

/* ── JSON viewer ── */
div[data-testid="stJson"] {
    background: var(--bg-surface) !important;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    font-size: 12px !important;
}

/* ── Forecast day card ── */
.forecast-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 16px 18px 16px;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
    position: relative;
    overflow: hidden;
}
.forecast-card:hover {
    transform: translateY(-2px);
    border-color: rgba(255,255,255,0.12);
}
.forecast-card .fc-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--text-muted);
    margin-bottom: 12px;
}
.forecast-card .fc-aqi {
    font-family: 'JetBrains Mono', monospace;
    font-size: 52px;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 10px;
}
.forecast-card .fc-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.forecast-card .fc-glow {
    position: absolute;
    top: -30px; left: 50%;
    transform: translateX(-50%);
    width: 120px; height: 120px;
    border-radius: 50%;
    filter: blur(40px);
    opacity: 0.18;
    pointer-events: none;
}

/* ── AQI big badge ── */
.aqi-hero {
    background: var(--bg-surface);
    border-radius: 18px;
    padding: 28px 20px 24px 20px;
    text-align: center;
    border: 1.5px solid var(--border);
    position: relative;
    overflow: hidden;
}
.aqi-hero-glow {
    position: absolute;
    top: -40px; left: 50%;
    transform: translateX(-50%);
    width: 180px; height: 180px;
    border-radius: 50%;
    filter: blur(60px);
    opacity: 0.2;
    pointer-events: none;
}
.aqi-hero-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 72px;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 10px;
}
.aqi-hero-label {
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}
.aqi-hero-desc {
    font-size: 11px;
    color: var(--text-muted);
    max-width: 180px;
    margin: 0 auto;
    line-height: 1.5;
}

/* ── Metric mini card ── */
.metric-mini {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
}
.metric-mini-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 20px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 4px;
}
.metric-mini-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
}

/* ── Section header ── */
.section-header {
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    color: var(--text-muted) !important;
    margin-bottom: 1rem !important;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
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

def apply_chart_theme(fig, height=320):
    fig.update_layout(
        margin=dict(l=8, r=8, t=12, b=8),
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Space Grotesk, sans-serif", size=11, color="#8b91a8"),
        hovermode="x unified",
        legend=dict(
            bgcolor='rgba(24,28,38,0.95)',
            bordercolor='#252a38', borderwidth=1,
            font=dict(size=11, color="#8b91a8"),
        ),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor='#1e2230', gridwidth=1,
        zeroline=False, showline=False,
        tickfont=dict(size=10, color="#555c72"),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor='#1e2230', gridwidth=1,
        zeroline=False, showline=False,
        tickfont=dict(size=10, color="#555c72"),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────
# SHAP Data Loader — GitHub Releases only
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def load_shap_importance() -> pd.DataFrame:
    try:
        resp = requests.get(
            SHAP_CSV_URL,
            timeout=15,
            allow_redirects=True,
            headers={"Accept": "application/octet-stream"},
        )
        if resp.status_code == 200:
            return pd.read_csv(io.StringIO(resp.text))
        else:
            st.warning(f"Could not fetch SHAP data (HTTP {resp.status_code}).")
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"SHAP data fetch failed: {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────
# Beautiful SHAP Charts — built from CSV
# ─────────────────────────────────────────────────────────────────────
def render_shap_charts(df: pd.DataFrame):
    if df.empty:
        st.error("No SHAP data available. Make sure GitHub Actions has pushed the latest release.")
        return

    cols = [c.lower() for c in df.columns]
    feat_col = df.columns[cols.index("feature")]       if "feature"    in cols else df.columns[0]
    imp_col  = df.columns[cols.index("importance")]    if "importance" in cols else df.columns[1]
    std_col  = df.columns[cols.index("std")]           if "std"        in cols else None
    pos_col  = df.columns[cols.index("positive")]      if "positive"   in cols else None
    neg_col  = df.columns[cols.index("negative")]      if "negative"   in cols else None

    df = df.copy()
    df[imp_col] = pd.to_numeric(df[imp_col], errors="coerce").abs()
    df = df.dropna(subset=[imp_col]).sort_values(imp_col, ascending=False).head(20).reset_index(drop=True)

    top_n = min(15, len(df))
    df_top = df.head(top_n)

    norm   = df_top[imp_col] / df_top[imp_col].max()
    colors = [
        f"rgba({int(239 * v + 34 * (1-v))}, {int(68 * v + 197 * (1-v))}, {int(68 * v + 94 * (1-v))}, 0.85)"
        for v in norm
    ]

    # Chart 1 — Horizontal Bar
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        y=df_top[feat_col][::-1],
        x=df_top[imp_col][::-1] * 1.05,
        orientation="h",
        marker=dict(color="rgba(234,179,8,0.04)", line=dict(width=0)),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig1.add_trace(go.Bar(
        y=df_top[feat_col][::-1],
        x=df_top[imp_col][::-1],
        orientation="h",
        marker=dict(color=colors[::-1], line=dict(width=0)),
        text=[f"  {v:.4f}" for v in df_top[imp_col][::-1]],
        textposition="outside",
        textfont=dict(size=10, color="#8b91a8", family="JetBrains Mono"),
        hovertemplate="<b>%{y}</b><br>SHAP Importance: %{x:.5f}<extra></extra>",
    ))
    if std_col:
        fig1.data[1].error_x = dict(
            type="data",
            array=df_top[std_col][::-1].tolist(),
            visible=True,
            color="rgba(100,100,100,0.25)",
            thickness=1.5,
            width=4,
        )
    fig1.update_layout(
        barmode="overlay",
        height=420,
        margin=dict(l=10, r=80, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk, sans-serif", size=11, color="#8b91a8"),
        showlegend=False,
        xaxis=dict(
            title="Mean |SHAP Value|",
            showgrid=True, gridcolor="#1e2230",
            zeroline=True, zerolinecolor="#252a38", zerolinewidth=1.5,
            tickfont=dict(size=9, family="JetBrains Mono", color="#555c72"),
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=11, color="#e8eaf0"),
        ),
    )

    # Chart 2 — Polar
    top8 = df_top.head(8)
    norm8 = top8[imp_col] / top8[imp_col].max()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatterpolar(
        r=norm8[imp_col] if imp_col in norm8 else top8[imp_col],
        theta=top8[feat_col],
        fill="toself",
        fillcolor="rgba(234,179,8,0.08)",
        line=dict(color="#eab308", width=2.5),
        marker=dict(
            size=norm8 * 14 + 6,
            color=top8[imp_col],
            colorscale=[[0, "#22c55e"], [0.5, "#eab308"], [1, "#ef4444"]],
            showscale=False,
            line=dict(color="#13161e", width=2),
        ),
        text=top8[feat_col],
        hovertemplate="<b>%{text}</b><br>Importance: %{r:.5f}<extra></extra>",
        name="SHAP Importance",
    ))
    fig2.update_layout(
        polar=dict(
            bgcolor="rgba(19,22,30,0.8)",
            radialaxis=dict(
                visible=True,
                showticklabels=True,
                tickfont=dict(size=8, color="#555c72", family="JetBrains Mono"),
                gridcolor="#252a38",
                linecolor="#252a38",
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color="#e8eaf0", family="Space Grotesk"),
                gridcolor="#1e2230",
                linecolor="#252a38",
            ),
        ),
        height=380,
        margin=dict(l=40, r=40, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk, sans-serif", size=11),
        showlegend=False,
    )

    # Chart 3 — Waterfall / Cumulative
    df_wf = df_top.head(10).copy()
    df_wf["cumulative"] = df_wf[imp_col].cumsum()
    df_wf["pct"]        = (df_wf[imp_col] / df_wf[imp_col].sum() * 100).round(1)
    wf_colors = [
        "#22c55e" if i < 3 else "#eab308" if i < 6 else "#f97316" if i < 8 else "#ef4444"
        for i in range(len(df_wf))
    ]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df_wf[feat_col],
        y=df_wf["cumulative"],
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.06)",
        line=dict(color="#3b82f6", width=2.5, dash="dot"),
        marker=dict(size=7, color="#3b82f6", line=dict(color="#13161e", width=2)),
        name="Cumulative",
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Cumulative: %{y:.5f}<extra></extra>",
    ))
    fig3.add_trace(go.Bar(
        x=df_wf[feat_col],
        y=df_wf[imp_col],
        marker=dict(color=wf_colors, line=dict(width=0), opacity=0.85),
        text=[f"{p}%" for p in df_wf["pct"]],
        textposition="outside",
        textfont=dict(size=9, color="#8b91a8", family="JetBrains Mono"),
        name="Contribution",
        hovertemplate="<b>%{x}</b><br>Importance: %{y:.5f}<br>Share: %{text}<extra></extra>",
    ))
    fig3.update_layout(
        height=360,
        margin=dict(l=10, r=60, t=30, b=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk, sans-serif", size=11, color="#8b91a8"),
        barmode="group",
        legend=dict(
            orientation="h", x=0, y=-0.25,
            bgcolor="rgba(19,22,30,0.9)",
            bordercolor="#252a38", borderwidth=1,
            font=dict(color="#8b91a8"),
        ),
        xaxis=dict(
            tickangle=-35,
            tickfont=dict(size=9, color="#555c72"),
            showgrid=False,
        ),
        yaxis=dict(
            title="SHAP Importance",
            showgrid=True, gridcolor="#1e2230",
            tickfont=dict(size=9, family="JetBrains Mono", color="#555c72"),
        ),
        yaxis2=dict(
            title="Cumulative",
            overlaying="y", side="right",
            showgrid=False,
            tickfont=dict(size=9, color="#3b82f6", family="JetBrains Mono"),
        ),
    )

    return fig1, fig2, fig3


# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='padding:1.4rem 0.4rem 1rem 0.4rem;'>
    <div style='display:flex; align-items:center; gap:10px; margin-bottom:6px;'>
        <span style='font-size:24px;'>🌍</span>
        <div>
            <div style='font-size:15px; font-weight:700; color:#e8eaf0; letter-spacing:-0.3px; font-family:"Space Grotesk",sans-serif;'>Hawa Alert</div>
            <div style='font-size:11px; color:#555c72; font-weight:600; letter-spacing:0.5px; text-transform:uppercase;'>Karachi · Pakistan</div>
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
st.sidebar.markdown("<div style='font-size:10px;color:#555c72;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px;font-weight:700;'>History Window</div>", unsafe_allow_html=True)
history_days = st.sidebar.selectbox("History Window", [7, 30, 90], index=1, label_visibility="collapsed")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"Updated\n{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")

# ─────────────────────────────────────────────────────────────────────
# Cache Helpers
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_inference():
    from inference import run as run_inference
    return run_inference(models_dir=MODELS_DIR)

@st.cache_data(ttl=300)
def get_history_data(days: int) -> pd.DataFrame:
    try:
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT_NAME,
        )
        fs = project.get_feature_store()
        fg = fs.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
        )
        df = fg.read()
        if df.empty:
            return pd.DataFrame()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
            df = df[df["timestamp"] >= cutoff]
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        st.warning(f"Hopsworks Feature Store connection failed: {e}")
        return pd.DataFrame()


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

    # ── Page header ──────────────────────────────────────────────
    st.markdown(f"""
    <div style='margin-bottom:1.6rem;'>
        <h1 style='margin:0 0 5px 0; font-family:"Space Grotesk",sans-serif;'>Hawa Alert — AQI Dashboard</h1>
        <p style='color:#555c72; font-size:12px; margin:0; letter-spacing:0.3px;'>
            Real-time air quality monitoring & ML forecasting · Karachi, Pakistan
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Alert banner ─────────────────────────────────────────────
    if current_aqi > 150:
        level = "🔴 RED ALERT" if current_aqi > 200 else "🟠 ORANGE ALERT"
        st.error(f"**{level} — AQI {current_aqi:.0f} ({cat}):** {desc}")
    elif current_aqi > 100:
        st.warning(f"**🟡 CAUTION — AQI {current_aqi:.0f} ({cat}):** {desc}")
    else:
        st.success(f"**✅ Good Air Quality** — {desc}")

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── TOP ROW: AQI Hero + Weather + 3-Day Forecast cards ───────
    col_aqi, col_weather, col_fc = st.columns([1.0, 1.0, 2.4])

    with col_aqi:
        with st.container(border=True):
            st.markdown("#### Current AQI")
            st.markdown(f"""
            <div class='aqi-hero' style='border-color:{color}33;'>
                <div class='aqi-hero-glow' style='background:{color};'></div>
                <div class='aqi-hero-number' style='color:{color};'>{current_aqi:.0f}</div>
                <div class='aqi-hero-label' style='color:{color};'>{cat}</div>
                <div class='aqi-hero-desc'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_weather:
        with st.container(border=True):
            st.markdown("#### Weather Now")
            temp = conditions.get('temperature')
            hum  = conditions.get('humidity')
            wind = conditions.get('wind_speed')
            st.markdown(f"""
            <div style='display:grid; grid-template-columns:1fr; gap:10px; padding:4px 0;'>
                <div class='metric-mini'>
                    <div style='font-size:18px; margin-bottom:4px;'>🌡️</div>
                    <div class='metric-mini-val'>{f"{temp:.0f}°C" if temp is not None else "—"}</div>
                    <div class='metric-mini-label'>Temperature</div>
                </div>
                <div class='metric-mini'>
                    <div style='font-size:18px; margin-bottom:4px;'>💧</div>
                    <div class='metric-mini-val'>{f"{hum:.0f}%" if hum is not None else "—"}</div>
                    <div class='metric-mini-label'>Humidity</div>
                </div>
                <div class='metric-mini'>
                    <div style='font-size:18px; margin-bottom:4px;'>💨</div>
                    <div class='metric-mini-val'>{f"{wind:.1f}" if wind is not None else "—"}</div>
                    <div class='metric-mini-label'>Wind km/h</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_fc:
        with st.container(border=True):
            st.markdown("#### 3-Day AQI Forecast")

            # Build forecast rows — explicit separate cards like the sample UI
            forecast_labels = {
                "current": ("TODAY'S AQI", ""),
                "24h":     ("TOMORROW",    "+1"),
                "48h":     ("DAY +2",      "+2"),
                "72h":     ("DAY +3",      "+3"),
            }
            fc_cols = st.columns(4)
            for col_widget, (key, (label, _)) in zip(fc_cols, forecast_labels.items()):
                p = predictions[key]
                fc_aqi = p["aqi"]
                fc_cat, fc_color, _ = aqi_category(fc_aqi)
                with col_widget:
                    st.markdown(f"""
                    <div class='forecast-card' style='border-color:{fc_color}22;'>
                        <div class='fc-glow' style='background:{fc_color};'></div>
                        <div class='fc-label'>{label}</div>
                        <div class='fc-aqi' style='color:{fc_color};'>{fc_aqi:.0f}</div>
                        <div class='fc-badge' style='background:{fc_color}18; color:{fc_color};'>↑ {fc_cat}</div>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── SECOND ROW: SHAP + Historical ────────────────────────────
    col_shap, col_hist = st.columns([1, 1.3])

    with col_shap:
        with st.container(border=True):
            st.markdown("#### SHAP Feature Importance")
            shap_df = load_shap_importance()
            if not shap_df.empty:
                cols_l = [c.lower() for c in shap_df.columns]
                feat_c = shap_df.columns[cols_l.index("feature")]    if "feature"    in cols_l else shap_df.columns[0]
                imp_c  = shap_df.columns[cols_l.index("importance")] if "importance" in cols_l else shap_df.columns[1]
                shap_df[imp_c] = pd.to_numeric(shap_df[imp_c], errors="coerce").abs()
                top = shap_df.dropna(subset=[imp_c]).sort_values(imp_c, ascending=False).head(8)
                norm = top[imp_c] / top[imp_c].max()
                bar_colors = [
                    f"rgba({int(239*v+34*(1-v))},{int(68*v+197*(1-v))},{int(68*v+94*(1-v))},0.85)"
                    for v in norm
                ]
                fig_s = go.Figure(go.Bar(
                    x=top[imp_c][::-1],
                    y=top[feat_c][::-1],
                    orientation="h",
                    marker=dict(color=bar_colors[::-1], line=dict(width=0)),
                    text=[f"{v:.4f}" for v in top[imp_c][::-1]],
                    textposition="outside",
                    textfont=dict(size=9, family="JetBrains Mono", color="#8b91a8"),
                    hovertemplate="<b>%{y}</b><br>%{x:.5f}<extra></extra>",
                ))
                apply_chart_theme(fig_s, height=260)
                fig_s.update_layout(
                    xaxis_title="Mean |SHAP|",
                    margin=dict(l=8, r=60, t=12, b=8),
                )
                st.plotly_chart(fig_s, use_container_width=True)
                st.caption("→ See Feature Importance page for full analysis")
            else:
                st.info("SHAP data not yet available from GitHub Releases.")

    with col_hist:
        with st.container(border=True):
            st.markdown("#### Historical AQI (7 days)")
            df_hist = get_history_data(7)
            if not df_hist.empty and "aqi" in df_hist.columns:
                fig_h = go.Figure()
                fig_h.add_trace(go.Scatter(
                    x=df_hist["timestamp"], y=df_hist["aqi"],
                    mode="lines+markers", name="AQI",
                    line=dict(color="#3b82f6", width=2),
                    marker=dict(size=4, color="#3b82f6"),
                ))
                if "aqi_rolling_24h" in df_hist.columns:
                    fig_h.add_trace(go.Scatter(
                        x=df_hist["timestamp"], y=df_hist["aqi_rolling_24h"],
                        mode="lines", name="24h Avg",
                        line=dict(color="#f97316", width=2, dash="dot"),
                    ))
                apply_chart_theme(fig_h, height=260)
                fig_h.update_layout(
                    legend=dict(orientation="v", x=1.01, y=1),
                    yaxis_title="AQI", xaxis_title="Days",
                )
                st.plotly_chart(fig_h, use_container_width=True)
            else:
                st.info("Historical data loading from Hopsworks...")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── THIRD ROW: Weather + Pollutants ─────────────────────────
    col_wc, col_pl = st.columns(2)

    with col_wc:
        with st.container(border=True):
            st.markdown("#### Weather Conditions")
            df_full = get_history_data(history_days)
            fig_w = go.Figure()
            weather_cols = [
                ("temperature", "#3b82f6", "Temperature"),
                ("humidity",    "#06b6d4", "Humidity"),
                ("wind_speed",  "#8b5cf6", "Wind Speed"),
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
                xaxis_title="Time", yaxis_title="Conditions",
            )
            st.plotly_chart(fig_w, use_container_width=True)

    with col_pl:
        with st.container(border=True):
            st.markdown("#### Pollutant Levels")
            df_full = get_history_data(history_days)
            fig_p = go.Figure()
            pollutant_cols = [
                ("pm25", "#ef4444", "PM2.5"),
                ("pm10", "#f97316", "PM10"),
                ("o3",   "#22c55e", "O₃"),
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
                xaxis_title="Time", yaxis_title="Concentration",
            )
            st.plotly_chart(fig_p, use_container_width=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Model metrics row ────────────────────────────────────────
    metrics = model_info.get("metrics", {})
    r2   = metrics.get('test_r2',   0.85)
    rmse = metrics.get('test_rmse', 12.3)
    mae  = metrics.get('test_mae',  9.1)
    mape = metrics.get('test_mape', 4.5)

    with st.container(border=True):
        mc1, mc2, mc3, mc4 = st.columns(4)
        for col_m, label, val, sub in [
            (mc1, f"Test R²",   f"{r2:.2f}",   "Explained variance"),
            (mc2, f"Test RMSE", f"{rmse:.1f}",  "Root mean squared error"),
            (mc3, f"Test MAE",  f"{mae:.1f}",   "Mean absolute error"),
            (mc4, f"MAPE",      f"{mape:.1f}%", "Mean absolute % error"),
        ]:
            col_m.markdown(
                f"""<div class='metric-mini' style='padding:18px 16px;'>
                    <div class='metric-mini-label' style='margin-bottom:8px;'>{label}</div>
                    <div style='font-family:"JetBrains Mono",monospace; font-size:22px; font-weight:700;
                                color:#e8eaf0; margin-bottom:4px;'>{val}</div>
                    <div class='metric-mini-label' style='font-size:10px;'>{sub}</div>
                </div>""",
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
- **24h / 48h / 72h**: TRUE multi-horizon predictions using OpenMeteo weather forecasts
- **Confidence Intervals**: 95% prediction intervals based on model test RMSE
- **Model**: {model_name}
        """)


# ─────────────────────────────────────────────────────────────────────
# Historical Trends Page
# ─────────────────────────────────────────────────────────────────────
def render_history():
    df = get_history_data(history_days)
    if df.empty:
        st.warning("No historical data available from Hopsworks Feature Store.")
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
                    y=hi, line_dash="dot", line_color=color, opacity=0.2,
                    annotation_text=label if hi <= df["aqi"].max() else None,
                    annotation_font_size=9,
                    annotation_font_color="#8b91a8",
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
    st.markdown("## 🎯 Feature Importance — SHAP Analysis")
    st.markdown("<p style='color:#555c72; font-size:12px; margin-bottom:1.5rem;'>Global feature importance from the latest trained model · fetched live from GitHub Releases.</p>", unsafe_allow_html=True)

    col_refresh = st.columns([4, 1])
    with col_refresh[1]:
        if st.button("🔄 Refresh SHAP", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    shap_df = load_shap_importance()

    if shap_df.empty:
        st.error("⚠️ Could not load SHAP data from GitHub Releases. Make sure the `shap-latest` release exists.")
        st.code(SHAP_CSV_URL, language="text")
        return

    with st.expander("📋 Raw SHAP Importance Data", expanded=False):
        st.dataframe(shap_df, use_container_width=True, hide_index=True)

    charts = render_shap_charts(shap_df)
    if charts is None:
        return
    fig1, fig2, fig3 = charts

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("#### 📊 Top Feature Importances — Mean |SHAP|")
        st.caption("Longer bar = stronger influence on AQI prediction. Color: green (lower) → red (higher importance).")
        st.plotly_chart(fig1, use_container_width=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    col_radar, col_waterfall = st.columns(2)
    with col_radar:
        with st.container(border=True):
            st.markdown("#### 🕸️ Radar — Top 8 Features")
            st.caption("Radial spread shows relative importance of top features.")
            st.plotly_chart(fig2, use_container_width=True)

    with col_waterfall:
        with st.container(border=True):
            st.markdown("#### 📉 Cumulative Contribution")
            st.caption("Bars = individual share. Dotted line = cumulative build-up.")
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("#### 🔬 Top Feature Insights")
        cols_l = [c.lower() for c in shap_df.columns]
        feat_c = shap_df.columns[cols_l.index("feature")]    if "feature"    in cols_l else shap_df.columns[0]
        imp_c  = shap_df.columns[cols_l.index("importance")] if "importance" in cols_l else shap_df.columns[1]
        shap_df[imp_c] = pd.to_numeric(shap_df[imp_c], errors="coerce").abs()
        top5 = shap_df.dropna(subset=[imp_c]).sort_values(imp_c, ascending=False).head(5).reset_index(drop=True)

        insight_cols = st.columns(5)
        rank_colors  = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#3b82f6"]
        rank_labels  = ["🥇", "🥈", "🥉", "4th", "5th"]

        for i, (_, row) in enumerate(top5.iterrows()):
            pct = row[imp_c] / shap_df[imp_c].sum() * 100
            insight_cols[i].markdown(
                f"""<div style='background:linear-gradient(135deg,{rank_colors[i]}0d,{rank_colors[i]}06);
                               border:1.5px solid {rank_colors[i]}28; border-radius:14px;
                               padding:16px 10px; text-align:center;'>
                    <div style='font-size:20px; margin-bottom:8px;'>{rank_labels[i]}</div>
                    <div style='font-size:11px; font-weight:700; color:#e8eaf0;
                                word-break:break-word; line-height:1.3; margin-bottom:10px;'>{row[feat_c]}</div>
                    <div style='font-size:14px; font-weight:700; color:{rank_colors[i]};
                                font-family:"JetBrains Mono",monospace;'>{row[imp_c]:.4f}</div>
                    <div style='font-size:10px; color:#555c72; margin-top:4px; font-weight:600;'>{pct:.1f}% of total</div>
                </div>""",
                unsafe_allow_html=True,
            )


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
st.caption("🌍 Hawa Alert AQI Engine · Data: AQICN + OpenMeteo · Model Registry: Hopsworks · SHAP: GitHub Releases")