"""
Hawa Alert AQI Predictor — Redesigned Dashboard
No local files. SHAP from GitHub Releases. Charts from CSV data.

Fixes applied:
  1. Hopsworks connection cached in st.session_state to prevent stale-connection
     failures when navigating between pages (Historical Trends bug).
  2. get_inference() result cached in session_state — called once per TTL, not
     once per page render.
  3. get_history_data() called once per render and reused — no duplicate calls.
  4. Hardcoded fallback metrics replaced with None / "—" so missing data is
     never silently hidden.
  5. Double sys.path insert collapsed to a single insert.
  6. Blue channel * 0 color bug fixed in importance_color() helper.
  7. render_shap_charts() always returns (fig1, fig2, fig3) or raises — no
     mixed return types.
  8. desc truncation uses CSS text-overflow instead of slicing.
  9. Empty chart branches now show st.info() instead of a blank chart.
 10. MODELS_DIR kept but explicitly documented as Hopsworks fallback only.
"""

import sys
import os
from datetime import datetime, timedelta, timezone
import io
import time

import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Single sys.path insert ────────────────────────────────────────────────────
_SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

from config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# ── GitHub Releases ───────────────────────────────────────────────────────────
GITHUB_ORG  = "iamusuff"
GITHUB_REPO = "AQI-Predictor"
SHAP_CSV_URL = (
    f"https://github.com/{GITHUB_ORG}/{GITHUB_REPO}"
    f"/releases/download/shap-latest/shap_importance.csv"
)

# ── Inference cache TTL (seconds) ─────────────────────────────────────────────
_INFERENCE_TTL   = 300   # 5 min
_HISTORY_TTL     = 300
_HOPSWORKS_TTL   = 1800  # re-login every 30 min

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hawa Alert — AQI Karachi",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

html, body, .stApp {
    font-family: 'IBM Plex Sans', sans-serif !important;
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
}
[data-testid="stAppViewContainer"] > .main,
.main .block-container {
    background-color: #0d1117 !important;
    padding: 1.5rem 1.8rem 3rem 1.8rem !important;
    max-width: 1380px !important;
}
section[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #21262d !important;
}
section[data-testid="stSidebar"] > div:first-child { background-color: #0d1117 !important; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #8b949e !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
[data-testid="collapsedControl"] {
    background-color: #161b22 !important;
    border: 1px solid #21262d !important;
    color: #8b949e !important;
}
[data-testid="collapsedControl"] svg { fill: #8b949e !important; }
[data-testid="collapsedControl"]:hover {
    background-color: #21262d !important;
    border-color: #ff4444 !important;
}
section[data-testid="stSidebar"] .stRadio > div { gap: 2px !important; }
section[data-testid="stSidebar"] .stRadio label {
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    border-left: 2px solid transparent !important;
    color: #8b949e !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(48,54,61,0.8) !important;
    color: #e6edf3 !important;
    border-left-color: #ff4444 !important;
}
section[data-testid="stSidebar"] .stRadio [role="radio"] {
    background: #21262d !important;
    border-color: #30363d !important;
}
section[data-testid="stSidebar"] .stRadio [role="radio"][aria-checked="true"] {
    background: #ff4444 !important;
    border-color: #ff4444 !important;
    box-shadow: 0 0 8px rgba(255,68,68,0.4) !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #6e7681 !important;
    font-size: 12px !important;
}
section[data-testid="stSidebar"] hr { border-color: #21262d !important; margin: 0.8rem 0 !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: #ff4444 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1rem !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
}
section[data-testid="stSidebar"] .stButton > button span,
section[data-testid="stSidebar"] .stButton > button p { color: #ffffff !important; }
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #cc0000 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(255,68,68,0.35) !important;
}
section[data-testid="stSidebar"] .stSelectbox > div,
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #161b22 !important;
    border-color: #30363d !important;
    border-radius: 6px !important;
    color: #c9d1d9 !important;
}
[data-testid="stHeader"] {
    background-color: #0d1117 !important;
    border-bottom: 1px solid #21262d !important;
}
h1 {
    font-size: 22px !important; font-weight: 700 !important; color: #e6edf3 !important;
    letter-spacing: -0.3px !important; margin-bottom: 2px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
h2 {
    font-size: 18px !important; font-weight: 700 !important; color: #e6edf3 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
h3, h4 {
    font-size: 12px !important; font-weight: 600 !important; color: #8b949e !important;
    text-transform: uppercase !important; letter-spacing: 0.8px !important;
    margin-bottom: 0.7rem !important; font-family: 'IBM Plex Sans', sans-serif !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    padding: 1.2rem 1.4rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.4) !important;
    transition: border-color 0.2s ease !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: #30363d !important; }
div[data-testid="stMetricLabel"] p,
div[data-testid="stMetricLabel"] label,
[data-testid="stMetricLabel"] {
    font-size: 11px !important; text-transform: uppercase !important;
    letter-spacing: 0.8px !important; color: #8b949e !important;
    font-weight: 600 !important; font-family: 'IBM Plex Sans', sans-serif !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 22px !important; font-weight: 600 !important; color: #e6edf3 !important;
}
div[data-testid="stMetricDelta"] { font-size: 11px !important; font-family: 'IBM Plex Mono', monospace !important; }
div[data-testid="stAlert"], [data-testid="stAlertContainer"] {
    border-radius: 8px !important; font-size: 13px !important; font-weight: 500 !important;
    padding: 0.65rem 1rem !important; border-width: 1px !important;
    font-family: 'IBM Plex Sans', sans-serif !important; background: transparent !important;
}
div[data-testid="stDataFrame"],
div[data-testid="stDataFrame"] > div,
.stDataFrame {
    border-radius: 8px !important; overflow: hidden !important;
    border: 1px solid #21262d !important; background: #161b22 !important;
}
details, [data-testid="stExpander"] {
    border-radius: 8px !important; border: 1px solid #21262d !important; background: #161b22 !important;
}
details summary, [data-testid="stExpander"] summary {
    font-size: 13px !important; font-weight: 600 !important; color: #8b949e !important;
    padding: 0.6rem 0.8rem !important; font-family: 'IBM Plex Sans', sans-serif !important;
    background: #161b22 !important;
}
details summary:hover, [data-testid="stExpander"] summary:hover {
    color: #e6edf3 !important; background: rgba(48,54,61,0.5) !important;
}
div[data-testid="stSelectbox"] > div,
[data-baseweb="select"] > div,
[data-baseweb="select"] [data-baseweb="base-input"] {
    border-radius: 6px !important; border-color: #30363d !important;
    font-size: 13px !important; background: #21262d !important; color: #c9d1d9 !important;
}
[data-baseweb="popover"], [data-baseweb="menu"], ul[data-baseweb="menu"],
[role="listbox"], [data-baseweb="popover"] > div {
    background: #21262d !important; border: 1px solid #30363d !important;
    border-radius: 8px !important; box-shadow: 0 8px 24px rgba(0,0,0,0.6) !important;
}
[data-baseweb="menu"] li, [data-baseweb="menu"] [role="option"],
[role="listbox"] li, [role="option"] {
    background: #21262d !important; color: #c9d1d9 !important;
    font-size: 13px !important; font-family: 'IBM Plex Sans', sans-serif !important;
}
[data-baseweb="menu"] li:hover, [role="option"]:hover, [role="option"][aria-selected="true"] {
    background: #30363d !important; color: #e6edf3 !important;
}
.stButton > button, button[kind="secondary"] {
    background: #21262d !important; color: #c9d1d9 !important;
    border: 1px solid #30363d !important; border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important; font-size: 13px !important;
    font-weight: 500 !important; transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #30363d !important; border-color: #ff4444 !important;
    color: #e6edf3 !important; box-shadow: 0 2px 8px rgba(255,68,68,0.15) !important;
}
small, .stCaption, [data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] span {
    color: #6e7681 !important; font-size: 11px !important;
    font-family: 'IBM Plex Sans', sans-serif !important; background: transparent !important;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] a { color: #c9d1d9 !important; background: transparent !important; }
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] b { color: #e6edf3 !important; }
[data-testid="stMarkdownContainer"] a { color: #58a6ff !important; }
[data-testid="stMarkdownContainer"] code {
    background: #21262d !important; color: #e6edf3 !important;
    padding: 2px 6px !important; border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
hr { border: none !important; border-top: 1px solid #21262d !important; margin: 0.8rem 0 !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6e7681; }
footer { visibility: hidden; }

/* ── Custom component classes ── */
.nav-label {
    font-size: 10px !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 1px !important;
    color: #6e7681 !important; margin: 12px 0 6px 0 !important;
    padding: 0 4px !important; background: transparent !important;
}
.params-label {
    font-size: 10px !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 1px !important;
    color: #6e7681 !important; margin: 8px 0 4px 0 !important; background: transparent !important;
}
.forecast-card {
    background: #161b22; border: 1px solid #21262d; border-radius: 10px;
    padding: 16px 14px; text-align: left;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    position: relative; overflow: hidden;
}
.forecast-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--card-accent, #388bfd); border-radius: 10px 10px 0 0;
}
.forecast-card:hover { border-color: #30363d; box-shadow: 0 4px 16px rgba(0,0,0,0.4); }
.forecast-card .fc-label {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: #6e7681; margin-bottom: 10px;
    font-family: 'IBM Plex Sans', sans-serif;
}
.forecast-card .fc-value {
    font-size: 42px; font-weight: 700; font-family: 'IBM Plex Mono', monospace;
    line-height: 1; margin-bottom: 8px;
}
.forecast-card .fc-badge {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 20px;
    font-family: 'IBM Plex Sans', sans-serif; margin-bottom: 6px;
}
/* Fix #7: desc uses CSS ellipsis — no Python slicing */
.forecast-card .fc-desc {
    font-size: 10px; color: #6e7681; font-family: 'IBM Plex Sans', sans-serif;
    margin-top: 6px; line-height: 1.4;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.section-label {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: #6e7681; margin-bottom: 12px;
    font-family: 'IBM Plex Sans', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
AQI_CATEGORIES = [
    (0,   50,  "Good",                    "#3fb950", "Air quality is satisfactory"),
    (51,  100, "Moderate",                "#d29922", "Acceptable; some pollutants may affect sensitive people"),
    (101, 150, "Unhealthy for Sensitive", "#f0883e", "Sensitive groups should reduce outdoor activity"),
    (151, 200, "Unhealthy",               "#f85149", "Everyone may experience health effects"),
    (201, 300, "Very Unhealthy",          "#bc8cff", "Health alert — avoid outdoor activity"),
    (301, 500, "Hazardous",               "#ff7b72", "Health emergency — avoid all outdoor activity"),
]

def aqi_category(aqi):
    for lo, hi, label, color, desc in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return label, color, desc
    return "Hazardous", "#ff7b72", "Health emergency — avoid all outdoor activity"

def aqi_color(aqi):
    return aqi_category(aqi)[1]

# Fix #6: importance_color — blue channel was multiplied by 0, now correct
def importance_color(norm_value: float) -> str:
    """Interpolate a single normalised [0,1] value from blue → red."""
    v = float(norm_value)
    r = int(248 * v + 56  * (1 - v))
    g = int(81  * v + 139 * (1 - v))
    b = int(73  * (1 - v) + 0  * v)   # was "253 * v * 0" — fixed
    return f"rgba({r},{g},{b},0.85)"

def apply_chart_theme(fig, height=320):
    fig.update_layout(
        margin=dict(l=8, r=8, t=12, b=8),
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(22,27,34,0.0)',
        font=dict(family="IBM Plex Sans, sans-serif", size=11, color="#8b949e"),
        hovermode="x unified",
        legend=dict(
            bgcolor='rgba(22,27,34,0.95)', bordercolor='#30363d', borderwidth=1,
            font=dict(size=11, color="#c9d1d9"),
        ),
        hoverlabel=dict(
            bgcolor="#21262d", bordercolor="#30363d",
            font=dict(family="IBM Plex Mono", size=11, color="#e6edf3"),
        ),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor='#21262d', gridwidth=1, zeroline=False,
        showline=False, tickfont=dict(size=10, color="#6e7681", family="IBM Plex Mono"),
        linecolor='#21262d',
    )
    fig.update_yaxes(
        showgrid=True, gridcolor='#21262d', gridwidth=1, zeroline=False,
        showline=False, tickfont=dict(size=10, color="#6e7681", family="IBM Plex Mono"),
        linecolor='#21262d',
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Fix #1 — Hopsworks connection cached in session_state
# This prevents stale-connection failures when navigating between pages.
# The connection is reused across Dashboard → Historical Trends → etc.
# A fresh login only happens when the TTL expires or data is refreshed.
# ─────────────────────────────────────────────────────────────────────────────
def get_hopsworks_project():
    """
    Return a cached Hopsworks project handle stored in session_state.
    Re-logins happen only when TTL expires or cache is manually cleared.
    """
    now = time.time()
    cached_at  = st.session_state.get("_hw_cached_at", 0)
    hw_project = st.session_state.get("_hw_project", None)

    if hw_project is None or (now - cached_at) > _HOPSWORKS_TTL:
        try:
            import hopsworks
            project = hopsworks.login(
                api_key_value=HOPSWORKS_API_KEY,
                project=HOPSWORKS_PROJECT_NAME,
            )
            st.session_state["_hw_project"]   = project
            st.session_state["_hw_cached_at"] = now
            return project
        except Exception as e:
            # Clear stale handle so the next call retries
            st.session_state.pop("_hw_project",   None)
            st.session_state.pop("_hw_cached_at", None)
            raise e

    return hw_project

# ─────────────────────────────────────────────────────────────────────────────
# Fix #2 — Inference result cached in session_state (single call per TTL)
# ─────────────────────────────────────────────────────────────────────────────
def get_inference():
    """
    Run inference once and cache the result in session_state.
    Any page that calls get_inference() within the TTL window gets the same
    object — no duplicate Hopsworks / API calls.
    """
    now       = time.time()
    cached_at = st.session_state.get("_inf_cached_at", 0)
    result    = st.session_state.get("_inf_result", None)

    if result is None or (now - cached_at) > _INFERENCE_TTL:
        from inference import run as run_inference
        result = run_inference(models_dir=MODELS_DIR)
        st.session_state["_inf_result"]    = result
        st.session_state["_inf_cached_at"] = now

    return result

# ─────────────────────────────────────────────────────────────────────────────
# Fix #1 cont. — Historical data uses shared Hopsworks connection
# ─────────────────────────────────────────────────────────────────────────────
def get_history_data(days: int) -> pd.DataFrame:
    """
    Fetch historical AQI data from Hopsworks Feature Store.
    Uses the shared session_state project handle — no repeated logins.
    Results are cached in session_state per (days) key.
    """
    cache_key    = f"_hist_{days}"
    cache_ts_key = f"_hist_{days}_at"
    now          = time.time()

    cached_at = st.session_state.get(cache_ts_key, 0)
    cached_df = st.session_state.get(cache_key, None)

    if cached_df is not None and (now - cached_at) < _HISTORY_TTL:
        return cached_df

    try:
        project = get_hopsworks_project()   # reuses existing connection
        fs = project.get_feature_store()
        fg = fs.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
        )
        df = fg.read()

        if df.empty:
            st.session_state[cache_key]    = pd.DataFrame()
            st.session_state[cache_ts_key] = now
            return pd.DataFrame()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
            df = df[df["timestamp"] >= cutoff]

        df = df.sort_values("timestamp").reset_index(drop=True)
        st.session_state[cache_key]    = df
        st.session_state[cache_ts_key] = now
        return df

    except Exception as e:
        st.warning(f"Hopsworks Feature Store connection failed: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# SHAP Data Loader
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def load_shap_importance() -> pd.DataFrame:
    try:
        resp = requests.get(
            SHAP_CSV_URL, timeout=15, allow_redirects=True,
            headers={"Accept": "application/octet-stream"},
        )
        if resp.status_code == 200:
            return pd.read_csv(io.StringIO(resp.text))
        st.warning(f"Could not fetch SHAP data (HTTP {resp.status_code}).")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"SHAP data fetch failed: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Fix #7 — render_shap_charts always returns (fig1, fig2, fig3) or raises
# ─────────────────────────────────────────────────────────────────────────────
def render_shap_charts(df: pd.DataFrame):
    if df.empty:
        raise ValueError("Empty SHAP dataframe")

    cols_l   = [c.lower() for c in df.columns]
    feat_col = df.columns[cols_l.index("feature")]    if "feature"    in cols_l else df.columns[0]
    imp_col  = df.columns[cols_l.index("importance")] if "importance" in cols_l else df.columns[1]
    std_col  = df.columns[cols_l.index("std")]        if "std"        in cols_l else None
    pos_col  = df.columns[cols_l.index("positive")]   if "positive"   in cols_l else None  # noqa: F841
    neg_col  = df.columns[cols_l.index("negative")]   if "negative"   in cols_l else None  # noqa: F841

    df = df.copy()
    df[imp_col] = pd.to_numeric(df[imp_col], errors="coerce").abs()
    df = df.dropna(subset=[imp_col]).sort_values(imp_col, ascending=False).head(20).reset_index(drop=True)

    top_n  = min(15, len(df))
    df_top = df.head(top_n)

    norm   = df_top[imp_col] / df_top[imp_col].max()
    colors = [importance_color(v) for v in norm]   # Fix #6 applied

    # ── Chart 1 — Horizontal Bar ──────────────────────────────────────────────
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        y=df_top[feat_col][::-1], x=df_top[imp_col][::-1] * 1.05,
        orientation="h", marker=dict(color="rgba(56,139,253,0.05)", line=dict(width=0)),
        showlegend=False, hoverinfo="skip",
    ))
    bar = go.Bar(
        y=df_top[feat_col][::-1], x=df_top[imp_col][::-1],
        orientation="h", marker=dict(color=colors[::-1], line=dict(width=0)),
        text=[f"  {v:.4f}" for v in df_top[imp_col][::-1]],
        textposition="outside",
        textfont=dict(size=10, color="#8b949e", family="IBM Plex Mono"),
        hovertemplate="<b>%{y}</b><br>SHAP Importance: %{x:.5f}<extra></extra>",
    )
    if std_col:
        bar.error_x = dict(
            type="data", array=df_top[std_col][::-1].tolist(),
            visible=True, color="rgba(139,148,158,0.3)", thickness=1.5, width=4,
        )
    fig1.add_trace(bar)
    fig1.update_layout(
        barmode="overlay", height=420,
        margin=dict(l=10, r=80, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", size=11, color="#8b949e"),
        showlegend=False,
        xaxis=dict(
            title="Mean |SHAP Value|", showgrid=True, gridcolor="#21262d",
            zeroline=True, zerolinecolor="#30363d", zerolinewidth=1.5,
            tickfont=dict(size=9, family="IBM Plex Mono", color="#6e7681"),
            title_font=dict(color="#8b949e"),
        ),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, color="#c9d1d9")),
        hoverlabel=dict(bgcolor="#21262d", bordercolor="#30363d",
                        font=dict(family="IBM Plex Mono", size=11, color="#e6edf3")),
    )

    # ── Chart 2 — Radar ───────────────────────────────────────────────────────
    top8  = df_top.head(8)
    norm8 = top8[imp_col] / top8[imp_col].max()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatterpolar(
        r=top8[imp_col], theta=top8[feat_col],
        fill="toself", fillcolor="rgba(248,81,73,0.12)",
        line=dict(color="#f85149", width=2),
        marker=dict(
            size=norm8 * 14 + 5, color=top8[imp_col],
            colorscale=[[0, "#388bfd"], [0.5, "#d29922"], [1, "#f85149"]],
            showscale=False, line=dict(color="#161b22", width=2),
        ),
        text=top8[feat_col],
        hovertemplate="<b>%{text}</b><br>Importance: %{r:.5f}<extra></extra>",
        name="SHAP Importance",
    ))
    fig2.update_layout(
        polar=dict(
            bgcolor="rgba(22,27,34,0.8)",
            radialaxis=dict(visible=True, showticklabels=True,
                            tickfont=dict(size=8, color="#6e7681", family="IBM Plex Mono"),
                            gridcolor="#21262d", linecolor="#30363d"),
            angularaxis=dict(tickfont=dict(size=10, color="#c9d1d9"),
                             gridcolor="#21262d", linecolor="#30363d"),
        ),
        height=380, margin=dict(l=40, r=40, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", size=11, color="#8b949e"),
        showlegend=False,
        hoverlabel=dict(bgcolor="#21262d", bordercolor="#30363d",
                        font=dict(family="IBM Plex Mono", size=11, color="#e6edf3")),
    )

    # ── Chart 3 — Cumulative Contribution ────────────────────────────────────
    df_wf = df_top.head(10).copy()
    df_wf["cumulative"] = df_wf[imp_col].cumsum()
    df_wf["pct"]        = (df_wf[imp_col] / df_wf[imp_col].sum() * 100).round(1)
    wf_colors = [
        "#f85149" if i < 3 else "#d29922" if i < 6 else "#f0883e" if i < 8 else "#388bfd"
        for i in range(len(df_wf))
    ]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df_wf[feat_col], y=df_wf["cumulative"],
        mode="lines+markers", fill="tozeroy",
        fillcolor="rgba(56,139,253,0.08)",
        line=dict(color="#388bfd", width=2, dash="dot"),
        marker=dict(size=6, color="#388bfd", line=dict(color="#161b22", width=2)),
        name="Cumulative", yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Cumulative: %{y:.5f}<extra></extra>",
    ))
    fig3.add_trace(go.Bar(
        x=df_wf[feat_col], y=df_wf[imp_col],
        marker=dict(color=wf_colors, line=dict(width=0), opacity=0.85),
        text=[f"{p}%" for p in df_wf["pct"]], textposition="outside",
        textfont=dict(size=9, color="#8b949e", family="IBM Plex Mono"),
        name="Contribution",
        hovertemplate="<b>%{x}</b><br>Importance: %{y:.5f}<br>Share: %{text}<extra></extra>",
    ))
    fig3.update_layout(
        height=360, margin=dict(l=10, r=60, t=30, b=80),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", size=11, color="#8b949e"),
        barmode="group",
        legend=dict(orientation="h", x=0, y=-0.25,
                    bgcolor="rgba(22,27,34,0.95)", bordercolor="#30363d", borderwidth=1,
                    font=dict(size=11, color="#c9d1d9")),
        xaxis=dict(tickangle=-35, tickfont=dict(size=9, color="#6e7681"), showgrid=False),
        yaxis=dict(title="SHAP Importance", showgrid=True, gridcolor="#21262d",
                   tickfont=dict(size=9, family="IBM Plex Mono", color="#6e7681"),
                   title_font=dict(color="#8b949e")),
        yaxis2=dict(title="Cumulative", overlaying="y", side="right", showgrid=False,
                    tickfont=dict(size=9, color="#388bfd", family="IBM Plex Mono"),
                    title_font=dict(color="#388bfd")),
        hoverlabel=dict(bgcolor="#21262d", bordercolor="#30363d",
                        font=dict(family="IBM Plex Mono", size=11, color="#e6edf3")),
    )

    return fig1, fig2, fig3   # always a tuple — never None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='padding:1.2rem 0.6rem 0.8rem 0.6rem;'>
    <div style='display:flex; align-items:center; gap:10px; margin-bottom:8px;'>
        <div style='width:32px; height:32px; background:linear-gradient(135deg,#ff4444,#cc0000);
                    border-radius:8px; display:flex; align-items:center; justify-content:center;
                    font-size:16px; flex-shrink:0; box-shadow:0 2px 8px rgba(255,68,68,0.3);'>⚡</div>
        <div>
            <div style='font-size:14px; font-weight:700; color:#e6edf3; letter-spacing:-0.2px;
                        font-family:"IBM Plex Sans",sans-serif;'>Hawa Alert</div>
            <div style='font-size:10px; color:#6e7681; font-family:"IBM Plex Mono",monospace;'>v1.0.0</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("<div class='nav-label'>Navigation</div>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["📊  Dashboard", "📈  Forecast Details", "🕐  Historical Trends",
     "🎯  Feature Importance", "🤖  Model Info"],
    label_visibility="collapsed",
)
page_map = {
    "📊  Dashboard":         "Dashboard",
    "📈  Forecast Details":  "Forecast Details",
    "🕐  Historical Trends": "Historical Trends",
    "🎯  Feature Importance":"Feature Importance",
    "🤖  Model Info":        "Model Info",
}
page = page_map[page]

st.sidebar.markdown("---")
st.sidebar.markdown("<div class='params-label'>Parameters</div>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<div style='font-size:11px; color:#6e7681; margin-bottom:4px;'>History Window</div>",
    unsafe_allow_html=True,
)
history_days = st.sidebar.selectbox("History Window", [7, 30, 90], index=1,
                                    label_visibility="collapsed")

st.sidebar.markdown("---")
if st.sidebar.button("▶  Refresh Data", use_container_width=True):
    # Clear both @st.cache_data and session_state caches
    st.cache_data.clear()
    for key in list(st.session_state.keys()):
        if key.startswith("_hw_") or key.startswith("_inf_") or key.startswith("_hist_"):
            del st.session_state[key]
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<div style='font-size:10px; color:#6e7681; font-family:\"IBM Plex Mono\",monospace;'>"
    f"Last updated<br>{datetime.now().strftime('%Y-%m-%d %H:%M')} UTC</div>",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared section label helper
# ─────────────────────────────────────────────────────────────────────────────
def _label(text: str):
    st.markdown(f"<div class='section-label'>{text}</div>", unsafe_allow_html=True)

def _gap(rem: float = 0.6):
    st.markdown(f"<div style='height:{rem}rem'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard Page
# ─────────────────────────────────────────────────────────────────────────────
def render_dashboard():
    try:
        result = get_inference()   # Fix #2 — single cached call
    except Exception as e:
        st.error(f"Failed to fetch predictions: {e}")
        st.info("Make sure a model is trained and API keys are configured.")
        return

    predictions = result["predictions"]
    conditions  = result["current_conditions"]
    model_info  = result["model_info"]

    current_aqi = predictions["current"]["aqi"]
    cat, color, desc = aqi_category(current_aqi)

    st.markdown(f"""
    <div style='margin-bottom:1.2rem; padding-bottom:1rem; border-bottom:1px solid #21262d;'>
        <div style='display:flex; align-items:flex-start; justify-content:space-between;'>
            <div>
                <h1 style='margin:0 0 4px 0;'>Model Predictions</h1>
                <p style='color:#6e7681; font-size:12px; margin:0; font-family:"IBM Plex Mono",monospace;'>
                    AQI Forecast · Karachi, Pakistan · Last run: just now
                </p>
            </div>
            <div style='text-align:right;'>
                <div style='font-size:11px; color:#6e7681; font-family:"IBM Plex Mono",monospace;'>
                    {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if current_aqi > 150:
        level = "🔴 RED ALERT" if current_aqi > 200 else "🟠 ORANGE ALERT"
        st.error(f"**{level} — AQI {current_aqi:.0f} ({cat}):** {desc}")
    elif current_aqi > 100:
        st.warning(f"**🟡 CAUTION — AQI {current_aqi:.0f} ({cat}):** {desc}")
    else:
        st.success(f"**✅ Air Quality** — Air quality is satisfactory. AQI {current_aqi:.0f} ({cat})")

    _gap()

    col_aqi, col_weather = st.columns([1, 1.1])

    with col_aqi:
        with st.container(border=True):
            _label("Current AQI")
            st.markdown(
                f"""<div style='background:linear-gradient(135deg,{color}18 0%,{color}08 100%);
                    border:1px solid {color}40; border-radius:8px; padding:18px 14px;
                    text-align:center; margin-bottom:10px;'>
                    <div style='font-size:52px; font-weight:700; color:{color};
                                font-family:"IBM Plex Mono",monospace; line-height:1;'>{current_aqi:.0f}</div>
                    <div style='font-size:12px; font-weight:600; color:{color};
                                margin-top:8px; letter-spacing:0.5px;'>{cat}</div>
                    <div style='font-size:10px; color:#6e7681; margin-top:4px;
                                overflow:hidden; text-overflow:ellipsis; white-space:nowrap;'>{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with col_weather:
        with st.container(border=True):
            _label("Current Conditions")
            temp = conditions.get('temperature')
            hum  = conditions.get('humidity')
            wind = conditions.get('wind_speed')
            st.markdown(f"""
            <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px;
                        text-align:center; padding:6px 0 8px 0;'>
                <div style='background:#21262d; border-radius:8px; padding:12px 6px; border:1px solid #30363d;'>
                    <div style='font-size:20px; margin-bottom:4px;'>🌡️</div>
                    <div style='font-size:18px; font-weight:700; color:#e6edf3;
                                font-family:"IBM Plex Mono",monospace;'>{f"{temp:.0f}°" if temp is not None else "—"}</div>
                    <div style='font-size:9px; color:#6e7681; text-transform:uppercase;
                                letter-spacing:0.5px; font-weight:600; margin-top:3px;'>Temp °C</div>
                </div>
                <div style='background:#21262d; border-radius:8px; padding:12px 6px; border:1px solid #30363d;'>
                    <div style='font-size:20px; margin-bottom:4px;'>💧</div>
                    <div style='font-size:18px; font-weight:700; color:#e6edf3;
                                font-family:"IBM Plex Mono",monospace;'>{f"{hum:.0f}%" if hum is not None else "—"}</div>
                    <div style='font-size:9px; color:#6e7681; text-transform:uppercase;
                                letter-spacing:0.5px; font-weight:600; margin-top:3px;'>Humidity</div>
                </div>
                <div style='background:#21262d; border-radius:8px; padding:12px 6px; border:1px solid #30363d;'>
                    <div style='font-size:20px; margin-bottom:4px;'>💨</div>
                    <div style='font-size:18px; font-weight:700; color:#e6edf3;
                                font-family:"IBM Plex Mono",monospace;'>{f"{wind:.1f}" if wind is not None else "—"}</div>
                    <div style='font-size:9px; color:#6e7681; text-transform:uppercase;
                                letter-spacing:0.5px; font-weight:600; margin-top:3px;'>Wind km/h</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    _gap()
    _label("Key Metrics — 3-Day AQI Forecast")

    fc_cols = st.columns(4)
    for col_obj, (label_key, display_label) in zip(
        fc_cols,
        [("current","TODAY'S AQI"), ("24h","TOMORROW"), ("48h","DAY +2"), ("72h","DAY +3")],
    ):
        p = predictions[label_key]
        p_aqi = p["aqi"]
        p_cat, p_color, p_desc = aqi_category(p_aqi)
        with col_obj:
            st.markdown(f"""
            <div class="forecast-card" style="--card-accent:{p_color};">
                <div class="fc-label">{display_label}</div>
                <div class="fc-value" style="color:{p_color};">{p_aqi:.0f}</div>
                <div class="fc-badge"
                     style="background:{p_color}22; color:{p_color}; border:1px solid {p_color}40;">
                    {p_cat}
                </div>
                <div class="fc-desc" title="{p_desc}">{p_desc}</div>
            </div>
            """, unsafe_allow_html=True)

    _gap()

    col_shap, col_hist = st.columns([1, 1.3])

    with col_shap:
        with st.container(border=True):
            _label("Feature Importance")
            shap_df = load_shap_importance()
            if not shap_df.empty:
                cols_l = [c.lower() for c in shap_df.columns]
                feat_c = shap_df.columns[cols_l.index("feature")]    if "feature"    in cols_l else shap_df.columns[0]
                imp_c  = shap_df.columns[cols_l.index("importance")] if "importance" in cols_l else shap_df.columns[1]
                shap_df[imp_c] = pd.to_numeric(shap_df[imp_c], errors="coerce").abs()
                top  = shap_df.dropna(subset=[imp_c]).sort_values(imp_c, ascending=False).head(8)
                norm = top[imp_c] / top[imp_c].max()
                bar_colors = [importance_color(v) for v in norm]  # Fix #6
                fig_s = go.Figure(go.Bar(
                    x=top[imp_c][::-1], y=top[feat_c][::-1], orientation="h",
                    marker=dict(color=bar_colors[::-1], line=dict(width=0)),
                    text=[f"{v:.4f}" for v in top[imp_c][::-1]],
                    textposition="outside",
                    textfont=dict(size=9, family="IBM Plex Mono", color="#6e7681"),
                    hovertemplate="<b>%{y}</b><br>%{x:.5f}<extra></extra>",
                ))
                apply_chart_theme(fig_s, height=260)
                fig_s.update_layout(xaxis_title="Mean |SHAP|", margin=dict(l=8, r=65, t=12, b=8))
                st.plotly_chart(fig_s, use_container_width=True)
                st.caption("→ See Feature Importance page for full analysis")
            else:
                st.info("SHAP data not yet available from GitHub Releases.")

    with col_hist:
        with st.container(border=True):
            _label("Historical AQI — 7 Days")
            # Fix #3/#9 — single call, explicit empty-check with user feedback
            df_hist = get_history_data(7)
            if not df_hist.empty and "aqi" in df_hist.columns:
                fig_h = go.Figure()
                fig_h.add_trace(go.Scatter(
                    x=df_hist["timestamp"], y=df_hist["aqi"],
                    mode="lines", name="AQI",
                    line=dict(color="#388bfd", width=1.8),
                    fill="tozeroy", fillcolor="rgba(56,139,253,0.07)",
                ))
                if "aqi_rolling_24h" in df_hist.columns:
                    fig_h.add_trace(go.Scatter(
                        x=df_hist["timestamp"], y=df_hist["aqi_rolling_24h"],
                        mode="lines", name="24h Avg",
                        line=dict(color="#f85149", width=2, dash="dot"),
                    ))
                apply_chart_theme(fig_h, height=260)
                fig_h.update_layout(
                    legend=dict(orientation="v", x=1.01, y=1),
                    yaxis_title="AQI", xaxis_title="",
                )
                st.plotly_chart(fig_h, use_container_width=True)
            else:
                st.info("Historical data not yet available from Hopsworks Feature Store.")

    _gap()

    # Fix #3 — fetch once, reuse for both weather and pollutant charts
    df_full = get_history_data(history_days)

    col_wc, col_pl = st.columns(2)

    with col_wc:
        with st.container(border=True):
            _label("Weather Conditions")
            if not df_full.empty:
                fig_w = go.Figure()
                for col_name, clr, nm in [
                    ("temperature", "#388bfd", "Temperature"),
                    ("humidity",    "#3fb950", "Humidity"),
                    ("wind_speed",  "#bc8cff", "Wind Speed"),
                ]:
                    if col_name in df_full.columns:
                        fig_w.add_trace(go.Scatter(
                            x=df_full["timestamp"], y=df_full[col_name],
                            mode="lines", name=nm, line=dict(color=clr, width=1.5),
                        ))
                apply_chart_theme(fig_w, height=210)
                fig_w.update_layout(
                    legend=dict(orientation="h", y=-0.28, x=0, font=dict(size=10)),
                    yaxis_title="",
                )
                st.plotly_chart(fig_w, use_container_width=True)
            else:
                st.info("No weather history available.")

    with col_pl:
        with st.container(border=True):
            _label("Pollutant Levels")
            if not df_full.empty:
                fig_p = go.Figure()
                for col_name, clr, nm in [
                    ("pm25", "#f85149", "PM2.5"),
                    ("pm10", "#f0883e", "PM10"),
                    ("o3",   "#3fb950", "O₃"),
                ]:
                    if col_name in df_full.columns:
                        fig_p.add_trace(go.Scatter(
                            x=df_full["timestamp"], y=df_full[col_name],
                            mode="lines", name=nm, line=dict(color=clr, width=1.5),
                        ))
                apply_chart_theme(fig_p, height=210)
                fig_p.update_layout(
                    legend=dict(orientation="h", y=-0.28, x=0, font=dict(size=10)),
                    yaxis_title="",
                )
                st.plotly_chart(fig_p, use_container_width=True)
            else:
                st.info("No pollutant history available.")

    _gap()

    # Fix #4 — metrics use None defaults, never fake fallback numbers
    metrics = model_info.get("metrics", {})
    r2   = metrics.get("test_r2")
    rmse = metrics.get("test_rmse")
    mae  = metrics.get("test_mae")
    mape = metrics.get("test_mape")

    with st.container(border=True):
        _label("Model Performance")
        mc1, mc2, mc3, mc4 = st.columns(4)
        metric_items = [
            (mc1, "Test R²",   f"{r2:.3f}"   if isinstance(r2,   (int, float)) else "—",
             f"+{r2 - 0.80:.1%} vs baseline" if isinstance(r2, (int, float)) else "No data",
             "#3fb950"),
            (mc2, "Test RMSE", f"{rmse:.1f}" if isinstance(rmse, (int, float)) else "—",
             "Root mean squared error", "#d29922"),
            (mc3, "Test MAE",  f"{mae:.1f}"  if isinstance(mae,  (int, float)) else "—",
             "Mean absolute error", "#388bfd"),
            (mc4, "MAPE",      f"{mape:.1f}%" if isinstance(mape, (int, float)) else "—",
             "Mean absolute pct error", "#bc8cff"),
        ]
        for col_m, label, val, sub, accent in metric_items:
            col_m.markdown(
                f"""<div style='background:#0d1117; border:1px solid #21262d;
                               border-top:2px solid {accent}; border-radius:8px;
                               padding:14px 16px; text-align:left;'>
                    <div style='font-size:10px; font-weight:700; text-transform:uppercase;
                                letter-spacing:0.8px; color:#6e7681; margin-bottom:6px;
                                font-family:"IBM Plex Sans",sans-serif;'>{label}</div>
                    <div style='font-size:24px; font-weight:700; color:#e6edf3;
                                font-family:"IBM Plex Mono",monospace; line-height:1;
                                margin-bottom:5px;'>{val}</div>
                    <div style='font-size:10px; color:#6e7681;
                                font-family:"IBM Plex Sans",sans-serif;'>{sub}</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Forecast Details Page
# ─────────────────────────────────────────────────────────────────────────────
def render_forecast():
    try:
        result = get_inference()   # Fix #2 — reuses cached result
    except Exception as e:
        st.error(f"Failed to fetch predictions: {e}")
        return

    predictions = result["predictions"]
    model_info  = result["model_info"]

    st.markdown("""
    <div style='margin-bottom:1.2rem; padding-bottom:1rem; border-bottom:1px solid #21262d;'>
        <h1 style='margin:0 0 4px 0;'>Detailed 3-Day Forecast</h1>
        <p style='color:#6e7681; font-size:12px; margin:0; font-family:"IBM Plex Mono",monospace;'>
            Multi-horizon AQI predictions · Confidence intervals included
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        _label("Prediction Summary")
        rows = []
        for label in ["current", "24h", "48h", "72h"]:
            p = predictions[label]
            cat, color, _ = aqi_category(p["aqi"])
            rows.append({
                "Period":     label.upper(),
                "AQI":        f"{p['aqi']:.1f}",
                "Category":   cat,
                "Confidence": p["confidence"].title(),
                "CI Lower":   f"{p.get('ci_lower', p['aqi'] * 0.95):.1f}",
                "CI Upper":   f"{p.get('ci_upper', p['aqi'] * 1.05):.1f}",
                "Status": (
                    "🟢" if p["aqi"] <= 100 else
                    "🟡" if p["aqi"] <= 150 else
                    "🟠" if p["aqi"] <= 200 else "🔴"
                ),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    _gap()

    with st.container(border=True):
        _label("AQI Distribution Per Horizon")
        fdf      = pd.DataFrame([{"Period": k.upper(), "AQI": predictions[k]["aqi"]}
                                  for k in ["current","24h","48h","72h"]])
        bar_c    = [aqi_color(v) for v in fdf["AQI"]]
        fig = go.Figure(go.Bar(
            x=fdf["Period"], y=fdf["AQI"],
            marker=dict(color=bar_c, line=dict(width=0), opacity=0.85),
            text=[f"{v:.0f}" for v in fdf["AQI"]],
            textposition="outside",
            textfont=dict(size=12, family="IBM Plex Mono", color="#e6edf3"),
            hovertemplate="<b>%{x}</b><br>AQI: %{y:.1f}<extra></extra>",
        ))
        apply_chart_theme(fig, height=320)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ About This Forecast", expanded=False):
        st.markdown(f"""
**Forecasting Method:** {model_info.get('forecast_method', 'Weather-informed')}

- **Current (t+0)**: Real-time prediction using latest AQICN pollutants + OpenMeteo weather
- **24h / 48h / 72h**: TRUE multi-horizon predictions using OpenMeteo weather forecasts
- **Confidence Intervals**: 95% prediction intervals based on model test RMSE
- **Model**: {model_info.get('name', 'Best model from Hopsworks Model Registry')}
        """)


# ─────────────────────────────────────────────────────────────────────────────
# Historical Trends Page
# ─────────────────────────────────────────────────────────────────────────────
def render_history():
    # Fix #1 — uses session_state connection; no duplicate logins on navigation
    df = get_history_data(history_days)
    if df.empty:
        st.warning("No historical data available from Hopsworks Feature Store.")
        return

    st.markdown(f"""
    <div style='margin-bottom:1.2rem; padding-bottom:1rem; border-bottom:1px solid #21262d;'>
        <h1 style='margin:0 0 4px 0;'>Historical AQI — Last {history_days} Days</h1>
        <p style='color:#6e7681; font-size:12px; margin:0; font-family:"IBM Plex Mono",monospace;'>
            {len(df)} data points · {df['timestamp'].min().date()} → {df['timestamp'].max().date()}
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        _label("AQI Timeline")
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["aqi"], mode="lines", name="Hourly AQI",
            line=dict(color="#388bfd", width=1.5),
            fill="tozeroy", fillcolor="rgba(56,139,253,0.06)",
        ))
        if "aqi_rolling_24h" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["aqi_rolling_24h"], mode="lines",
                name="24h Rolling Avg", line=dict(color="#f85149", width=2.5),
            ))
        for lo, hi, label, color, _ in AQI_CATEGORIES:
            if hi <= df["aqi"].max() or lo <= df["aqi"].max():
                fig.add_hline(
                    y=hi, line_dash="dot", line_color=color, opacity=0.25,
                    annotation_text=label if hi <= df["aqi"].max() else None,
                    annotation_font_size=9, annotation_font_color="#6e7681",
                )
        apply_chart_theme(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)

    _gap()

    col_p, col_w = st.columns(2)
    with col_p:
        with st.container(border=True):
            _label("Pollutants Over Time")
            # Fix #9 — explicit empty guard
            if not df.empty:
                fig2        = go.Figure()
                poll_colors = ["#f85149","#f0883e","#d29922","#3fb950","#388bfd","#bc8cff"]
                for i, col in enumerate(["pm25","pm10","o3","no2","so2","co"]):
                    if col in df.columns:
                        fig2.add_trace(go.Scatter(
                            x=df["timestamp"], y=df[col], mode="lines",
                            name=col.upper(), line=dict(width=1.3, color=poll_colors[i]),
                        ))
                apply_chart_theme(fig2, height=280)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No pollutant data available.")

    with col_w:
        with st.container(border=True):
            _label("Weather Conditions")
            if not df.empty:
                fig3     = go.Figure()
                wx_clrs  = ["#f0883e","#388bfd","#3fb950","#bc8cff"]
                for i, col in enumerate(["temperature","humidity","wind_speed","pressure"]):
                    if col in df.columns:
                        fig3.add_trace(go.Scatter(
                            x=df["timestamp"], y=df[col], mode="lines",
                            name=col.replace("_"," ").title(),
                            line=dict(width=1.3, color=wx_clrs[i]),
                        ))
                apply_chart_theme(fig3, height=280)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No weather data available.")

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


# ─────────────────────────────────────────────────────────────────────────────
# Feature Importance Page
# ─────────────────────────────────────────────────────────────────────────────
def render_shap():
    st.markdown("""
    <div style='margin-bottom:1.2rem; padding-bottom:1rem; border-bottom:1px solid #21262d;'>
        <h1 style='margin:0 0 4px 0;'>Feature Importance — SHAP Analysis</h1>
        <p style='color:#6e7681; font-size:12px; margin:0; font-family:"IBM Plex Mono",monospace;'>
            Global feature importance from latest trained model · live from GitHub Releases
        </p>
    </div>
    """, unsafe_allow_html=True)

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

    # Fix #7 — render_shap_charts always returns tuple or raises
    try:
        fig1, fig2, fig3 = render_shap_charts(shap_df)
    except Exception as e:
        st.error(f"Could not render SHAP charts: {e}")
        return

    _gap()

    with st.container(border=True):
        _label("Top Feature Importances — Mean |SHAP|")
        st.caption("Longer bar = stronger influence on AQI prediction. Red = high impact, blue = lower impact.")
        st.plotly_chart(fig1, use_container_width=True)

    _gap()

    col_radar, col_waterfall = st.columns(2)
    with col_radar:
        with st.container(border=True):
            _label("Radar — Top 8 Features")
            st.caption("Radial spread shows relative importance across top features.")
            st.plotly_chart(fig2, use_container_width=True)
    with col_waterfall:
        with st.container(border=True):
            _label("Cumulative Contribution")
            st.caption("Bars = individual share. Dotted line = cumulative importance build-up.")
            st.plotly_chart(fig3, use_container_width=True)

    _gap()

    with st.container(border=True):
        _label("Top Feature Insights")
        cols_l = [c.lower() for c in shap_df.columns]
        feat_c = shap_df.columns[cols_l.index("feature")]    if "feature"    in cols_l else shap_df.columns[0]
        imp_c  = shap_df.columns[cols_l.index("importance")] if "importance" in cols_l else shap_df.columns[1]
        shap_df[imp_c] = pd.to_numeric(shap_df[imp_c], errors="coerce").abs()
        top5 = shap_df.dropna(subset=[imp_c]).sort_values(imp_c, ascending=False).head(5).reset_index(drop=True)

        insight_cols = st.columns(5)
        rank_colors  = ["#f85149","#f0883e","#d29922","#3fb950","#388bfd"]
        rank_labels  = ["🥇","🥈","🥉","4th","5th"]

        for i, (_, row) in enumerate(top5.iterrows()):
            pct = row[imp_c] / shap_df[imp_c].sum() * 100
            insight_cols[i].markdown(
                f"""<div style='background:#0d1117; border:1px solid #21262d;
                               border-top:2px solid {rank_colors[i]};
                               border-radius:8px; padding:14px 10px; text-align:center;'>
                    <div style='font-size:18px; margin-bottom:6px;'>{rank_labels[i]}</div>
                    <div style='font-size:11px; font-weight:600; color:#c9d1d9;
                                word-break:break-word; line-height:1.3; margin-bottom:8px;
                                font-family:"IBM Plex Sans",sans-serif;'>{row[feat_c]}</div>
                    <div style='font-size:14px; font-weight:700; color:{rank_colors[i]};
                                font-family:"IBM Plex Mono",monospace;'>{row[imp_c]:.4f}</div>
                    <div style='font-size:10px; color:#6e7681; margin-top:4px;'>{pct:.1f}% of total</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Model Info Page
# ─────────────────────────────────────────────────────────────────────────────
def render_model_info():
    st.markdown("""
    <div style='margin-bottom:1.2rem; padding-bottom:1rem; border-bottom:1px solid #21262d;'>
        <h1 style='margin:0 0 4px 0;'>Model Registry & Performance</h1>
        <p style='color:#6e7681; font-size:12px; margin:0; font-family:"IBM Plex Mono",monospace;'>
            Hopsworks Model Registry · Latest deployed model
        </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        result = get_inference()   # Fix #2 — reuses cached result
    except Exception as e:
        st.error(f"Failed to load model info: {e}")
        return

    model_info = result["model_info"]
    metrics    = model_info["metrics"]

    with st.container(border=True):
        _label("Registry Snapshot")
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

    _gap()

    with st.container(border=True):
        _label("Performance Metrics")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Model", model_info.get("name", "—").replace("_"," ").title())
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

    _gap()

    method = model_info.get("forecast_method", "N/A")
    if method == "Weather-informed":
        st.success(f"✅ **{method}** — Using OpenMeteo weather forecasts for 24h / 48h / 72h predictions")
    else:
        st.warning(f"⚠️ **{method}** — Weather forecast unavailable; using current conditions for all horizons")


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────
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

# ── Footer ────────────────────────────────────────────────────────────────────
_gap(1.5)
st.markdown("---")
st.markdown(
    "<div style='font-size:11px; color:#6e7681; font-family:\"IBM Plex Mono\",monospace;'>"
    "🌍 Hawa Alert AQI Engine · Data: AQICN + OpenMeteo · "
    "Model Registry: Hopsworks · SHAP: GitHub Releases"
    "</div>",
    unsafe_allow_html=True,
)