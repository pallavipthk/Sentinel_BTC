"""BTC Sentinel - Institutional Quantitative Bitcoin Analytics & Crash Radar.
Explainable, Regime-Aware Bitcoin Forecasting & Drawdown Early-Warning Dashboard.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & ROOT PATH RESOLUTION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BTC Sentinel | Quantitative Bitcoin Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "reports" / "results"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
DATA_FILE = ROOT_DIR / "data" / "processed" / "btc_daily.csv"


# -----------------------------------------------------------------------------
# 2. STATE & VIEW CONTROLLER CALLBACKS
# -----------------------------------------------------------------------------
def set_view(view_name: str):
    """Callback to switch application views reliably before rerun."""
    st.session_state["app_view"] = view_name
    try:
        st.query_params["view"] = view_name
    except Exception:
        pass


# Initialize state from query params or default to "landing"
if "view" in st.query_params and st.query_params["view"] in ["landing", "dashboard"]:
    st.session_state["app_view"] = st.query_params["view"]
elif "app_view" not in st.session_state:
    st.session_state["app_view"] = "landing"


# -----------------------------------------------------------------------------
# 3. DESIGN SYSTEM & MODERN SAAS CSS INJECTION
# -----------------------------------------------------------------------------
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800;900&display=swap" rel="stylesheet">

    <style>
    /* Global Reset & Base Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0e172a 0%, #080c14 100%) !important;
        color: #f1f5f9;
    }

    /* Hide Streamlit Default Chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0px !important;
    }
    
    /* Layout Container Spacing */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1440px !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
        color: #ffffff !important;
    }

    .code-font {
        font-family: 'Fira Code', monospace !important;
    }

    /* =========================================================================
       TOP GLOBAL NAVIGATION BAR
       ========================================================================= */
    .sentinel-nav-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(135deg, rgba(19, 27, 46, 0.85) 0%, rgba(13, 20, 36, 0.95) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 12px 24px;
        margin-bottom: 16px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), 0 0 20px rgba(56, 189, 248, 0.05);
    }

    .nav-brand-box {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .nav-logo-icon {
        font-size: 26px;
        filter: drop-shadow(0 0 10px rgba(236, 72, 153, 0.6));
    }

    .nav-brand-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 20px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #ffffff 30%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }

    .nav-brand-sub {
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #ec4899;
    }

    .nav-status-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-family: 'Fira Code', monospace;
        font-size: 11.5px;
        font-weight: 600;
        color: #34d399;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 6px 14px;
        border-radius: 30px;
    }

    .pulse-green {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 12px #10b981;
        animation: pulseAnimation 2s infinite;
    }

    @keyframes pulseAnimation {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* =========================================================================
       LIVE MARKET TICKER RIBBON
       ========================================================================= */
    .market-ticker-ribbon {
        display: flex;
        align-items: center;
        gap: 16px;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 8px 18px;
        margin-bottom: 20px;
        overflow-x: auto;
        white-space: nowrap;
        font-size: 12px;
    }
    
    .ticker-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .ticker-label {
        color: #94a3b8;
        font-weight: 500;
    }
    .ticker-val {
        font-family: 'Fira Code', monospace;
        font-weight: 600;
        color: #f8fafc;
    }
    .ticker-sep {
        color: rgba(255, 255, 255, 0.15);
    }

    /* =========================================================================
       HERO BANNER & CARDS
       ========================================================================= */
    .hero-badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-family: 'Fira Code', monospace;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        background: rgba(236, 72, 153, 0.15);
        color: #f472b6;
        padding: 5px 14px;
        border-radius: 20px;
        margin-bottom: 16px;
        border: 1px solid rgba(236, 72, 153, 0.35);
    }

    .hero-title-large {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 46px;
        font-weight: 900;
        letter-spacing: -1.4px;
        line-height: 1.12;
        margin-bottom: 14px;
        background: linear-gradient(135deg, #ffffff 0%, #38bdf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-tagline-code {
        font-family: 'Fira Code', monospace;
        font-size: 14.5px;
        font-weight: 500;
        color: #cbd5e1;
        margin-bottom: 16px;
    }

    .hero-description-text {
        font-size: 15.5px;
        color: #94a3b8;
        line-height: 1.65;
        max-width: 640px;
        margin-bottom: 28px;
    }

    .hero-stats-flex {
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        margin-bottom: 28px;
    }

    .hero-stat-box {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px 18px;
        min-width: 125px;
        text-align: center;
        backdrop-filter: blur(8px);
    }

    .hero-stat-box-val {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 20px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 2px;
    }

    .hero-stat-box-lbl {
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
    }

    /* =========================================================================
       GLASSMORPHIC KPI CARDS
       ========================================================================= */
    .kpi-card {
        background: linear-gradient(145deg, rgba(20, 29, 48, 0.75) 0%, rgba(13, 20, 36, 0.9) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 20px;
        min-height: 124px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 20px -4px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        margin-bottom: 14px;
        position: relative;
        overflow: hidden;
    }

    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px -6px rgba(0, 0, 0, 0.6), 0 0 20px rgba(56, 189, 248, 0.15);
        border-color: rgba(56, 189, 248, 0.4);
    }

    .kpi-top-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 4px;
    }

    .kpi-label {
        font-size: 11.5px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #94a3b8;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .kpi-value-text {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 26px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
        line-height: 1.15;
        margin: 2px 0 6px 0;
    }

    .kpi-badge-pill {
        display: inline-block;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 6px;
        width: fit-content;
    }

    .badge-emerald { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }
    .badge-cyan    { background: rgba(6, 182, 212, 0.15);  color: #38bdf8; border: 1px solid rgba(6, 182, 212, 0.35); }
    .badge-amber   { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }
    .badge-rose    { background: rgba(244, 63, 94, 0.15);  color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.35); }
    .badge-purple  { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.35); }

    /* =========================================================================
       PILLARS & SYNTHESIS CONTAINERS
       ========================================================================= */
    .pillar-card {
        background: linear-gradient(145deg, rgba(20, 29, 48, 0.75) 0%, rgba(13, 20, 36, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 22px 24px;
        min-height: 195px;
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.3);
        border-top: 3px solid #ec4899;
        margin-bottom: 14px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .pillar-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px -4px rgba(0, 0, 0, 0.5), 0 0 15px rgba(236, 72, 153, 0.15);
    }
    .pillar-icon {
        font-size: 26px;
        margin-bottom: 8px;
    }
    .pillar-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 16px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
    }
    .pillar-desc {
        font-size: 13px;
        color: #94a3b8;
        line-height: 1.55;
    }

    .pipeline-step-box {
        background: linear-gradient(145deg, rgba(20, 29, 48, 0.75) 0%, rgba(13, 20, 36, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        margin-bottom: 8px;
    }
    .pipeline-step-badge {
        font-size: 11px;
        font-weight: 700;
        color: #f472b6;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .pipeline-step-heading {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 13.5px;
        font-weight: 700;
        color: #ffffff;
        margin: 4px 0px;
    }
    .pipeline-step-detail {
        font-size: 11.5px;
        color: #94a3b8;
        line-height: 1.4;
    }

    .glass-card {
        background: linear-gradient(145deg, rgba(20, 29, 48, 0.7) 0%, rgba(13, 20, 36, 0.85) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.4);
    }

    .glass-card-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 15px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .factor-pill-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }

    /* Terminal Code Window */
    .terminal-box {
        background: #070a10;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        overflow: hidden;
        font-family: 'Fira Code', monospace;
        box-shadow: 0 20px 45px -10px rgba(0, 0, 0, 0.8), 0 0 25px rgba(236, 72, 153, 0.15);
    }
    .terminal-topbar {
        background: #0f172a;
        padding: 10px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .terminal-circles {
        display: flex;
        gap: 6px;
    }
    .t-circle {
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }
    .t-c-red { background: #ef4444; }
    .t-c-yellow { background: #f59e0b; }
    .t-c-green { background: #10b981; }

    /* Custom Styled Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 6px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 10px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 13.5px;
        font-weight: 600;
        color: #94a3b8;
        border: none !important;
        background: transparent;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: linear-gradient(135deg, rgba(236, 72, 153, 0.25) 0%, rgba(56, 189, 248, 0.25) 100%) !important;
        border: 1px solid rgba(236, 72, 153, 0.4) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }

    /* Primary and Secondary Streamlit Buttons */
    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="stBaseButton-primary"],
    div.stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%) !important;
        border: none !important;
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        padding: 10px 24px !important;
        border-radius: 30px !important;
        box-shadow: 0 4px 18px rgba(236, 72, 153, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="stBaseButton-primary"]:hover,
    div.stButton > button[data-testid="baseButton-primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(236, 72, 153, 0.6) !important;
    }
    div.stButton > button[kind="secondary"],
    div.stButton > button[data-testid="stBaseButton-secondary"],
    div.stButton > button[data-testid="baseButton-secondary"] {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        color: #f8fafc !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
        padding: 8px 20px !important;
        border-radius: 25px !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="secondary"]:hover,
    div.stButton > button[data-testid="stBaseButton-secondary"]:hover,
    div.stButton > button[data-testid="baseButton-secondary"]:hover {
        background: rgba(255, 255, 255, 0.12) !important;
        border-color: rgba(56, 189, 248, 0.4) !important;
    }

    /* Custom Dataframe & Table Styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 4. DATA LOADING & RESILIENT FALLBACK MANAGEMENT
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_json_artifact(filename: str) -> Optional[Dict[str, Any]]:
    """Safely loads a JSON report artifact from the results directory."""
    path = RESULTS_DIR / filename
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


@st.cache_data(ttl=300)
def load_csv_artifact(filename: str) -> Optional[pd.DataFrame]:
    """Safely loads a CSV artifact from the results directory."""
    path = RESULTS_DIR / filename
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


@st.cache_data(ttl=300)
def load_market_data() -> Optional[pd.DataFrame]:
    """Loads historical daily Bitcoin price series."""
    if DATA_FILE.exists():
        try:
            df = pd.read_csv(DATA_FILE)
            df["Date"] = pd.to_datetime(df["Date"])
            return df
        except Exception:
            return None
    return None


# -----------------------------------------------------------------------------
# 5. REUSABLE UI COMPONENTS
# -----------------------------------------------------------------------------
def render_header(telemetry: Dict[str, Any]):
    """Renders the top persistent institutional navigation header with rock-solid callbacks."""
    as_of = telemetry.get("as_of_date", "2026-09-29")
    current_view = st.session_state.get("app_view", "landing")

    col_brand, col_nav = st.columns([1.5, 1], gap="medium")

    with col_brand:
        st.markdown(
            f"""
            <div class="nav-brand-box">
                <div class="nav-logo-icon">🛡️</div>
                <div>
                    <div class="nav-brand-title">BTC SENTINEL</div>
                    <div class="nav-brand-sub">Institutional Quantitative Analytics & Crash Radar</div>
                </div>
                <div style="margin-left: 12px;" class="nav-status-chip">
                    <span class="pulse-green"></span> AS OF {as_of}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_nav:
        b_col1, b_col2 = st.columns([1, 1.3])
        with b_col1:
            if current_view == "dashboard":
                st.button(
                    "🏠 Landing Page",
                    on_click=set_view,
                    args=("landing",),
                    type="secondary",
                    use_container_width=True,
                    key="btn_nav_to_landing",
                )
            else:
                st.markdown(
                    """
                    <div style="text-align: center; padding: 8px 0; color: #34d399; font-size: 12px; font-weight: 700;">
                        ● System Online
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with b_col2:
            if current_view == "landing":
                st.button(
                    "🚀 Open Terminal ➔",
                    on_click=set_view,
                    args=("dashboard",),
                    type="primary",
                    use_container_width=True,
                    key="btn_nav_to_terminal",
                )
            else:
                st.markdown(
                    """
                    <div style="text-align: right; padding: 8px 0; color: #38bdf8; font-size: 12px; font-weight: 700;">
                        📊 Pro Terminal Active
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)


def render_market_ticker(market_df: pd.DataFrame, telemetry: Dict[str, Any]):
    """Renders the continuous live Bitcoin market stats ticker ribbon."""
    curr_p = telemetry.get("current_price", 83701.79)
    prev_close = market_df["Close"].iloc[-2] if len(market_df) > 1 else curr_p
    p_change = (curr_p - prev_close) / prev_close * 100.0
    p_change_str = f"{p_change:+.2f}%"
    p_color = "#34d399" if p_change >= 0 else "#f87171"

    f_price = telemetry.get("forecast_price_1d", 83832.87)
    exp_ret = telemetry.get("expected_return_1d_pct", 0.16)
    garch_vol = telemetry.get("volatility_forecast_annualized_pct", 47.3)
    crash_pct = telemetry.get("model_estimated_crash_risk_pct", 27.5)
    crash_lvl = telemetry.get("crash_risk_level", "MODERATE")
    regime = telemetry.get("market_regime", "BULL_MOMENTUM")
    conf = telemetry.get("qualitative_confidence", "HIGH")
    analog_date = telemetry.get("top_similar_date", "2023-06-29")
    analog_score = telemetry.get("top_similarity_score_pct", 93.7)

    st.markdown(
        f"""
        <div class="market-ticker-ribbon">
            <div class="ticker-item">
                <span class="ticker-label">BTC/USD:</span>
                <span class="ticker-val" style="color: #38bdf8;">${curr_p:,.2f}</span>
                <span style="color: {p_color}; font-weight: 700;">({p_change_str})</span>
            </div>
            <span class="ticker-sep">|</span>
            <div class="ticker-item">
                <span class="ticker-label">1-Day Target:</span>
                <span class="ticker-val" style="color: #34d399;">${f_price:,.2f} ({exp_ret:+.2f}%)</span>
            </div>
            <span class="ticker-sep">|</span>
            <div class="ticker-item">
                <span class="ticker-label">GARCH Vol:</span>
                <span class="ticker-val" style="color: #fbbf24;">{garch_vol:.1f}% Ann.</span>
            </div>
            <span class="ticker-sep">|</span>
            <div class="ticker-item">
                <span class="ticker-label">14d Crash Risk:</span>
                <span class="ticker-val" style="color: {'#34d399' if crash_lvl=='LOW' else ('#fbbf24' if crash_lvl=='MODERATE' else '#f87171')};">{crash_pct:.1f}% [{crash_lvl}]</span>
            </div>
            <span class="ticker-sep">|</span>
            <div class="ticker-item">
                <span class="ticker-label">Macro Regime:</span>
                <span class="ticker-val" style="color: #c084fc;">{regime}</span>
            </div>
            <span class="ticker-sep">|</span>
            <div class="ticker-item">
                <span class="ticker-label">Confidence:</span>
                <span class="ticker-val" style="color: #34d399;">{conf} (3/4)</span>
            </div>
            <span class="ticker-sep">|</span>
            <div class="ticker-item">
                <span class="ticker-label">Top Analog:</span>
                <span class="ticker-val" style="color: #38bdf8;">{analog_date} ({analog_score:.1f}%)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_barometer_grid(market_df: pd.DataFrame, telemetry: Dict[str, Any]):
    """Renders a 6-card institutional barometer grid."""
    curr_price = telemetry.get("current_price", 83701.79)
    prev_close = market_df["Close"].iloc[-2] if len(market_df) > 1 else curr_price
    price_change = (curr_price - prev_close) / prev_close * 100.0
    p_badge = "badge-emerald" if price_change >= 0 else "badge-rose"

    f_price = telemetry.get("forecast_price_1d", 83832.87)
    exp_ret = telemetry.get("expected_return_1d_pct", 0.16)
    f_badge = "badge-emerald" if exp_ret >= 0 else "badge-rose"

    vol_pct = telemetry.get("volatility_forecast_annualized_pct", 47.3)
    vol_reg = telemetry.get("volatility_regime", "LOW")
    v_badge = "badge-emerald" if vol_reg == "LOW" else ("badge-amber" if vol_reg == "MEDIUM" else "badge-rose")

    crash_pct = telemetry.get("model_estimated_crash_risk_pct", 27.5)
    crash_lvl = telemetry.get("crash_risk_level", "MODERATE")
    c_badge = "badge-emerald" if crash_lvl == "LOW" else ("badge-amber" if crash_lvl == "MODERATE" else "badge-rose")

    regime = telemetry.get("market_regime", "BULL_MOMENTUM")
    r_badge = "badge-emerald" if "BULL" in regime else ("badge-rose" if "BEAR" in regime else "badge-cyan")

    conf = telemetry.get("qualitative_confidence", "HIGH")
    score = telemetry.get("confidence_score", 3)
    q_badge = "badge-emerald" if conf == "HIGH" else ("badge-amber" if conf == "MEDIUM" else "badge-rose")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-top-row">
                    <div class="kpi-label">💰 Spot Bitcoin Price</div>
                    <span class="kpi-badge-pill {p_badge}">{price_change:+.2f}% (24h)</span>
                </div>
                <div class="kpi-value-text">${curr_price:,.2f}</div>
                <div style="font-size: 11px; color: #94a3b8;">Latest settled close on exchange</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-top-row">
                    <div class="kpi-label">🎯 1-Day Price Target</div>
                    <span class="kpi-badge-pill {f_badge}">{exp_ret:+.2f}% Expected Return</span>
                </div>
                <div class="kpi-value-text">${f_price:,.2f}</div>
                <div style="font-size: 11px; color: #94a3b8;">XGBoost return-compounded point forecast</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-top-row">
                    <div class="kpi-label">🌊 GARCH(1,1) Volatility</div>
                    <span class="kpi-badge-pill {v_badge}">{vol_reg} Volatility</span>
                </div>
                <div class="kpi-value-text">{vol_pct:.1f}%</div>
                <div style="font-size: 11px; color: #94a3b8;">Conditional annualized variance forecast</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-top-row">
                    <div class="kpi-label">⚠️ 14d Crash Risk Probability</div>
                    <span class="kpi-badge-pill {c_badge}">{crash_lvl} Tier</span>
                </div>
                <div class="kpi-value-text">{crash_pct:.1f}%</div>
                <div style="font-size: 11px; color: #94a3b8;">Calibrated chance of &ge;10% drop in 14 days</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-top-row">
                    <div class="kpi-label">🧭 Macro Market Regime</div>
                    <span class="kpi-badge-pill {r_badge}">Rule-Based State</span>
                </div>
                <div class="kpi-value-text" style="font-size: 22px;">{regime.replace('_', ' ')}</div>
                <div style="font-size: 11px; color: #94a3b8;">Multi-factor trend & momentum alignment</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col6:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-top-row">
                    <div class="kpi-label">🛡️ Model Confidence</div>
                    <span class="kpi-badge-pill {q_badge}">{score}/4 Factors</span>
                </div>
                <div class="kpi-value-text">{conf}</div>
                <div style="font-size: 11px; color: #94a3b8;">Qualitative consensus & stability score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# 6. LANDING PAGE VIEW
# -----------------------------------------------------------------------------
def render_landing_page(market_df: pd.DataFrame, telemetry: Dict[str, Any]):
    """Renders the executive, high-converting portfolio landing page."""
    # Top Split Hero Section
    hero_l, hero_r = st.columns([1.15, 0.85], gap="large")

    with hero_l:
        st.markdown(
            f"""
            <div style="padding-top: 10px;">
                <div class="hero-badge-pill">
                    <span class="pulse-green"></span> LIVE QUANTITATIVE ENGINE • ZERO DATA LEAKAGE
                </div>
                <div class="hero-title-large">
                    BTC Sentinel
                </div>
                <div class="hero-tagline-code">
                    &lt; Regime-Aware AI & ML • Crash Early-Warning Engine /&gt;
                </div>
                <div class="hero-description-text">
                    Explainable, econometrically grounded Bitcoin forecasting and severe drawdown prediction. Engineered with GARCH(1,1) volatility clustering, calibrated classification ensembles, and zero-leakage walk-forward validation.
                </div>
                <div class="hero-stats-flex">
                    <div class="hero-stat-box">
                        <div class="hero-stat-box-val">${telemetry['forecast_price_1d']:,.0f}</div>
                        <div class="hero-stat-box-lbl">1-Day Target</div>
                    </div>
                    <div class="hero-stat-box">
                        <div class="hero-stat-box-val">{telemetry['volatility_forecast_annualized_pct']:.1f}%</div>
                        <div class="hero-stat-box-lbl">GARCH Vol</div>
                    </div>
                    <div class="hero-stat-box">
                        <div class="hero-stat-box-val">{telemetry['model_estimated_crash_risk_pct']:.1f}%</div>
                        <div class="hero-stat-box-lbl">Crash Hazard</div>
                    </div>
                    <div class="hero-stat-box">
                        <div class="hero-stat-box-val">5 Folds</div>
                        <div class="hero-stat-box-lbl">Walk-Forward</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Primary Call to Action Buttons with on_click callbacks
        btn_c1, btn_c2 = st.columns([1.3, 1])
        with btn_c1:
            st.button(
                "🚀 Launch Analytics Terminal ➔",
                on_click=set_view,
                args=("dashboard",),
                type="primary",
                use_container_width=True,
                key="landing_hero_launch_btn",
            )
        with btn_c2:
            st.markdown(
                """
                <a href="#architecture-pillars" style="text-decoration: none;">
                    <div style="background: rgba(255, 255, 255, 0.06); border: 1px solid rgba(255, 255, 255, 0.15); color: #f8fafc; font-weight: 700; font-size: 13.5px; padding: 10px 18px; border-radius: 30px; text-align: center;">
                        View Architecture 📖
                    </div>
                </a>
                """,
                unsafe_allow_html=True,
            )

    with hero_r:
        st.markdown(
            f"""
            <div class="terminal-box">
                <div class="terminal-topbar">
                    <div class="terminal-circles">
                        <div class="t-circle t-c-red"></div>
                        <div class="t-circle t-c-yellow"></div>
                        <div class="t-circle t-c-green"></div>
                    </div>
                    <div style="font-size: 11px; color: #94a3b8;">sentinel_telemetry.json</div>
                    <div style="font-size: 10px; color: #60a5fa; background: rgba(59,130,246,0.15); padding: 2px 6px; border-radius: 4px;">JSON</div>
                </div>
                <div style="padding: 16px 20px; font-size: 12.5px; line-height: 1.7; color: #e2e8f0;">
                    <div>&#123;</div>
                    <div style="padding-left: 14px;"><span style="color: #60a5fa;">"asset"</span>: <span style="color: #34d399;">"BTC-USD"</span>,</div>
                    <div style="padding-left: 14px;"><span style="color: #60a5fa;">"current_price"</span>: <span style="color: #f472b6;">${telemetry['current_price']:,.2f}</span>,</div>
                    <div style="padding-left: 14px;"><span style="color: #60a5fa;">"forecast_1d"</span>: <span style="color: #f472b6;">${telemetry['forecast_price_1d']:,.2f}</span>,</div>
                    <div style="padding-left: 14px;"><span style="color: #60a5fa;">"expected_return_1d"</span>: <span style="color: #34d399;">"{telemetry['expected_return_1d_pct']:+.2f}%"</span>,</div>
                    <div style="padding-left: 14px;"><span style="color: #60a5fa;">"market_regime"</span>: <span style="color: #fbbf24;">"{telemetry['market_regime']}"</span>,</div>
                    <div style="padding-left: 14px;"><span style="color: #60a5fa;">"garch_volatility"</span>: <span style="color: #f472b6;">{telemetry['volatility_forecast_annualized_pct']:.1f}%</span>,</div>
                    <div style="padding-left: 14px;"><span style="color: #60a5fa;">"crash_hazard_14d"</span>: <span style="color: #f87171;">{telemetry['model_estimated_crash_risk_pct']:.1f}%</span>,</div>
                    <div style="padding-left: 14px;"><span style="color: #60a5fa;">"qualitative_confidence"</span>: <span style="color: #c084fc;">"{telemetry['qualitative_confidence']}"</span>,</div>
                    <div style="padding-left: 14px;"><span style="color: #60a5fa;">"zero_data_leakage"</span>: <span style="color: #34d399;">true</span></div>
                    <div>&#125;</div>
                    <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.08); color: #f472b6; font-weight: 600;">
                        $ python scripts/predict.py --sentinel
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)

    # Market Pulse Section
    st.markdown("### 📊 Live Quantitative Market Barometer")
    render_kpi_barometer_grid(market_df, telemetry)

    st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

    # Real-Time Regime & Momentum Synthesis
    col_syn1, col_syn2 = st.columns(2)
    with col_syn1:
        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 3px solid #38bdf8; min-height: 140px;">
                <div style="font-size: 11px; font-weight: 700; color: #60a5fa; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                    🎯 Real-Time Regime & Momentum Synthesis (As of {telemetry['as_of_date']})
                </div>
                <div style="font-size: 14.5px; font-weight: 600; color: #ffffff; margin-bottom: 6px;">
                    {telemetry['regime_explanation']}
                </div>
                <div style="font-size: 12.5px; color: #94a3b8; line-height: 1.5;">
                    {telemetry['confidence_rationale']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_syn2:
        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 3px solid #f59e0b; min-height: 140px;">
                <div style="font-size: 11px; font-weight: 700; color: #fbbf24; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                    ⚠️ Drawdown Early-Warning Protocol
                </div>
                <div style="font-size: 14.5px; font-weight: 600; color: #ffffff; margin-bottom: 6px;">
                    Estimated 14-Day Crash Probability: {telemetry['model_estimated_crash_risk_pct']:.1f}% ({telemetry['crash_risk_level']} Risk Tier)
                </div>
                <div style="font-size: 12.5px; color: #94a3b8; line-height: 1.5;">
                    {telemetry['crash_qualification']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div id='architecture-pillars'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # 6 Core Quantitative Pillars Grid (3x2 Balanced Layout)
    st.markdown("### 🏛️ Core Research Pillars & Engineering Architecture")
    p_col1, p_col2, p_col3 = st.columns(3)

    with p_col1:
        st.markdown(
            """
            <div class="pillar-card" style="border-top-color: #60a5fa;">
                <div class="pillar-icon">📈</div>
                <div class="pillar-title">1. Stationarity-Preserving Forecasting</div>
                <div class="pillar-desc">
                    Raw non-stationary prices are converted to 1-day percentage returns to satisfy ADF stationarity, projected via XGBoost, and compounded to price levels with 95% walk-forward empirical error bands.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p_col2:
        st.markdown(
            """
            <div class="pillar-card" style="border-top-color: #34d399;">
                <div class="pillar-icon">🌊</div>
                <div class="pillar-title">2. GARCH(1,1) Volatility Dynamics</div>
                <div class="pillar-desc">
                    Captures conditional heteroskedasticity and volatility persistence via Maximum Likelihood Estimation, benchmarking against realized Parkinson range volatility with empirical quantile cutoffs.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p_col3:
        st.markdown(
            """
            <div class="pillar-card" style="border-top-color: #f43f5e;">
                <div class="pillar-icon">⚠️</div>
                <div class="pillar-title">3. Drawdown Early-Warning Radar</div>
                <div class="pillar-desc">
                    Evaluates forward 14-day &ge;10% drawdown hazard using probability-calibrated Random Forest & Logistic Regression classifiers, validated via out-of-sample ROC-AUC and Precision-Recall metrics.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    p_col4, p_col5, p_col6 = st.columns(3)

    with p_col4:
        st.markdown(
            """
            <div class="pillar-card" style="border-top-color: #c084fc;">
                <div class="pillar-icon">🔍</div>
                <div class="pillar-title">4. Historical Pattern Similarity</div>
                <div class="pillar-desc">
                    Identifies historical analog cycles by comparing the current 30-day normalized price trajectory against all historical epochs using normalized Euclidean distance and Pearson correlation shape matching.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p_col5:
        st.markdown(
            """
            <div class="pillar-card" style="border-top-color: #fbbf24;">
                <div class="pillar-icon">🔄</div>
                <div class="pillar-title">5. Zero-Leakage Walk-Forward Backtest</div>
                <div class="pillar-desc">
                    5 expanding-window chronological folds spanning 450 out-of-sample days evaluate real-world forecasting efficacy against ARIMA(1,1,0) and Naive Persistence benchmarks.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p_col6:
        st.markdown(
            """
            <div class="pillar-card" style="border-top-color: #38bdf8;">
                <div class="pillar-icon">🧠</div>
                <div class="pillar-title">6. Transparent Attribution & Governance</div>
                <div class="pillar-desc">
                    Factual risk attribution breakdown and SHAP feature importance explain exactly why models output specific risk scores, eliminating black-box opacity and detailing scientific bounds.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # End-to-End Pipeline Architecture Flow
    st.markdown("### ⚙️ End-to-End Quantitative Data Pipeline")
    pipe1, pipe2, pipe3, pipe4, pipe5 = st.columns(5)

    with pipe1:
        st.markdown(
            """
            <div class="pipeline-step-box">
                <div class="pipeline-step-badge">Step 1</div>
                <div class="pipeline-step-heading">Market Ingestion</div>
                <div class="pipeline-step-detail">Daily OHLCV series ingestion with validation against missing timestamps.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with pipe2:
        st.markdown(
            """
            <div class="pipeline-step-box">
                <div class="pipeline-step-badge">Step 2</div>
                <div class="pipeline-step-heading">Feature Engineering</div>
                <div class="pipeline-step-detail">Stationary returns, Parkinson volatility, SMAs, RSI, and drawdown features.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with pipe3:
        st.markdown(
            """
            <div class="pipeline-step-box">
                <div class="pipeline-step-badge">Step 3</div>
                <div class="pipeline-step-heading">Model Estimation</div>
                <div class="pipeline-step-detail">XGBoost return projections + GARCH(1,1) MLE volatility fitting.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with pipe4:
        st.markdown(
            """
            <div class="pipeline-step-box">
                <div class="pipeline-step-badge">Step 4</div>
                <div class="pipeline-step-heading">Hazard Classification</div>
                <div class="pipeline-step-detail">Calibrated Random Forest early-warning radar for 14d &ge;10% drawdowns.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with pipe5:
        st.markdown(
            """
            <div class="pipeline-step-box">
                <div class="pipeline-step-badge">Step 5</div>
                <div class="pipeline-step-heading">Synthesis & Audit</div>
                <div class="pipeline-step-detail">Factual attributions, confidence scoring, and telemetry generation.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)

    # Bottom Call to Action Banner
    cta_c1, cta_c2, cta_c3 = st.columns([1, 2, 1])
    with cta_c2:
        st.button(
            "🚀 Launch Interactive Analytics Terminal ➔",
            on_click=set_view,
            args=("dashboard",),
            type="primary",
            use_container_width=True,
            key="landing_bottom_launch_btn",
        )


# -----------------------------------------------------------------------------
# 7. DASHBOARD TERMINAL VIEW (7 MULTI-WORKSTATION TABS)
# -----------------------------------------------------------------------------
def render_dashboard(
    market_df: pd.DataFrame,
    telemetry: Dict[str, Any],
    volatility_data: Optional[Dict],
    crash_data: Optional[Dict],
    similarity_matches: Optional[pd.DataFrame],
    backtest_data: Optional[Dict],
    explainability_data: Optional[Dict],
):
    """Renders the comprehensive quantitative analytics dashboard."""
    # Top Action Bar inside Dashboard
    top_c1, top_c2 = st.columns([3, 1])
    with top_c1:
        st.markdown(
            f"""
            <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 24px; font-weight: 800; color: #ffffff;">
                🛡️ Pro Quantitative Analytics Workstation
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_c2:
        st.button(
            "🏠 ← Return to Landing Page",
            on_click=set_view,
            args=("landing",),
            type="secondary",
            use_container_width=True,
            key="dash_return_home_btn",
        )

    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    # 6-Card KPI Barometer
    render_kpi_barometer_grid(market_df, telemetry)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Multi-Workstation Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Price Forecast & Bands",
        "🌊 Volatility & GARCH",
        "⚠️ Crash Early-Warning",
        "🔍 Historical Similarity",
        "🔄 Walk-Forward Backtest",
        "🧠 Explainability & Bounds",
        "📥 Artifact Export",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: PRICE FORECAST & CONFIDENCE INTERVALS
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("Bitcoin 1-Day Price Forecast with 95% Statistical Confidence Bands")
        col_main, col_sidebar = st.columns([7, 3], gap="large")

        with col_main:
            lookback_days = st.slider("Historical Horizon Lookback (Days)", min_value=30, max_value=365, value=120, step=15, key="dash_lookback")
            chart_slice = market_df.iloc[-lookback_days:].copy()

            fig = go.Figure()

            # Historical Close Price
            fig.add_trace(go.Scatter(
                x=chart_slice["Date"],
                y=chart_slice["Close"],
                name="Historical Close",
                line=dict(color="#38bdf8", width=2.2),
                fill="tozeroy",
                fillcolor="rgba(56, 189, 248, 0.05)",
            ))

            # 50-day SMA
            if len(chart_slice) >= 50:
                sma_50 = market_df["Close"].rolling(50).mean().iloc[-lookback_days:]
                fig.add_trace(go.Scatter(
                    x=chart_slice["Date"],
                    y=sma_50,
                    name="SMA 50",
                    line=dict(color="#fbbf24", width=1.4, dash="dot"),
                ))

            # Forecast point and 95% interval
            last_date = chart_slice["Date"].iloc[-1]
            next_date = last_date + pd.Timedelta(days=1)
            last_price = chart_slice["Close"].iloc[-1]
            f_price = telemetry["forecast_price_1d"]
            low_95, high_95 = telemetry["forecast_interval_95"]

            # Connecting dashed line
            fig.add_trace(go.Scatter(
                x=[last_date, next_date],
                y=[last_price, f_price],
                name="1-Day Forecast (XGB)",
                line=dict(color="#34d399", width=2.5, dash="dash"),
            ))

            # Forecast marker
            fig.add_trace(go.Scatter(
                x=[next_date],
                y=[f_price],
                mode="markers",
                name="Forecast Target",
                marker=dict(color="#34d399", size=11, symbol="diamond", line=dict(color="#ffffff", width=1.5)),
            ))

            # 95% Confidence Interval Band (Cone)
            fig.add_trace(go.Scatter(
                x=[last_date, next_date, next_date, last_date],
                y=[last_price, high_95, low_95, last_price],
                fill="toself",
                fillcolor="rgba(52, 211, 153, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="95% Forecast Interval",
            ))

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d1424",
                plot_bgcolor="#0d1424",
                height=450,
                margin=dict(l=40, r=40, t=20, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", showgrid=True),
                yaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", showgrid=True, title="USD ($)"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_sidebar:
            st.markdown("#### Forecast Summary")
            st.markdown(
                f"""
                <div class="glass-card">
                    <div style="font-size: 12px; color: #94a3b8;">AS OF DATE</div>
                    <div style="font-size: 14px; font-weight: 700; color: #38bdf8;">{telemetry['as_of_date']}</div>
                    <hr style="border-color: rgba(255,255,255,0.08); margin: 8px 0;"/>
                    <div style="font-size: 12px; color: #94a3b8;">CURRENT SPOT</div>
                    <div style="font-size: 20px; font-weight: 800; color: #ffffff;">${telemetry['current_price']:,.2f}</div>
                    <hr style="border-color: rgba(255,255,255,0.08); margin: 8px 0;"/>
                    <div style="font-size: 12px; color: #94a3b8;">1-DAY FORECAST TARGET</div>
                    <div style="font-size: 22px; font-weight: 800; color: #34d399;">${telemetry['forecast_price_1d']:,.2f}</div>
                    <div style="font-size: 12px; color: #34d399; font-weight: 600;">{telemetry['expected_return_1d_pct']:+.2f}% Expected Return</div>
                    <hr style="border-color: rgba(255,255,255,0.08); margin: 8px 0;"/>
                    <div style="font-size: 12px; color: #94a3b8;">95% EMPIRICAL RANGE</div>
                    <div style="font-size: 13.5px; font-weight: 700; color: #fbbf24;">${low_95:,.2f} &ndash; ${high_95:,.2f}</div>
                    <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">Std Error: &plusmn;${telemetry['forecast_interval_error_std']:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # TAB 2: VOLATILITY & GARCH
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("Bitcoin Volatility Dynamics & Statistical Regime Classification")

        v_col1, v_col2 = st.columns([7, 3], gap="large")
        with v_col1:
            vol_df = load_csv_artifact("volatility_forecasts.csv")
            if vol_df is not None:
                vol_slice = vol_df.iloc[-180:].copy()
                vol_slice["Date"] = pd.to_datetime(vol_slice["Date"])

                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(
                    x=vol_slice["Date"],
                    y=vol_slice["Realized_Parkinson_Vol"] * 100.0,
                    name="Realized Parkinson Vol",
                    line=dict(color="#94a3b8", width=1.2, dash="dot"),
                ))
                fig_vol.add_trace(go.Scatter(
                    x=vol_slice["Date"],
                    y=vol_slice["Rolling_30d_Vol"] * 100.0,
                    name="30d Rolling Volatility",
                    line=dict(color="#60a5fa", width=1.6),
                ))
                fig_vol.add_trace(go.Scatter(
                    x=vol_slice["Date"],
                    y=vol_slice["GARCH_Conditional_Vol"] * 100.0,
                    name="GARCH(1,1) Conditional Vol",
                    line=dict(color="#f43f5e", width=2.2),
                ))

                if volatility_data:
                    low_th = volatility_data["regime_thresholds_annualized"]["low_cutoff"] * 100.0
                    high_th = volatility_data["regime_thresholds_annualized"]["high_cutoff"] * 100.0
                    fig_vol.add_hline(y=low_th, line_dash="dash", line_color="#34d399", annotation_text=f"Low Cutoff ({low_th:.1f}%)")
                    fig_vol.add_hline(y=high_th, line_dash="dash", line_color="#fbbf24", annotation_text=f"High Cutoff ({high_th:.1f}%)")

                fig_vol.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0d1424",
                    plot_bgcolor="#0d1424",
                    height=420,
                    margin=dict(l=40, r=40, t=20, b=40),
                    yaxis=dict(title="Annualized Volatility (%)", gridcolor="rgba(255, 255, 255, 0.06)"),
                    xaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)"),
                )
                st.plotly_chart(fig_vol, use_container_width=True)

        with v_col2:
            st.markdown("#### Regime Thresholds & Parameters")
            if volatility_data:
                v_params = volatility_data["garch_parameters"]
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div style="font-size: 13px; color: #cbd5e1; margin-bottom: 6px;"><b>GARCH &alpha; (shock):</b> <code class="code-font" style="color: #34d399;">{v_params['alpha']:.4f}</code></div>
                        <div style="font-size: 13px; color: #cbd5e1; margin-bottom: 6px;"><b>GARCH &beta; (memory):</b> <code class="code-font" style="color: #fbbf24;">{v_params['beta']:.4f}</code></div>
                        <div style="font-size: 13px; color: #cbd5e1; margin-bottom: 6px;"><b>Persistence (&alpha;+&beta;):</b> <code class="code-font" style="color: #ec4899;">{v_params['persistence']:.4f}</code></div>
                        <hr style="border-color: rgba(255,255,255,0.08); margin: 8px 0;"/>
                        <div style="font-size: 12.5px; color: #94a3b8;"><b>Low / Med Cutoff:</b> {low_th:.1f}%</div>
                        <div style="font-size: 12.5px; color: #94a3b8;"><b>Med / High Cutoff:</b> {high_th:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # -------------------------------------------------------------------------
    # TAB 3: CRASH EARLY-WARNING
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("Crash & Severe Drawdown Early-Warning Radar")
        st.caption("Target definition: Probability that Bitcoin maximum drawdown from current price exceeds -10.0% over the next 14 calendar days.")

        c_col1, c_col2 = st.columns([4, 6], gap="large")
        with c_col1:
            st.markdown(
                f"""
                <div class="kpi-card" style="padding: 24px; text-align: center;">
                    <div class="kpi-label" style="justify-content: center;">Model-Estimated Crash Risk</div>
                    <div class="kpi-value-text" style="font-size: 44px; color: {'#34d399' if telemetry['crash_risk_level']=='LOW' else ('#fbbf24' if telemetry['crash_risk_level']=='MODERATE' else '#f87171')}; margin: 10px 0;">
                        {telemetry['model_estimated_crash_risk_pct']:.1f}%
                    </div>
                    <div class="kpi-badge-pill {'badge-emerald' if telemetry['crash_risk_level']=='LOW' else ('badge-amber' if telemetry['crash_risk_level']=='MODERATE' else 'badge-rose')}" style="margin: 0 auto; font-size: 13px;">
                        {telemetry['crash_risk_level']} RISK TIER
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.info(telemetry["crash_qualification"])

        with c_col2:
            if crash_data:
                rf_m = crash_data["model_comparison"]["Random_Forest"]
                lr_m = crash_data["model_comparison"]["Logistic_Regression"]

                comp_df = pd.DataFrame({
                    "Metric": ["ROC-AUC", "PR-AUC (Avg Precision)", "F1-Score", "Recall", "Precision"],
                    "Calibrated Random Forest": [
                        f"{rf_m.get('roc_auc', 0):.3f}",
                        f"{rf_m.get('pr_auc', 0):.3f}",
                        f"{rf_m.get('f1', 0):.3f}",
                        f"{rf_m.get('recall', 0):.3f}",
                        f"{rf_m.get('precision', 0):.3f}",
                    ],
                    "Logistic Regression": [
                        f"{lr_m.get('roc_auc', 0):.3f}",
                        f"{lr_m.get('pr_auc', 0):.3f}",
                        f"{lr_m.get('f1', 0):.3f}",
                        f"{lr_m.get('recall', 0):.3f}",
                        f"{lr_m.get('precision', 0):.3f}",
                    ],
                })
                st.markdown("#### Out-of-Sample Crash Detection Performance")
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Top Predictive Features for Crash Warning")
        if crash_data and "top_crash_drivers" in crash_data:
            drivers = crash_data["top_crash_drivers"]
            df_drivers = pd.DataFrame(drivers, columns=["Indicator", "Attribution Weight"])
            fig_imp = go.Figure(go.Bar(
                x=df_drivers["Attribution Weight"].iloc[:10],
                y=df_drivers["Indicator"].iloc[:10],
                orientation="h",
                marker_color="#f43f5e",
            ))
            fig_imp.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d1424",
                plot_bgcolor="#0d1424",
                height=320,
                margin=dict(l=40, r=40, t=10, b=30),
                yaxis=dict(autorange="reversed", gridcolor="rgba(255, 255, 255, 0.06)"),
                xaxis=dict(title="Gini Feature Importance", gridcolor="rgba(255, 255, 255, 0.06)"),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 4: HISTORICAL PATTERN SIMILARITY
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("Historical Pattern Similarity & Subsequent Trajectories")
        st.caption("Compares current 30-day price trajectory against all non-overlapping 30-day historical epochs.")

        sim_col1, sim_col2 = st.columns([1, 2], gap="large")
        with sim_col1:
            st.markdown(
                f"""
                <div class="kpi-card" style="padding: 20px;">
                    <div class="kpi-label">Top Historical Analog</div>
                    <div class="kpi-value-text" style="font-size: 26px;">{telemetry['top_similar_date']}</div>
                    <div class="kpi-badge-pill badge-cyan" style="margin-top: 4px;">{telemetry['top_similarity_score_pct']:.1f}% SHAPE SIMILARITY</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("⚠️ **Research Disclaimer:** Historical similarity indicates analogous momentum/range configurations, NOT a guarantee of identical subsequent performance.")

        with sim_col2:
            if similarity_matches is not None:
                display_matches = similarity_matches[[
                    "rank", "match_end_date", "similarity_score_pct",
                    "subsequent_return_7d_pct", "subsequent_return_14d_pct", "subsequent_return_30d_pct"
                ]].copy()
                display_matches.columns = [
                    "Rank", "Historical Period End", "Similarity (%)",
                    "+7d Return (%)", "+14d Return (%)", "+30d Return (%)"
                ]
                st.dataframe(display_matches, use_container_width=True, hide_index=True)

        fig_path = Path("reports/figures/pattern_similarity_overlay.png")
        if fig_path.exists():
            st.image(str(fig_path), caption="Historical Pattern Similarity Trajectory Overlay", use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 5: WALK-FORWARD BACKTESTING
    # -------------------------------------------------------------------------
    with tab5:
        st.subheader("Chronological Walk-Forward Backtesting (Zero Data Leakage)")
        st.caption("5 sequential expanding-window out-of-sample evaluation folds across 450 out-of-sample days.")

        if backtest_data:
            models = backtest_data["models"]
            b_summary_df = pd.DataFrame([
                {
                    "Model": "Naive Persistence Benchmark",
                    "MAE ($)": f"${models['Naive_Persistence']['mae']:,.2f}",
                    "RMSE ($)": f"${models['Naive_Persistence']['rmse']:,.2f}",
                    "MAPE (%)": f"{models['Naive_Persistence']['mape']:.2f}%",
                    "Directional Accuracy (%)": f"{models['Naive_Persistence'].get('directional_accuracy', 0):.1f}%",
                },
                {
                    "Model": "ARIMA(1, 1, 0)",
                    "MAE ($)": f"${models['ARIMA']['mae']:,.2f}",
                    "RMSE ($)": f"${models['ARIMA']['rmse']:,.2f}",
                    "MAPE (%)": f"{models['ARIMA']['mape']:.2f}%",
                    "Directional Accuracy (%)": f"{models['ARIMA'].get('directional_accuracy', 0):.1f}%",
                },
                {
                    "Model": "XGBoost (Return-Compounded)",
                    "MAE ($)": f"${models['XGBoost']['price_mae']:,.2f}",
                    "RMSE ($)": f"${models['XGBoost']['price_rmse']:,.2f}",
                    "MAPE (%)": f"{models['XGBoost']['price_mape']:.2f}%",
                    "Directional Accuracy (%)": f"{models['XGBoost'].get('directional_accuracy', 0):.1f}%",
                },
            ])
            st.dataframe(b_summary_df, use_container_width=True, hide_index=True)

        wf_img = Path("reports/figures/walk_forward_comparison.png")
        cum_img = Path("reports/figures/cumulative_absolute_error.png")
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            if wf_img.exists():
                st.image(str(wf_img), caption="Out-Of-Sample Walk-Forward Predictions vs Actual Price", use_container_width=True)
        with col_img2:
            if cum_img.exists():
                st.image(str(cum_img), caption="Cumulative Absolute Forecast Error Drift Over Time", use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 6: EXPLAINABILITY & METHODOLOGY
    # -------------------------------------------------------------------------
    with tab6:
        st.subheader("Explainable Model Attributions & Scientific Foundations")

        if explainability_data and "current_risk_driver_breakdown" in explainability_data:
            st.markdown("#### Factual Risk Attributions (Current State vs Historical Percentiles)")
            drivers = explainability_data["current_risk_driver_breakdown"]["top_drivers"]
            for d in drivers:
                st.markdown(f"• **{d['feature']}:** {d['factual_statement']}")

        st.markdown("---")
        st.markdown(
            """
            ### Research Foundations & Engineering Methodology
            
            1. **Stationarity & Target Transformation:**
               - Raw BTC price series fail Augmented Dickey-Fuller (ADF) tests ($p > 0.05$).
               - Machine learning models forecast stationary 1-day percentage returns $R_{t+1}$, which are mathematically compounded to reconstruct forward price levels: $\\hat{P}_{t+1} = P_t \\times (1 + \\hat{R}_{t+1})$.
            
            2. **Volatility Modeling (GARCH vs Machine Learning):**
               - Conditional variance exhibits heavy clustering in cryptocurrency markets.
               - GARCH(1,1) is fitted via Maximum Likelihood Estimation with stationary bounds ($\\alpha + \\beta < 1$).
               - Evaluated against realized Parkinson range volatility using Quasi-Likelihood (QLIKE) and RMSE.
            
            3. **Crash Early-Warning & Epsilon Drawdowns:**
               - Forward drawdowns are defined over a 14-day horizon: $\\min_{1 \\le k \\le 14} (P_{t+k} - P_t) / P_t \\le -10\\%$.
               - Calibrated Random Forest and balanced Logistic Regression yield verified probabilities without uncalibrated overconfidence.
            
            4. **Known Scientific Limitations:**
               - **Black Swan Events:** Quantitative models extrapolate from historical distributions and cannot anticipate exogenous regulatory shocks, exchange insolvencies, or macroeconomic regime breaks.
               - **Execution Friction:** Forecasts assume frictionless market conditions and do not account for slippage or exchange fees.
            """
        )

    # -------------------------------------------------------------------------
    # TAB 7: ARTIFACT EXPORT CENTER
    # -------------------------------------------------------------------------
    with tab7:
        st.subheader("📥 Quantitative Artifact Export Center")
        st.caption("Download certified model telemetry, out-of-sample backtest predictions, and feature importance matrices.")

        ex_c1, ex_c2, ex_c3 = st.columns(3)
        with ex_c1:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="glass-card-title">📄 Sentinel Telemetry (JSON)</div>
                    <div style="font-size: 12px; color: #94a3b8; margin-bottom: 14px;">Complete executive telemetry output, regime states, confidence scores, and projections.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.download_button(
                label="Download Telemetry JSON",
                data=json.dumps(telemetry, indent=2),
                file_name="btc_sentinel_telemetry.json",
                mime="application/json",
                use_container_width=True,
                key="dash_dl_json",
            )

        with ex_c2:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="glass-card-title">📊 Backtest Predictions (CSV)</div>
                    <div style="font-size: 12px; color: #94a3b8; margin-bottom: 14px;">450 out-of-sample walk-forward predictions with ARIMA and XGBoost errors.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            wf_df = load_csv_artifact("walk_forward_backtest_predictions.csv")
            if wf_df is not None:
                st.download_button(
                    label="Download Backtest CSV",
                    data=wf_df.to_csv(index=False),
                    file_name="walk_forward_backtest_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dash_dl_backtest",
                )

        with ex_c3:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="glass-card-title">🌊 Volatility Series (CSV)</div>
                    <div style="font-size: 12px; color: #94a3b8; margin-bottom: 14px;">Historical GARCH conditional volatility vs Parkinson realized range volatility.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            vol_df = load_csv_artifact("volatility_forecasts.csv")
            if vol_df is not None:
                st.download_button(
                    label="Download Volatility CSV",
                    data=vol_df.to_csv(index=False),
                    file_name="volatility_forecasts.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dash_dl_vol",
                )


# -----------------------------------------------------------------------------
# 8. MAIN CONTROLLER & ROUTING
# -----------------------------------------------------------------------------
def main():
    market_df = load_market_data()
    telemetry = load_json_artifact("sentinel_inference_output.json")
    backtest_data = load_json_artifact("walk_forward_backtest_results.json")
    volatility_data = load_json_artifact("volatility_results.json")
    crash_data = load_json_artifact("crash_risk_results.json")
    similarity_matches = load_csv_artifact("pattern_similarity_matches.csv")
    explainability_data = load_json_artifact("explainability_report.json")

    # Fallback alert if pipeline has not been executed
    if market_df is None or telemetry is None:
        st.error(
            "⚠️ Pre-computed model telemetry or processed dataset is missing.\n\n"
            "Please execute the complete automated training pipeline:\n"
            "```bash\npython scripts/train.py\n```"
        )
        return

    # 1. Render Persistent Global Navigation Header
    render_header(telemetry)

    # 2. Render Live Market Ticker Ribbon
    render_market_ticker(market_df, telemetry)

    # 3. Route Views strictly based on session state: "landing" vs "dashboard"
    current_view = st.session_state.get("app_view", "landing")
    if current_view == "landing":
        render_landing_page(market_df, telemetry)
    else:
        render_dashboard(
            market_df=market_df,
            telemetry=telemetry,
            volatility_data=volatility_data,
            crash_data=crash_data,
            similarity_matches=similarity_matches,
            backtest_data=backtest_data,
            explainability_data=explainability_data,
        )


if __name__ == "__main__":
    main()
