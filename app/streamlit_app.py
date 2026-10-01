"""BTC Sentinel - Streamlit Research & Quantitative Analytics Dashboard.
Explainable, Regime-Aware Bitcoin Forecasting & Crash Early-Warning System.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Configure page
st.set_page_config(
    page_title="BTC Sentinel | Quantitative Bitcoin Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize current_page state (Defaults to "landing" so first open is Landing Page)
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "landing"

# Custom Typography & Dark Cyberpunk / Portfolio Theme Styling
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

    <style>
    /* Global Obsidian Dark Theme & Typography */
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700;
        letter-spacing: -0.4px;
        color: #f8fafc !important;
    }

    .code-font {
        font-family: 'Fira Code', monospace !important;
    }

    /* Layout Container Spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1350px !important;
    }

    /* Top Navigation Bar */
    .top-nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(145deg, #131b2e 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 12px 20px;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.3);
    }
    .top-nav-logo {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 19px;
        font-weight: 800;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Hero Banner Container */
    .hero-wrapper {
        background: linear-gradient(145deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 40px 45px;
        margin-bottom: 28px;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6), 0 0 30px rgba(139, 92, 246, 0.1);
        position: relative;
        overflow: hidden;
    }

    .hero-top-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-family: 'Fira Code', monospace;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        background: rgba(236, 72, 153, 0.15);
        color: #f472b6;
        padding: 5px 14px;
        border-radius: 25px;
        margin-bottom: 18px;
        border: 1px solid rgba(236, 72, 153, 0.35);
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #ec4899;
        box-shadow: 0 0 10px #ec4899;
    }

    .hero-main-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 46px;
        font-weight: 900;
        letter-spacing: -1.2px;
        line-height: 1.15;
        margin-bottom: 14px;
        background: linear-gradient(135deg, #ffffff 0%, #f472b6 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-tagline {
        font-family: 'Fira Code', monospace;
        font-size: 14px;
        font-weight: 500;
        color: #cbd5e1;
        margin-bottom: 16px;
        letter-spacing: -0.2px;
    }

    .hero-description {
        font-size: 15px;
        color: #94a3b8;
        line-height: 1.65;
        margin-bottom: 26px;
        max-width: 620px;
    }

    /* Hero Stats Grid in Row */
    .hero-stats-row {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 28px;
    }
    .hero-stat-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px 18px;
        min-width: 120px;
        text-align: center;
        backdrop-filter: blur(8px);
    }
    .hero-stat-value {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 20px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 2px;
    }
    .hero-stat-label {
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
    }

    /* Code Terminal Box Mockup */
    .terminal-window {
        background: #090d16;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8), 0 0 30px rgba(236, 72, 153, 0.15);
        overflow: hidden;
        font-family: 'Fira Code', monospace;
    }
    .terminal-header {
        background: #131b2e;
        padding: 10px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .terminal-dots {
        display: flex;
        gap: 6px;
    }
    .t-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }
    .t-dot-red { background: #ef4444; }
    .t-dot-yellow { background: #f59e0b; }
    .t-dot-green { background: #10b981; }
    .terminal-title {
        font-size: 11px;
        color: #94a3b8;
        font-weight: 500;
    }
    .terminal-badge {
        font-size: 10px;
        font-weight: 700;
        color: #60a5fa;
        background: rgba(59, 130, 246, 0.15);
        padding: 2px 6px;
        border-radius: 4px;
    }
    .terminal-body {
        padding: 18px 20px;
        font-size: 12.5px;
        line-height: 1.7;
        color: #e2e8f0;
    }
    .t-line-num {
        color: #475569;
        margin-right: 12px;
        user-select: none;
    }
    .t-key { color: #60a5fa; }
    .t-str { color: #34d399; }
    .t-num { color: #f472b6; }
    .t-bool { color: #fbbf24; }
    .terminal-prompt {
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        color: #f472b6;
        font-weight: 600;
    }
    .terminal-cursor {
        display: inline-block;
        width: 7px;
        height: 14px;
        background: #34d399;
        margin-left: 4px;
        vertical-align: middle;
        animation: blink 1s step-start infinite;
    }
    @keyframes blink { 50% { opacity: 0; } }

    /* Top Header Card for Dashboard */
    .app-header-card {
        background: linear-gradient(145deg, #131b2e 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px -5px rgba(0, 0, 0, 0.4);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .app-header-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 22px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .app-header-subtitle {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 2px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        font-weight: 600;
        color: #34d399;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
    }

    /* KPI Metric Cards */
    .kpi-container {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        background: linear-gradient(145deg, #131b2e 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 18px;
        min-height: 112px;
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }
    .kpi-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.5);
        border-color: rgba(236, 72, 153, 0.3);
    }
    .kpi-title {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .kpi-value {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 24px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 6px;
        line-height: 1.1;
    }
    .kpi-badge {
        font-size: 11px;
        font-weight: 600;
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        width: fit-content;
    }
    .badge-green { background-color: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }
    .badge-blue  { background-color: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.35); }
    .badge-amber { background-color: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }
    .badge-red   { background-color: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.35); }
    
    /* Pillar Feature Card */
    .pillar-card {
        background: linear-gradient(145deg, #131b2e 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 22px 24px;
        min-height: 185px;
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.3);
        border-top: 3px solid #ec4899;
        margin-bottom: 14px;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }
    .pillar-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px -4px rgba(0, 0, 0, 0.5), 0 0 15px rgba(236, 72, 153, 0.15);
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

    /* Factor card */
    .factor-card {
        background: linear-gradient(145deg, #131b2e 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 3px solid #ec4899;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    .factor-title {
        font-size: 11px;
        color: #94a3b8;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .factor-desc {
        font-size: 13px;
        color: #f1f5f9;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Synthesis Callout */
    .synthesis-card {
        background: linear-gradient(145deg, #131b2e 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px 24px;
        min-height: 145px;
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }

    /* Pipeline Step */
    .pipeline-step {
        background: linear-gradient(145deg, #131b2e 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        margin-bottom: 8px;
    }
    .pipeline-step-num {
        font-size: 11px;
        font-weight: 700;
        color: #f472b6;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .pipeline-step-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 13px;
        font-weight: 700;
        color: #ffffff;
        margin: 4px 0px;
    }
    .pipeline-step-desc {
        font-size: 11.5px;
        color: #94a3b8;
        line-height: 1.4;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 4px;
        margin-bottom: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 6px 6px 0px 0px;
        font-size: 13.5px;
        font-weight: 600;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        color: #f472b6 !important;
        border-bottom: 2px solid #ec4899 !important;
    }

    /* Primary Gradient Button Override */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%) !important;
        border: none !important;
        color: white !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        padding: 10px 24px !important;
        border-radius: 30px !important;
        box-shadow: 0 4px 15px rgba(236, 72, 153, 0.35) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(236, 72, 153, 0.6) !important;
    }

    /* Secondary Navigation Button */
    div.stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #f8fafc !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
        padding: 8px 18px !important;
        border-radius: 30px !important;
        transition: background 0.15s ease, border-color 0.15s ease !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.12) !important;
        border-color: rgba(236, 72, 153, 0.4) !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #090d16;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

RESULTS_DIR = Path("reports/results")
DATA_FILE = Path("data/processed/btc_daily.csv")


@st.cache_data(ttl=300)
def load_json_artifact(filename: str) -> Optional[Dict[str, Any]]:
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
    path = RESULTS_DIR / filename
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


@st.cache_data(ttl=300)
def load_market_data() -> Optional[pd.DataFrame]:
    if DATA_FILE.exists():
        try:
            df = pd.read_csv(DATA_FILE)
            df["Date"] = pd.to_datetime(df["Date"])
            return df
        except Exception:
            return None
    return None


def render_kpi_barometer(market_df: pd.DataFrame, telemetry: Dict[str, Any]):
    """Renders a balanced 3x2 grid of institutional KPI metric cards."""
    # Row 1 (Price, 1-Day Forecast, Volatility)
    r1_col1, r1_col2, r1_col3 = st.columns(3)

    # 1. Current Price
    with r1_col1:
        prev_close = market_df["Close"].iloc[-2] if len(market_df) > 1 else telemetry["current_price"]
        price_change = (telemetry["current_price"] - prev_close) / prev_close * 100.0
        badge_class = "badge-green" if price_change >= 0 else "badge-red"
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-title">💰 Current Bitcoin Price</div>
                <div class="kpi-value">${telemetry['current_price']:,.2f}</div>
                <div class="kpi-badge {badge_class}">{price_change:+.2f}% (24h Change)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. 1-Day Forecast
    with r1_col2:
        exp_ret = telemetry["expected_return_1d_pct"]
        f_badge = "badge-green" if exp_ret >= 0 else "badge-red"
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-title">🎯 1-Day Forecast Target (XGBoost)</div>
                <div class="kpi-value">${telemetry['forecast_price_1d']:,.2f}</div>
                <div class="kpi-badge {f_badge}">{exp_ret:+.2f}% Expected Return</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Volatility Forecast
    with r1_col3:
        vol_pct = telemetry["volatility_forecast_annualized_pct"]
        vol_reg = telemetry["volatility_regime"]
        v_badge = "badge-green" if vol_reg == "LOW" else ("badge-amber" if vol_reg == "MEDIUM" else "badge-red")
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-title">🌊 GARCH(1,1) Volatility Forecast</div>
                <div class="kpi-value">{vol_pct:.1f}%</div>
                <div class="kpi-badge {v_badge}">{vol_reg} Volatility Regime</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Row 2 (Crash Risk, Macro Regime, Model Confidence)
    r2_col1, r2_col2, r2_col3 = st.columns(3)

    # 4. Crash Risk
    with r2_col1:
        c_risk = telemetry["model_estimated_crash_risk_pct"]
        c_level = telemetry["crash_risk_level"]
        c_badge = "badge-green" if c_level == "LOW" else ("badge-amber" if c_level == "MODERATE" else "badge-red")
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-title">⚠️ 14-Day Crash & Drawdown Risk</div>
                <div class="kpi-value">{c_risk:.1f}%</div>
                <div class="kpi-badge {c_badge}">{c_level} Hazard Tier (&ge;10% Drawdown)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 5. Market Macro Regime
    with r2_col2:
        m_reg = telemetry["market_regime"]
        r_badge = "badge-green" if "BULL" in m_reg else ("badge-red" if "BEAR" in m_reg else "badge-blue")
        clean_reg = m_reg.replace("_", " ")
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-title">🧭 Macro Market Regime</div>
                <div class="kpi-value" style="font-size: 20px; margin-top: 2px;">{clean_reg}</div>
                <div class="kpi-badge {r_badge}">Rule-Based Multi-Factor State</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 6. Qualitative Model Confidence
    with r2_col3:
        q_conf = telemetry["qualitative_confidence"]
        q_score = telemetry["confidence_score"]
        q_badge = "badge-green" if q_conf == "HIGH" else ("badge-amber" if q_conf == "MEDIUM" else "badge-red")
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-title">🛡️ Qualitative Model Confidence</div>
                <div class="kpi-value">{q_conf}</div>
                <div class="kpi-badge {q_badge}">{q_score}/4 Quantitative Factors Aligned</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_landing_page(market_df: pd.DataFrame, telemetry: Dict[str, Any]):
    """Renders the executive hero landing page with developer-portfolio aesthetics."""
    # Top Hero Section (Split Left Content / Right Terminal)
    hero_col_left, hero_col_right = st.columns([1.15, 0.85], gap="large")

    with hero_col_left:
        st.markdown(
            f"""
            <div style="padding-top: 10px;">
                <div class="hero-top-badge">
                    <span class="pulse-dot"></span> LIVE QUANTITATIVE ENGINE • ZERO DATA LEAKAGE
                </div>
                <div class="hero-main-title">
                    BTC Sentinel
                </div>
                <div class="hero-tagline">
                    &lt; Regime-Aware AI & ML • Crash Early-Warning Engine /&gt;
                </div>
                <div class="hero-description">
                    Explainable, econometrically grounded Bitcoin forecasting and drawdown early-warning system. Engineered with GARCH(1,1) volatility clustering, calibrated classification ensembles, and zero-leakage walk-forward validation.
                </div>
                <div class="hero-stats-row">
                    <div class="hero-stat-card">
                        <div class="hero-stat-value">${telemetry['forecast_price_1d']:,.0f}</div>
                        <div class="hero-stat-label">1-Day Target</div>
                    </div>
                    <div class="hero-stat-card">
                        <div class="hero-stat-value">{telemetry['volatility_forecast_annualized_pct']:.1f}%</div>
                        <div class="hero-stat-label">GARCH Vol</div>
                    </div>
                    <div class="hero-stat-card">
                        <div class="hero-stat-value">{telemetry['model_estimated_crash_risk_pct']:.1f}%</div>
                        <div class="hero-stat-label">Crash Hazard</div>
                    </div>
                    <div class="hero-stat-card">
                        <div class="hero-stat-value">5 Folds</div>
                        <div class="hero-stat-label">Walk-Forward</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Action Buttons
        btn_c1, btn_c2 = st.columns([1.2, 1])
        with btn_c1:
            if st.button("🚀 Explore Live Terminal ➔", type="primary", use_container_width=True, key="landing_btn_explore"):
                st.session_state["current_page"] = "dashboard"
                st.rerun()
        with btn_c2:
            st.markdown(
                """
                <a href="#core-pillars" style="text-decoration: none;">
                    <div style="background: rgba(255, 255, 255, 0.06); border: 1px solid rgba(255, 255, 255, 0.15); color: #f8fafc; font-weight: 700; font-size: 13.5px; padding: 9px 18px; border-radius: 30px; text-align: center;">
                        View Architecture 📖
                    </div>
                </a>
                """,
                unsafe_allow_html=True,
            )

    with hero_col_right:
        st.markdown(
            f"""
            <div class="terminal-window">
                <div class="terminal-header">
                    <div class="terminal-dots">
                        <div class="t-dot t-dot-red"></div>
                        <div class="t-dot t-dot-yellow"></div>
                        <div class="t-dot t-dot-green"></div>
                    </div>
                    <div class="terminal-title">sentinel_telemetry.json</div>
                    <div class="terminal-badge">JSON</div>
                </div>
                <div class="terminal-body">
                    <div><span class="t-line-num">1</span>&#123;</div>
                    <div><span class="t-line-num">2</span>  <span class="t-key">"asset"</span>: <span class="t-str">"BTC-USD"</span>,</div>
                    <div><span class="t-line-num">3</span>  <span class="t-key">"current_price"</span>: <span class="t-num">${telemetry['current_price']:,.2f}</span>,</div>
                    <div><span class="t-line-num">4</span>  <span class="t-key">"forecast_1d"</span>: <span class="t-num">${telemetry['forecast_price_1d']:,.2f}</span>,</div>
                    <div><span class="t-line-num">5</span>  <span class="t-key">"expected_return_1d"</span>: <span class="t-str">"{telemetry['expected_return_1d_pct']:+.2f}%"</span>,</div>
                    <div><span class="t-line-num">6</span>  <span class="t-key">"market_regime"</span>: <span class="t-str">"{telemetry['market_regime']}"</span>,</div>
                    <div><span class="t-line-num">7</span>  <span class="t-key">"garch_volatility"</span>: <span class="t-num">{telemetry['volatility_forecast_annualized_pct']:.1f}%</span>,</div>
                    <div><span class="t-line-num">8</span>  <span class="t-key">"crash_hazard_14d"</span>: <span class="t-num">{telemetry['model_estimated_crash_risk_pct']:.1f}%</span>,</div>
                    <div><span class="t-line-num">9</span>  <span class="t-key">"qualitative_confidence"</span>: <span class="t-str">"{telemetry['qualitative_confidence']}"</span>,</div>
                    <div><span class="t-line-num">10</span> <span class="t-key">"zero_data_leakage"</span>: <span class="t-bool">true</span></div>
                    <div><span class="t-line-num">11</span>&#125;</div>
                    <div class="terminal-prompt">
                        $ python scripts/predict.py --sentinel<span class="terminal-cursor"></span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)

    # Market Pulse Section
    st.markdown("### 📊 Live Quantitative Barometer")
    render_kpi_barometer(market_df, telemetry)

    st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

    # Executive Summary & Live Rationale (2 Balanced Columns)
    col_syn1, col_syn2 = st.columns(2)
    with col_syn1:
        st.markdown(
            f"""
            <div class="synthesis-card">
                <div style="font-size: 11px; font-weight: 700; color: #60a5fa; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                    🎯 Real-Time Regime & Momentum Synthesis (As of {telemetry['as_of_date']})
                </div>
                <div style="font-size: 15px; font-weight: 600; color: #ffffff; margin-bottom: 6px;">
                    {telemetry['regime_explanation']}
                </div>
                <div style="font-size: 13px; color: #94a3b8; line-height: 1.5;">
                    {telemetry['confidence_rationale']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_syn2:
        st.markdown(
            f"""
            <div class="synthesis-card" style="border-left: 4px solid #f59e0b;">
                <div style="font-size: 11px; font-weight: 700; color: #fbbf24; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                    ⚠️ Drawdown Early-Warning Protocol
                </div>
                <div style="font-size: 15px; font-weight: 600; color: #ffffff; margin-bottom: 6px;">
                    Estimated 14-Day Crash Probability: {telemetry['model_estimated_crash_risk_pct']:.1f}% ({telemetry['crash_risk_level']} Risk Tier)
                </div>
                <div style="font-size: 13px; color: #94a3b8; line-height: 1.5;">
                    {telemetry['crash_qualification']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div id='core-pillars'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # 6 Core Quantitative Pillars Grid (3x2 Balanced Layout)
    st.markdown("### 🏛️ Core Research Pillars & Capabilities")
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
            <div class="pipeline-step">
                <div class="pipeline-step-num">Step 1</div>
                <div class="pipeline-step-title">Market Ingestion</div>
                <div class="pipeline-step-desc">Daily OHLCV series ingestion with validation against missing timestamps.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with pipe2:
        st.markdown(
            """
            <div class="pipeline-step">
                <div class="pipeline-step-num">Step 2</div>
                <div class="pipeline-step-title">Feature Engineering</div>
                <div class="pipeline-step-desc">Stationary returns, Parkinson volatility, SMAs, RSI, and drawdown features.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with pipe3:
        st.markdown(
            """
            <div class="pipeline-step">
                <div class="pipeline-step-num">Step 3</div>
                <div class="pipeline-step-title">Model Estimation</div>
                <div class="pipeline-step-desc">XGBoost return projections + GARCH(1,1) MLE volatility fitting.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with pipe4:
        st.markdown(
            """
            <div class="pipeline-step">
                <div class="pipeline-step-num">Step 4</div>
                <div class="pipeline-step-title">Hazard Classification</div>
                <div class="pipeline-step-desc">Calibrated Random Forest early-warning radar for 14d &ge;10% drawdowns.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with pipe5:
        st.markdown(
            """
            <div class="pipeline-step">
                <div class="pipeline-step-num">Step 5</div>
                <div class="pipeline-step-title">Synthesis & Audit</div>
                <div class="pipeline-step-desc">Factual attributions, confidence scoring, and telemetry generation.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 24px; text-align: center;'></div>", unsafe_allow_html=True)
    
    # Bottom Center CTA
    cta_col1, cta_col2, cta_col3 = st.columns([1, 2, 1])
    with cta_col2:
        if st.button("🚀 Launch Interactive Analytics Terminal ➔", type="primary", use_container_width=True, key="landing_btn_bottom"):
            st.session_state["current_page"] = "dashboard"
            st.rerun()


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
    # Top Navigation Row with Back Button
    nav_left, nav_right = st.columns([3, 1])
    with nav_left:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 22px; font-weight: 800; color: #ffffff;">
                    🛡️ BTC Sentinel Analytics Dashboard
                </div>
                <div class="status-pill">
                    ● Telemetry Verified (As of {telemetry['as_of_date']})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with nav_right:
        if st.button("🏠 ← Back to Landing Page", kind="secondary", use_container_width=True, key="btn_back_home"):
            st.session_state["current_page"] = "landing"
            st.rerun()

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Top KPI Barometer
    render_kpi_barometer(market_df, telemetry)

    st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)

    # Multi-Workstation Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Price Forecast & Bands",
        "🌊 Volatility & Regimes",
        "⚠️ Crash Early-Warning",
        "🔍 Historical Similarity",
        "🔄 Walk-Forward Backtest",
        "🧠 Explainability & Foundations",
    ])

    # -------------------------------------------------------------
    # TAB 1: PRICE FORECAST & CONFIDENCE INTERVALS
    # -------------------------------------------------------------
    with tab1:
        st.subheader("Bitcoin 1-Day Price Forecast with 95% Statistical Confidence Bands")
        col_main, col_sidebar = st.columns([7, 3])

        with col_main:
            lookback_days = st.slider("Historical Horizon Lookback (Days)", min_value=30, max_value=365, value=120, step=15)
            chart_slice = market_df.iloc[-lookback_days:].copy()

            fig = go.Figure()

            # Historical Close Price
            fig.add_trace(go.Scatter(
                x=chart_slice["Date"],
                y=chart_slice["Close"],
                name="Historical Close",
                line=dict(color="#38bdf8", width=2),
            ))

            # 50-day SMA
            if len(chart_slice) >= 50:
                sma_50 = market_df["Close"].rolling(50).mean().iloc[-lookback_days:]
                fig.add_trace(go.Scatter(
                    x=chart_slice["Date"],
                    y=sma_50,
                    name="SMA 50",
                    line=dict(color="#fbbf24", width=1.2, dash="dot"),
                ))

            # Forecast point and 95% interval
            last_date = chart_slice["Date"].iloc[-1]
            next_date = last_date + pd.Timedelta(days=1)
            last_price = chart_slice["Close"].iloc[-1]
            f_price = telemetry["forecast_price_1d"]
            low_95, high_95 = telemetry["forecast_interval_95"]

            # Connecting line
            fig.add_trace(go.Scatter(
                x=[last_date, next_date],
                y=[last_price, f_price],
                name="1-Day Forecast (XGB)",
                line=dict(color="#34d399", width=2.5, dash="dash"),
            ))

            # Forecast point
            fig.add_trace(go.Scatter(
                x=[next_date],
                y=[f_price],
                mode="markers",
                name="Forecast Target",
                marker=dict(color="#34d399", size=10, symbol="diamond"),
            ))

            # 95% Confidence Interval Band (Cone)
            fig.add_trace(go.Scatter(
                x=[last_date, next_date, next_date, last_date],
                y=[last_price, high_95, low_95, last_price],
                fill="toself",
                fillcolor="rgba(52, 211, 153, 0.18)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="95% Forecast Interval",
            ))

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#131b2e",
                plot_bgcolor="#131b2e",
                height=460,
                margin=dict(l=40, r=40, t=20, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(gridcolor="rgba(255, 255, 255, 0.08)", showgrid=True),
                yaxis=dict(gridcolor="rgba(255, 255, 255, 0.08)", showgrid=True, title="USD ($)"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_sidebar:
            st.markdown("#### Forecast Summary")
            st.write(f"**As of Date:** {telemetry['as_of_date']}")
            st.write(f"**Current Price:** `${telemetry['current_price']:,.2f}`")
            st.write(f"**1-Day Forecast:** `${telemetry['forecast_price_1d']:,.2f}`")
            st.write(f"**Expected Return:** `{telemetry['expected_return_1d_pct']:+.2f}%`")
            st.write(f"**95% Range:** `${low_95:,.2f} – ${high_95:,.2f}`")
            st.write(f"**Forecast Std Error:** `±${telemetry['forecast_interval_error_std']:,.2f}`")

            st.markdown("---")
            st.markdown("#### Individual Model Projections")
            for m_name, m_val in telemetry["model_predictions"].items():
                m_label = m_name.replace("_", " ")
                m_diff = (m_val - telemetry["current_price"]) / telemetry["current_price"] * 100.0
                st.write(f"• **{m_label}:** `${m_val:,.2f}` ({m_diff:+.2f}%)")

            st.caption("Intervals derived from out-of-sample walk-forward empirical residual standard errors.")

    # -------------------------------------------------------------
    # TAB 2: VOLATILITY & REGIMES
    # -------------------------------------------------------------
    with tab2:
        st.subheader("Bitcoin Volatility Dynamics & Statistical Regime Classification")

        v_col1, v_col2 = st.columns([7, 3])
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
                    line=dict(color="#94a3b8", width=1, dash="dot"),
                ))
                fig_vol.add_trace(go.Scatter(
                    x=vol_slice["Date"],
                    y=vol_slice["Rolling_30d_Vol"] * 100.0,
                    name="30d Rolling Volatility",
                    line=dict(color="#60a5fa", width=1.5),
                ))
                fig_vol.add_trace(go.Scatter(
                    x=vol_slice["Date"],
                    y=vol_slice["GARCH_Conditional_Vol"] * 100.0,
                    name="GARCH(1,1) Conditional Vol",
                    line=dict(color="#f43f5e", width=2),
                ))

                # Regime cutoffs
                if volatility_data:
                    low_th = volatility_data["regime_thresholds_annualized"]["low_cutoff"] * 100.0
                    high_th = volatility_data["regime_thresholds_annualized"]["high_cutoff"] * 100.0
                    fig_vol.add_hline(y=low_th, line_dash="dash", line_color="#34d399", annotation_text=f"Low Cutoff ({low_th:.1f}%)")
                    fig_vol.add_hline(y=high_th, line_dash="dash", line_color="#fbbf24", annotation_text=f"High Cutoff ({high_th:.1f}%)")

                fig_vol.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#131b2e",
                    plot_bgcolor="#131b2e",
                    height=390,
                    margin=dict(l=40, r=40, t=20, b=40),
                    yaxis=dict(title="Annualized Volatility (%)", gridcolor="rgba(255, 255, 255, 0.08)"),
                    xaxis=dict(gridcolor="rgba(255, 255, 255, 0.08)"),
                )
                st.plotly_chart(fig_vol, use_container_width=True)

        with v_col2:
            st.markdown("#### Regime Thresholds")
            if volatility_data:
                v_params = volatility_data["garch_parameters"]
                st.write(f"• **GARCH alpha (shock):** `{v_params['alpha']:.4f}`")
                st.write(f"• **GARCH beta (memory):** `{v_params['beta']:.4f}`")
                st.write(f"• **Persistence (α+β):** `{v_params['persistence']:.4f}`")
                st.write(f"• **Low/Med Cutoff:** `{low_th:.1f}%`")
                st.write(f"• **Med/High Cutoff:** `{high_th:.1f}%`")
                st.caption("Thresholds determined by historical 33.3rd and 66.7th quantiles of annualized volatility.")

        st.markdown("---")
        st.subheader("Current Market Regime Supporting Factors")
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)

        with f_col1:
            st.markdown(
                f"""
                <div class="factor-card" style="border-left-color: #60a5fa;">
                    <div class="factor-title">Trend Factor</div>
                    <div class="factor-desc">{telemetry['supporting_factors']['trend']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with f_col2:
            st.markdown(
                f"""
                <div class="factor-card" style="border-left-color: #34d399;">
                    <div class="factor-title">Momentum Factor</div>
                    <div class="factor-desc">{telemetry['supporting_factors']['momentum']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with f_col3:
            st.markdown(
                f"""
                <div class="factor-card" style="border-left-color: #fbbf24;">
                    <div class="factor-title">Volatility Factor</div>
                    <div class="factor-desc">{telemetry['supporting_factors']['volatility']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with f_col4:
            st.markdown(
                f"""
                <div class="factor-card" style="border-left-color: #f43f5e;">
                    <div class="factor-title">Drawdown Factor</div>
                    <div class="factor-desc">{telemetry['supporting_factors']['drawdown']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.info(f"**Regime Explanation:** {telemetry['regime_explanation']}")

    # -------------------------------------------------------------
    # TAB 3: CRASH EARLY-WARNING
    # -------------------------------------------------------------
    with tab3:
        st.subheader("Crash & Severe Drawdown Early-Warning Radar")
        st.caption("Target definition: Probability that Bitcoin maximum drawdown from current price exceeds -10.0% over the next 14 calendar days.")

        c_col1, c_col2 = st.columns([4, 6])
        with c_col1:
            st.markdown(
                f"""
                <div class="kpi-container" style="padding: 24px; text-align: center;">
                    <div class="kpi-title" style="font-size: 13px; justify-content: center;">Model-Estimated Crash Risk</div>
                    <div class="kpi-value" style="font-size: 44px; color: {'#34d399' if telemetry['crash_risk_level']=='LOW' else ('#fbbf24' if telemetry['crash_risk_level']=='MODERATE' else '#f87171')}; margin: 10px 0;">
                        {telemetry['model_estimated_crash_risk_pct']:.1f}%
                    </div>
                    <div class="kpi-badge {'badge-green' if telemetry['crash_risk_level']=='LOW' else ('badge-amber' if telemetry['crash_risk_level']=='MODERATE' else 'badge-red')}" style="margin: 0 auto; font-size: 13px;">
                        {telemetry['crash_risk_level']} RISK TIER
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
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
                paper_bgcolor="#131b2e",
                plot_bgcolor="#131b2e",
                height=320,
                margin=dict(l=40, r=40, t=10, b=30),
                yaxis=dict(autorange="reversed", gridcolor="rgba(255, 255, 255, 0.08)"),
                xaxis=dict(title="Gini Feature Importance", gridcolor="rgba(255, 255, 255, 0.08)"),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: HISTORICAL PATTERN SIMILARITY
    # -------------------------------------------------------------
    with tab4:
        st.subheader("Historical Pattern Similarity & Subsequent Trajectories")
        st.caption("Compares current 30-day price trajectory against all non-overlapping 30-day historical epochs.")

        sim_col1, sim_col2 = st.columns([1, 2])
        with sim_col1:
            st.markdown(
                f"""
                <div class="kpi-container" style="padding: 20px;">
                    <div class="kpi-title">Top Historical Analog</div>
                    <div class="kpi-value" style="font-size: 26px;">{telemetry['top_similar_date']}</div>
                    <div class="kpi-badge badge-blue" style="margin-top: 4px;">{telemetry['top_similarity_score_pct']:.1f}% SHAPE SIMILARITY</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
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

    # -------------------------------------------------------------
    # TAB 5: WALK-FORWARD BACKTESTING
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # TAB 6: EXPLAINABILITY & METHODOLOGY
    # -------------------------------------------------------------
    with tab6:
        st.subheader("Explainable Model Attributions & Scientific Foundations")

        if explainability_data:
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
               - Our machine learning models forecast stationary 1-day percentage returns $R_{t+1}$, which are mathematically compounded to reconstruct forward price levels: $\\hat{P}_{t+1} = P_t \\times (1 + \\hat{R}_{t+1})$.
            
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


def main():
    # Load core data
    market_df = load_market_data()
    telemetry = load_json_artifact("sentinel_inference_output.json")
    backtest_data = load_json_artifact("walk_forward_backtest_results.json")
    volatility_data = load_json_artifact("volatility_results.json")
    crash_data = load_json_artifact("crash_risk_results.json")
    similarity_matches = load_csv_artifact("pattern_similarity_matches.csv")
    explainability_data = load_json_artifact("explainability_report.json")

    if market_df is None or telemetry is None:
        st.warning(
            "⚠️ Pre-computed model telemetry or processed dataset is not yet found.\n\n"
            "Please run the complete automated training pipeline from the terminal:\n\n"
            "```bash\npython scripts/train.py\n```"
        )
        return

    # Sidebar Navigation & Platform Switcher
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: left; padding: 4px 0px 14px 0px; border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 16px;">
                <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 20px; font-weight: 800; color: #ffffff; display: flex; align-items: center; gap: 8px;">
                    🛡️ BTC Sentinel
                </div>
                <div style="font-size: 11px; color: #f472b6; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;">
                    Institutional Quant Intelligence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Navigation**")
        
        # Navigation Buttons in Sidebar
        if st.session_state["current_page"] == "dashboard":
            if st.button("🏠 Home / Landing Page", use_container_width=True, key="sb_btn_home"):
                st.session_state["current_page"] = "landing"
                st.rerun()
        else:
            if st.button("📊 Open Dashboard Terminal", type="primary", use_container_width=True, key="sb_btn_dash"):
                st.session_state["current_page"] = "dashboard"
                st.rerun()

        st.markdown("---")
        st.markdown("**Telemetry Snapshot**")
        st.markdown(
            f"""
            <div style="font-size: 12px; color: #94a3b8; line-height: 1.7; font-family: 'Inter', sans-serif;">
                • <b>Status:</b> <span style="color: #34d399; font-weight: 600;">● Online</span><br/>
                • <b>Date:</b> <code class="code-font" style="color: #60a5fa;">{telemetry['as_of_date']}</code><br/>
                • <b>BTC:</b> <code class="code-font" style="color: #34d399;">${telemetry['current_price']:,.2f}</code><br/>
                • <b>Regime:</b> <code class="code-font" style="color: #fbbf24;">{telemetry['market_regime']}</code><br/>
                • <b>Crash Hazard:</b> <code class="code-font" style="color: #f87171;">{telemetry['crash_risk_level']}</code><br/>
                • <b>Confidence:</b> <code class="code-font" style="color: #c084fc;">{telemetry['qualitative_confidence']}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.caption("BTC Sentinel v1.0 • Institutional Research Platform • Zero-Leakage Validation")

    # Route interface strictly based on current_page
    if st.session_state["current_page"] == "landing":
        render_landing_page(market_df, telemetry)
    else:
        render_dashboard(
            market_df,
            telemetry,
            volatility_data,
            crash_data,
            similarity_matches,
            backtest_data,
            explainability_data,
        )


if __name__ == "__main__":
    main()
