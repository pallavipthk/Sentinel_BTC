"""Unit tests for Market Regime Detection."""

import pandas as pd
import pytest

from src.regime.regime_detector import MarketRegimeDetector


def test_market_regime_classification_rules():
    """Verifies rule-based deterministic regime classification."""
    detector = MarketRegimeDetector(vol_high_threshold=0.65, vol_low_threshold=0.50, drawdown_bear_threshold=-0.20)

    # 1. Bull Momentum row
    bull_row = pd.Series({
        "Date": "2023-03-01",
        "Close": 28000.0,
        "sma_50": 24000.0,
        "sma_200": 20000.0,
        "momentum_30d": 0.15,  # +15%
        "rsi_14": 62.0,
        "rolling_volatility_30d": 0.52,
        "rolling_drawdown_90d": -0.05,
        "drawdown_from_ath": -0.30,
    })
    bull_state = detector.evaluate_single_observation(bull_row)
    assert bull_state.regime == "BULL_MOMENTUM"
    assert "Bullish" in bull_state.trend_state

    # 2. Bear Downtrend row
    bear_row = pd.Series({
        "Date": "2022-06-15",
        "Close": 20000.0,
        "sma_50": 30000.0,
        "sma_200": 40000.0,
        "momentum_30d": -0.35, # -35%
        "rsi_14": 28.0,
        "rolling_volatility_30d": 0.60,
        "rolling_drawdown_90d": -0.45,
        "drawdown_from_ath": -0.70,
    })
    bear_state = detector.evaluate_single_observation(bear_row)
    assert bear_state.regime == "BEAR_DOWNTREND"
    assert "Bearish" in bear_state.trend_state

    # 3. High Volatility Expansion row
    vol_row = pd.Series({
        "Date": "2020-03-15",
        "Close": 5500.0,
        "sma_50": 8000.0,
        "sma_200": 8500.0,
        "momentum_30d": -0.45,
        "rsi_14": 22.0,
        "rolling_volatility_30d": 0.95, # 95% vol
        "rolling_drawdown_90d": -0.50,
        "drawdown_from_ath": -0.75,
    })
    vol_state = detector.evaluate_single_observation(vol_row)
    assert vol_state.regime == "HIGH_VOLATILITY_EXPANSION"
