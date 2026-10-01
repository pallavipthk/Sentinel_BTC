"""Leakage-Safe Feature Engineering for BTC Sentinel.
Computes technical, statistical, momentum, volatility, and volume indicators.
Guarantees that feature values at time t utilize only historical observations (s <= t).
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("feature_engineer")


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI) using Wilder's smoothing.
    Strictly causal (no lookahead leakage).
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's exponential moving average
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Average True Range (ATR) over past 'period' days.
    Causal calculation using lagged close for gap detection.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


def calculate_features(
    df: pd.DataFrame,
    include_target: bool = True,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Generates leakage-safe feature representation for Bitcoin time-series forecasting.

    Timing contract:
    - For row at timestamp t, every feature f_t is computed strictly from observations <= t.
    - If include_target=True, target_* columns represent future ground truth at t+1.

    Args:
        df: Input DataFrame containing clean OHLCV data (Date, Open, High, Low, Close, Volume).
        include_target: Whether to attach 1-day-ahead prediction targets.
        drop_na: If True, drops initial warm-up rows containing NaNs from rolling windows.

    Returns:
        pd.DataFrame enriched with engineered features and optional prediction targets.
    """
    logger.info("Computing leakage-safe features...")
    data = df.copy()

    # Ensure chronological order
    data["DatetimeIndex"] = pd.to_datetime(data["Date"])
    data = data.sort_values(by="DatetimeIndex").reset_index(drop=True)
    data = data.drop(columns=["DatetimeIndex"])

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    # 1. Log Returns & Simple Returns (historical up to t)
    data["log_return_1d"] = np.log(close / close.shift(1))
    data["return_1d"] = close.pct_change(1)
    data["log_return_3d"] = np.log(close / close.shift(3))
    data["log_return_7d"] = np.log(close / close.shift(7))
    data["log_return_14d"] = np.log(close / close.shift(14))
    data["log_return_30d"] = np.log(close / close.shift(30))

    # 2. Lagged Returns & Lagged Prices
    for lag in [1, 2, 3, 5, 7, 14]:
        data[f"return_lag_{lag}"] = data["return_1d"].shift(lag)
        data[f"close_ratio_lag_{lag}"] = close / close.shift(lag)

    # 3. Rolling Moving Averages & Price-to-MA Ratios
    for window in [7, 14, 30, 50, 200]:
        ma = close.rolling(window=window, min_periods=window).mean()
        data[f"sma_{window}"] = ma
        data[f"ratio_close_to_sma_{window}"] = close / (ma + 1e-10)

    # Exponential Moving Averages
    data["ema_12"] = close.ewm(span=12, adjust=False).mean()
    data["ema_26"] = close.ewm(span=26, adjust=False).mean()
    data["macd"] = data["ema_12"] - data["ema_26"]
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    # 4. Rolling Standard Deviation & Volatility (Annualized: daily * sqrt(365))
    sqrt_365 = np.sqrt(365.0)
    for window in [7, 14, 30, 60, 90]:
        roll_std = data["log_return_1d"].rolling(window=window, min_periods=window).std()
        data[f"rolling_std_{window}d"] = roll_std
        data[f"rolling_volatility_{window}d"] = roll_std * sqrt_365

    # Parkinson Volatility estimator (using High and Low)
    # sigma_p = sqrt( 1 / (4 * ln(2)) * sum( ln(H/L)^2 ) )
    log_hl = np.log(high / low)
    parkinson_factor = 1.0 / (4.0 * np.log(2.0))
    for window in [14, 30]:
        parkinson_var = parkinson_factor * (log_hl ** 2).rolling(window=window).mean()
        data[f"parkinson_volatility_{window}d"] = np.sqrt(parkinson_var) * sqrt_365

    # 5. Momentum Indicators
    data["momentum_7d"] = (close / close.shift(7)) - 1.0
    data["momentum_14d"] = (close / close.shift(14)) - 1.0
    data["momentum_30d"] = (close / close.shift(30)) - 1.0
    data["momentum_90d"] = (close / close.shift(90)) - 1.0

    # 6. Rolling Extremes & Price Position in Range
    for window in [14, 30, 90]:
        roll_high = high.rolling(window=window, min_periods=window).max()
        roll_low = low.rolling(window=window, min_periods=window).min()
        data[f"rolling_high_{window}d"] = roll_high
        data[f"rolling_low_{window}d"] = roll_low
        # Stochastic-style position: (Close - Low_w) / (High_w - Low_w)
        data[f"range_position_{window}d"] = (close - roll_low) / ((roll_high - roll_low) + 1e-10)

    # 7. Drawdown from Historical All-Time High & Rolling Peaks
    # ATH up to day t (no lookahead)
    ath_series = close.cummax()
    data["drawdown_from_ath"] = (close - ath_series) / ath_series
    for window in [30, 90, 180]:
        rolling_peak = close.rolling(window=window, min_periods=window).max()
        data[f"rolling_drawdown_{window}d"] = (close - rolling_peak) / (rolling_peak + 1e-10)

    # 8. Volume Dynamics & Relative Volume
    data["volume_change_1d"] = volume.pct_change(1)
    for window in [7, 14, 30]:
        vol_mean = volume.rolling(window=window, min_periods=window).mean()
        data[f"volume_ratio_to_mean_{window}d"] = volume / (vol_mean + 1e-10)

    # Volume-Price interaction: Close Return * Normalized Volume
    data["volume_flow_proxy"] = data["return_1d"] * data["volume_ratio_to_mean_14d"]

    # 9. Technical Indicators: RSI, Bollinger Bands, ATR
    data["rsi_14"] = calculate_rsi(close, period=14)
    data["rsi_7"] = calculate_rsi(close, period=7)

    # Bollinger Bands (20-day, 2 std)
    bb_mean = close.rolling(window=20, min_periods=20).mean()
    bb_std = close.rolling(window=20, min_periods=20).std()
    bb_upper = bb_mean + (2.0 * bb_std)
    bb_lower = bb_mean - (2.0 * bb_std)
    data["bb_percent_b"] = (close - bb_lower) / ((bb_upper - bb_lower) + 1e-10)
    data["bb_bandwidth"] = (bb_upper - bb_lower) / (bb_mean + 1e-10)

    # ATR
    data["atr_14"] = calculate_atr(high, low, close, period=14)
    data["atr_ratio"] = data["atr_14"] / (close + 1e-10)

    # 10. TARGET CREATION (Strictly shifted into the future t+1)
    if include_target:
        # 1-day-ahead close price
        data["target_close_next"] = close.shift(-1)
        # 1-day-ahead simple return
        data["target_return_next"] = (data["target_close_next"] - close) / close
        # 1-day-ahead log return
        data["target_log_return_next"] = np.log(data["target_close_next"] / close)
        # 1-day-ahead binary direction: 1 if price rises, 0 if flat or falls
        data["target_direction_next"] = (data["target_return_next"] > 0.0).astype(int)

    if drop_na:
        # Note: If target is included, the last row will have NaN for target_* because tomorrow is unobserved.
        # When drop_na=True, drop rows where features or targets are NaN
        initial_len = len(data)
        data = data.dropna().reset_index(drop=True)
        dropped_len = initial_len - len(data)
        logger.info(f"Dropped {dropped_len} warm-up/incomplete rows. Final feature matrix size: {len(data)} rows, {len(data.columns)} columns.")

    return data


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Returns the list of pure feature input column names (excluding date, raw targets, metadata).

    Args:
        df: Feature DataFrame.

    Returns:
        List of feature column names.
    """
    excluded = {
        "Date",
        "target_close_next",
        "target_return_next",
        "target_log_return_next",
        "target_direction_next",
    }
    return [col for col in df.columns if col not in excluded]
