"""Tests for Stage 2 Feature Engineering and Naive Baseline.
Includes strict mathematical leakage verification, indicator range bounds,
and baseline evaluation correctness.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineer import calculate_features, get_feature_columns
from src.forecasting.naive_baseline import NaivePersistenceModel


@pytest.fixture
def sample_ohlcv_series() -> pd.DataFrame:
    """Generates synthetic 250-day trending and oscillating OHLCV series."""
    np.random.seed(42)
    n = 250
    dates = pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y-%m-%d")

    # Generate geometric Brownian motion
    returns = np.random.normal(0.001, 0.03, size=n)
    prices = 20000.0 * np.exp(np.cumsum(returns))

    highs = prices * (1.0 + np.abs(np.random.normal(0.01, 0.005, size=n)))
    lows = prices * (1.0 - np.abs(np.random.normal(0.01, 0.005, size=n)))
    opens = (highs + lows) / 2.0
    volumes = np.random.uniform(5000, 50000, size=n)

    return pd.DataFrame({
        "Date": dates,
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": prices,
        "Volume": volumes,
    })


def test_leakage_strict_mathematical_invariance(sample_ohlcv_series):
    """CRITICAL TEST: Mathematically proves that features at time t do NOT leak future data.

    Protocol:
    1. Compute features on unmodified series.
    2. Pick an anchor index k (e.g. index 180).
    3. Drastically alter future observations at index > k (e.g. multiply price by 10x).
    4. Recompute features on the mutated series.
    5. Assert that every single feature at index <= k is bitwise/identical within float tolerance.
    """
    df_original = sample_ohlcv_series.copy()
    features_orig = calculate_features(df_original, include_target=False, drop_na=False)

    anchor_k = 180
    feature_cols = get_feature_columns(features_orig)

    # Mutate the future (indices anchor_k + 1 to end)
    df_mutated = sample_ohlcv_series.copy()
    future_mask = df_mutated.index > anchor_k
    df_mutated.loc[future_mask, "Close"] = df_mutated.loc[future_mask, "Close"] * 10.0
    df_mutated.loc[future_mask, "High"] = df_mutated.loc[future_mask, "High"] * 10.0
    df_mutated.loc[future_mask, "Low"] = df_mutated.loc[future_mask, "Low"] * 10.0
    df_mutated.loc[future_mask, "Volume"] = df_mutated.loc[future_mask, "Volume"] * 0.1

    features_mutated = calculate_features(df_mutated, include_target=False, drop_na=False)

    # Check every feature at and before anchor_k
    for col in feature_cols:
        orig_vals = features_orig.loc[:anchor_k, col].values
        mutated_vals = features_mutated.loc[:anchor_k, col].values

        # Both may contain initial NaNs from rolling warmup; where not NaN, must match exactly
        valid_mask = ~np.isnan(orig_vals)
        if valid_mask.sum() > 0:
            diff = np.abs(orig_vals[valid_mask] - mutated_vals[valid_mask])
            max_diff = np.max(diff)
            assert max_diff < 1e-9, f"LEAKAGE DETECTED in feature '{col}'! Difference at or before anchor: {max_diff}"


def test_feature_ranges_and_properties(sample_ohlcv_series):
    """Verifies statistical boundaries and structural properties of engineered features."""
    df_feats = calculate_features(sample_ohlcv_series, include_target=True, drop_na=True)

    # 1. Feature columns helper works
    feat_cols = get_feature_columns(df_feats)
    assert len(feat_cols) > 30
    assert "target_close_next" not in feat_cols
    assert "Date" not in feat_cols

    # 2. RSI bounds [0, 100]
    assert (df_feats["rsi_14"] >= 0.0).all()
    assert (df_feats["rsi_14"] <= 100.0).all()

    # 3. Drawdowns must be non-positive (<= 0.0)
    assert (df_feats["drawdown_from_ath"] <= 1e-6).all()
    assert (df_feats["rolling_drawdown_30d"] <= 1e-6).all()

    # 4. Target close next is exactly close shifted by -1
    # Check first row
    assert np.isclose(df_feats["target_close_next"].iloc[0], sample_ohlcv_series["Close"].iloc[len(sample_ohlcv_series) - len(df_feats)])


def test_naive_persistence_baseline_evaluation(tmp_path):
    """Tests Naive baseline forecasting accuracy calculations and artifact serialization."""
    test_df = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
        "Close": [20000.0, 20500.0, 20200.0, 21000.0],
        "target_close_next": [20500.0, 20200.0, 21000.0, 21500.0],
    })

    model = NaivePersistenceModel()
    metrics, pred_df = model.evaluate(test_df, save_artifacts=True, output_dir=tmp_path)

    # Persistence predictions must equal Close
    assert list(pred_df["Predicted_Next_Close"]) == [20000.0, 20500.0, 20200.0, 21000.0]
    # Check MAE calculation: mean of |20500-20000|, |20200-20500|, |21000-20200|, |21500-21000|
    # Errors: 500, 300, 800, 500 -> sum=2100 / 4 = 525.0
    assert np.isclose(metrics["mae"], 525.0)

    # Artifacts exist
    assert (tmp_path / "baseline_results.json").exists()
    assert (tmp_path / "baseline_predictions.csv").exists()
