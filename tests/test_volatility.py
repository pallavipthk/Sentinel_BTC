"""Unit tests for GARCH(1,1) Volatility Modeling and Regime Segmentation."""

import numpy as np
import pandas as pd
import pytest

from src.volatility.garch_model import (
    GARCHModel,
    compute_volatility_regimes,
    evaluate_volatility_forecasts,
)


@pytest.fixture
def synthetic_return_series():
    """Generates synthetic return series with volatility clustering."""
    np.random.seed(42)
    n = 300
    # Simulate GARCH(1,1) series
    omega, alpha, beta = 1e-5, 0.1, 0.85
    variances = np.zeros(n)
    variances[0] = 0.0004
    returns = np.zeros(n)
    returns[0] = np.random.normal(0, np.sqrt(variances[0]))

    for t in range(1, n):
        variances[t] = omega + alpha * (returns[t - 1] ** 2) + beta * variances[t - 1]
        returns[t] = np.random.normal(0, np.sqrt(variances[t]))

    return returns


def test_garch_model_fit_and_filter(synthetic_return_series):
    """Tests GARCH MLE fitting, parameter stationarity, and conditional volatility filtering."""
    model = GARCHModel()
    model.fit(synthetic_return_series)

    # Stationarity condition
    assert model.alpha + model.beta < 1.0
    assert model.alpha > 0.0
    assert model.beta > 0.0
    assert model.omega > 0.0

    # Filter conditional volatility
    cond_vol = model.filter_conditional_volatility(synthetic_return_series)
    assert len(cond_vol) == len(synthetic_return_series)
    assert np.all(cond_vol > 0.0)
    assert np.all(np.isfinite(cond_vol))

    # Test 1-step forecast
    next_vol = model.forecast_1step_ahead(last_return=synthetic_return_series[-1], last_variance=0.0004)
    assert next_vol > 0.0
    assert np.isfinite(next_vol)


def test_volatility_regime_segmentation():
    """Verifies empirical quantile regime classification into LOW, MEDIUM, HIGH."""
    vols = pd.Series([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
    regimes, thresholds = compute_volatility_regimes(vols, low_percentile=33.3, high_percentile=66.7)

    assert "low_cutoff" in thresholds
    assert "high_cutoff" in thresholds
    assert thresholds["low_cutoff"] < thresholds["high_cutoff"]

    # Check regime labels
    assert set(regimes.unique()).issubset({"LOW", "MEDIUM", "HIGH"})
    assert regimes.iloc[0] == "LOW"
    assert regimes.iloc[-1] == "HIGH"


def test_volatility_evaluation_metrics():
    """Tests QLIKE, MAE, and RMSE calculation."""
    actual = np.array([0.5, 0.6, 0.4])
    forecast = np.array([0.52, 0.58, 0.42])

    metrics = evaluate_volatility_forecasts(actual, forecast)
    assert "qlike" in metrics
    assert "volatility_mae" in metrics
    assert "volatility_rmse" in metrics
    assert metrics["volatility_mae"] >= 0.0
    assert metrics["volatility_rmse"] >= metrics["volatility_mae"]
