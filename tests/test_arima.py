"""Tests for ARIMA forecasting, stationarity testing, and residual diagnostics."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.arima_model import (
    ARIMAForecaster,
    check_stationarity,
    find_optimal_arima_order,
)


@pytest.fixture
def synthetic_price_series() -> pd.Series:
    """Generates synthetic random walk price series."""
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, size=150)
    prices = 30000.0 * np.exp(np.cumsum(returns))
    return pd.Series(prices)


def test_stationarity_check(synthetic_price_series):
    """Verifies ADF stationarity identification on raw prices vs differences."""
    # Raw random walk price should fail stationarity
    raw_res = check_stationarity(synthetic_price_series)
    assert "adf_statistic" in raw_res
    assert "p_value" in raw_res

    # Differenced series should be stationary
    diff_series = synthetic_price_series.diff().dropna()
    diff_res = check_stationarity(diff_series)
    assert diff_res["p_value"] < 0.05
    assert diff_res["is_stationary"] is True


def test_arima_order_selection_and_forecasting(synthetic_price_series):
    """Tests grid search, fitting, diagnostics, and 1-step rolling forecast."""
    train_series = synthetic_price_series.iloc[:120]
    test_series = synthetic_price_series.iloc[120:]

    # Test order search with restricted grid for test speed
    best_order, best_aic, history = find_optimal_arima_order(
        train_series,
        p_range=range(0, 2),
        d_range=range(1, 2),
        q_range=range(0, 2),
    )
    assert len(best_order) == 3
    assert best_order[1] == 1  # differencing is 1
    assert np.isfinite(best_aic)

    forecaster = ARIMAForecaster(order=best_order)
    forecaster.fit(train_series)

    # Verify residual diagnostics
    diag = forecaster.diagnostics_
    assert "aic" in diag
    assert "residual_diagnostics" in diag
    res_diag = diag["residual_diagnostics"]
    assert "mean_residual" in res_diag
    assert "jarque_bera" in res_diag

    # Verify rolling forecast
    preds = forecaster.forecast_rolling(train_series, test_series)
    assert len(preds) == len(test_series)
    assert np.all(np.isfinite(preds))
    assert np.all(preds > 0.0)
