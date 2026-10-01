"""Unit tests for XGBoost Price Forecaster."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.forecasting.xgboost_forecaster import XGBoostPriceForecaster


@pytest.fixture
def synthetic_ml_dataset():
    """Generates synthetic feature matrix and return targets."""
    np.random.seed(42)
    n = 200
    features = {f"feat_{i}": np.random.normal(0, 1, size=n) for i in range(10)}
    X = pd.DataFrame(features)
    # Synthetic target correlated with feat_0 and feat_1
    y = 0.4 * X["feat_0"] - 0.3 * X["feat_1"] + np.random.normal(0, 0.02, size=n)
    return X, pd.Series(y)


def test_xgboost_fit_predict_and_feature_importances(synthetic_ml_dataset, tmp_path):
    """Verifies XGBoost training, prediction compounding, importance extraction, and save/load."""
    X, y = synthetic_ml_dataset

    train_X, val_X, test_X = X.iloc[:140], X.iloc[140:170], X.iloc[170:]
    train_y, val_y, test_y = y.iloc[:140], y.iloc[140:170], y.iloc[170:]

    forecaster = XGBoostPriceForecaster(n_estimators=50, max_depth=3)
    forecaster.fit(train_X, train_y, val_X, val_y)

    # Predictions
    pred_returns = forecaster.predict_returns(test_X)
    assert len(pred_returns) == len(test_X)
    assert np.all(np.isfinite(pred_returns))

    # Compounding price
    curr_prices = np.full(len(test_X), 50000.0)
    pred_prices = forecaster.predict_prices(curr_prices, test_X)
    assert len(pred_prices) == len(test_X)
    assert np.all(pred_prices > 0.0)

    # Feature importances
    assert len(forecaster.feature_importances_) == 10
    top_feature = list(forecaster.feature_importances_.keys())[0]
    # feat_0 or feat_1 should be among top features
    assert top_feature in ["feat_0", "feat_1"]

    # Save and reload test
    save_file = tmp_path / "test_xgb.joblib"
    forecaster.save(save_file)
    assert save_file.exists()

    loaded = XGBoostPriceForecaster.load(save_file)
    assert loaded.feature_names_ == forecaster.feature_names_
    loaded_preds = loaded.predict_returns(test_X)
    assert np.allclose(pred_returns, loaded_preds)
