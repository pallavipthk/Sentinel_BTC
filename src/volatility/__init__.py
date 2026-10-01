"""Volatility modeling and forecasting package for BTC Sentinel."""

from src.volatility.garch_model import (
    GARCHModel,
    compute_volatility_regimes,
    evaluate_volatility_forecasts,
    run_volatility_pipeline,
)

__all__ = [
    "GARCHModel",
    "compute_volatility_regimes",
    "evaluate_volatility_forecasts",
    "run_volatility_pipeline",
]
