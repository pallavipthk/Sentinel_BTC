"""XGBoost Price Forecasting Engine.
Predicts stationary 1-day-ahead returns using leakage-safe features.
Converts predicted returns back to future price levels: P_{t+1}_hat = P_t * (1 + r_{t+1}_hat).
Strictly adheres to chronological splitting with early stopping on an out-of-sample validation slice.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.evaluation.metrics import calculate_regression_metrics
from src.features.feature_engineer import calculate_features, get_feature_columns
from src.utils.logger import setup_logger

logger = setup_logger("xgboost_forecaster")

DEFAULT_RESULTS_DIR = Path("reports/results")
DEFAULT_FIGURES_DIR = Path("reports/figures")
DEFAULT_MODELS_DIR = Path("models")


class XGBoostPriceForecaster:
    """XGBoost regression forecaster for Bitcoin 1-day-ahead price and return."""

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 3,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "early_stopping_rounds": 30,
        }
        self.model = XGBRegressor(**self.params)
        self.feature_names_: List[str] = []
        self.feature_importances_: Dict[str, float] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "XGBoostPriceForecaster":
        """Fits XGBoost model with early stopping on validation split."""
        self.feature_names_ = list(X_train.columns)
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        logger.info(f"Training XGBoost regressor on {len(X_train)} samples across {len(self.feature_names_)} features...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        importances = self.model.feature_importances_
        self.feature_importances_ = {
            feat: float(imp)
            for feat, imp in sorted(zip(self.feature_names_, importances), key=lambda x: x[1], reverse=True)
        }
        top_3 = list(self.feature_importances_.items())[:3]
        logger.info(f"Top 3 most influential features: {top_3}")
        return self

    def predict_returns(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts 1-day-ahead return."""
        return self.model.predict(X[self.feature_names_])

    def predict_prices(self, current_prices: np.ndarray, X: pd.DataFrame) -> np.ndarray:
        """Predicts 1-day-ahead price levels by compounding predicted returns with current price."""
        pred_returns = self.predict_returns(X)
        c_p = np.asarray(current_prices, dtype=float)
        return c_p * (1.0 + pred_returns)

    def save(self, filepath: Path) -> None:
        """Serializes model and metadata to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names_,
                "feature_importances": self.feature_importances_,
                "params": self.params,
            },
            filepath,
        )
        logger.info(f"XGBoost model saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "XGBoostPriceForecaster":
        """Loads serialized model."""
        data = joblib.load(filepath)
        instance = cls()
        instance.model = data["model"]
        instance.feature_names_ = data["feature_names"]
        instance.feature_importances_ = data["feature_importances"]
        instance.params = data["params"]
        return instance


def plot_price_comparison(
    dates: pd.Series,
    actual: np.ndarray,
    baseline_pred: np.ndarray,
    arima_pred: Optional[np.ndarray],
    xgb_pred: np.ndarray,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> None:
    """Plots actual price versus Naive, ARIMA, and XGBoost forecasts."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(dates, actual, label="Actual BTC Close", color="#111827", linewidth=1.8)
    ax.plot(dates, baseline_pred, label="Naive Baseline", color="#9ca3af", linestyle=":", linewidth=1.2)
    if arima_pred is not None:
        ax.plot(dates, arima_pred, label="ARIMA(1,1,0)", color="#f59e0b", linestyle="--", linewidth=1.2)
    ax.plot(dates, xgb_pred, label="XGBoost Return-Compounded", color="#10b981", linewidth=1.4)

    ax.set_title("Bitcoin 1-Day Price Forecast: Model Comparison (Out-of-Sample)", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Price (USD)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)

    step = max(1, len(dates) // 8)
    ax.set_xticks(dates.iloc[::step])
    ax.set_xticklabels(dates.iloc[::step], rotation=30, ha="right")

    plt.tight_layout()
    plot_path = figures_dir / "price_forecast_comparison.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    logger.info(f"Price forecast comparison figure saved to {plot_path}")


def plot_feature_importance(
    importances: Dict[str, float],
    top_n: int = 15,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> None:
    """Plots top N most important features in the XGBoost forecasting model."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    top_items = list(importances.items())[:top_n]
    feats = [item[0] for item in reversed(top_items)]
    scores = [item[1] for item in reversed(top_items)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feats, scores, color="#3b82f6", alpha=0.85, edgecolor="#1d4ed8")
    ax.set_title(f"XGBoost Price Model: Top {top_n} Feature Importances (Gain)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Relative Importance Score", fontsize=10)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)

    plt.tight_layout()
    feat_plot_path = figures_dir / "xgboost_feature_importance.png"
    fig.savefig(feat_plot_path, dpi=200)
    plt.close(fig)
    logger.info(f"Feature importance plot saved to {feat_plot_path}")


def run_xgboost_price_pipeline(
    processed_data_path: Path = Path("data/processed/btc_daily.csv"),
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Runs complete XGBoost price forecasting training, evaluation, and comparison.

    Args:
        processed_data_path: Path to processed daily OHLCV dataset.
        train_ratio: Fraction for training split.
        val_ratio: Fraction for validation split.
        results_dir: Results directory.
        figures_dir: Figures directory.
        models_dir: Models serialization directory.

    Returns:
        Tuple of (comparison_results_dict, test_predictions_df).
    """
    logger.info("=== STEP 1: Feature Engineering for XGBoost Price Forecaster ===")
    raw_df = pd.read_csv(processed_data_path)
    feature_df = calculate_features(raw_df, include_target=True, drop_na=True)

    feature_cols = get_feature_columns(feature_df)
    target_col = "target_return_next"

    # Chronological 3-way split: Train (70%) -> Val (10%) -> Test (20%)
    n_total = len(feature_df)
    train_end = int(n_total * train_ratio)
    val_end = int(n_total * (train_ratio + val_ratio))

    train_data = feature_df.iloc[:train_end].copy()
    val_data = feature_df.iloc[train_end:val_end].copy()
    test_data = feature_df.iloc[val_end:].copy()

    logger.info(
        f"Chronological Splits: Train={len(train_data)} ({train_data['Date'].iloc[0]} to {train_data['Date'].iloc[-1]}), "
        f"Val={len(val_data)} ({val_data['Date'].iloc[0]} to {val_data['Date'].iloc[-1]}), "
        f"Test={len(test_data)} ({test_data['Date'].iloc[0]} to {test_data['Date'].iloc[-1]})"
    )

    X_train, y_train = train_data[feature_cols], train_data[target_col]
    X_val, y_val = val_data[feature_cols], val_data[target_col]
    X_test, y_test = test_data[feature_cols], test_data[target_col]

    # STEP 2: Model Training
    forecaster = XGBoostPriceForecaster()
    forecaster.fit(X_train, y_train, X_val, y_val)

    # STEP 3: Out-of-sample Test Predictions
    current_prices = test_data["Close"].values
    actual_next_prices = test_data["target_close_next"].values
    actual_next_returns = test_data["target_return_next"].values

    pred_returns = forecaster.predict_returns(X_test)
    pred_prices = forecaster.predict_prices(current_prices, X_test)

    # STEP 4: Compute Metrics for XGBoost
    xgb_metrics = calculate_regression_metrics(
        y_true=actual_next_returns,
        y_pred=pred_returns,
        current_prices=current_prices,
        is_return_target=True,
    )

    # STEP 5: Baseline on same test slice
    baseline_pred_prices = current_prices.copy()
    baseline_pred_returns = np.zeros_like(pred_returns)
    baseline_metrics = calculate_regression_metrics(
        y_true=actual_next_prices,
        y_pred=baseline_pred_prices,
        current_prices=current_prices,
        is_return_target=False,
    )

    # STEP 6: Load ARIMA predictions on same test dates if available
    arima_results_file = results_dir / "arima_predictions.csv"
    arima_test_preds = None
    arima_metrics = None
    if arima_results_file.exists():
        try:
            arima_df = pd.read_csv(arima_results_file)
            merged = pd.merge(test_data[["Date"]], arima_df[["Date", "Predicted_Close"]], on="Date", how="left")
            if not merged["Predicted_Close"].isnull().any():
                arima_test_preds = merged["Predicted_Close"].values
                arima_metrics = calculate_regression_metrics(
                    y_true=actual_next_prices,
                    y_pred=arima_test_preds,
                    current_prices=current_prices,
                    is_return_target=False,
                )
        except Exception as e:
            logger.warning(f"Could not align ARIMA predictions: {e}")

    # STEP 7: Plots
    plot_price_comparison(
        dates=test_data["Date"],
        actual=actual_next_prices,
        baseline_pred=baseline_pred_prices,
        arima_pred=arima_test_preds,
        xgb_pred=pred_prices,
        figures_dir=figures_dir,
    )

    plot_feature_importance(forecaster.feature_importances_, top_n=15, figures_dir=figures_dir)

    # STEP 8: Save Artifacts
    test_pred_df = pd.DataFrame({
        "Date": test_data["Date"].values,
        "Current_Close": current_prices,
        "Actual_Next_Close": actual_next_prices,
        "Actual_Next_Return": actual_next_returns,
        "XGB_Predicted_Return": pred_returns,
        "XGB_Predicted_Close": pred_prices,
        "Baseline_Predicted_Close": baseline_pred_prices,
        "XGB_Absolute_Error": np.abs(actual_next_prices - pred_prices),
        "XGB_Percentage_Error": np.abs((actual_next_prices - pred_prices) / actual_next_prices) * 100.0,
    })
    if arima_test_preds is not None:
        test_pred_df["ARIMA_Predicted_Close"] = arima_test_preds

    test_pred_path = results_dir / "xgboost_price_predictions.csv"
    test_pred_df.to_csv(test_pred_path, index=False)

    comparison_summary = {
        "evaluation_period": [str(test_data["Date"].iloc[0]), str(test_data["Date"].iloc[-1])],
        "n_test_samples": len(test_data),
        "target_variable": "1-day-ahead return (target_return_next) compounded to price",
        "models": {
            "Naive_Persistence": baseline_metrics,
            "XGBoost": {
                "price_mae": xgb_metrics.get("price_mae"),
                "price_rmse": xgb_metrics.get("price_rmse"),
                "price_mape": xgb_metrics.get("price_mape"),
                "price_smape": xgb_metrics.get("price_smape"),
                "directional_accuracy": xgb_metrics.get("directional_accuracy"),
                "return_mae": xgb_metrics.get("mae"),
                "return_rmse": xgb_metrics.get("rmse"),
            },
        },
        "top_features": list(forecaster.feature_importances_.items())[:10],
    }
    if arima_metrics is not None:
        comparison_summary["models"]["ARIMA"] = arima_metrics

    results_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = results_dir / "price_model_comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    forecaster.save(models_dir / "xgboost_price_model.joblib")

    logger.info("=== Model Performance Comparison ===")
    logger.info(f"Naive:   MAE=${baseline_metrics['mae']:,.2f} | RMSE=${baseline_metrics['rmse']:,.2f} | MAPE={baseline_metrics['mape']:.2f}% | DirAcc={baseline_metrics.get('directional_accuracy', 0.0):.1f}%")
    if arima_metrics:
        logger.info(f"ARIMA:   MAE=${arima_metrics['mae']:,.2f} | RMSE=${arima_metrics['rmse']:,.2f} | MAPE={arima_metrics['mape']:.2f}% | DirAcc={arima_metrics.get('directional_accuracy', 0.0):.1f}%")
    logger.info(f"XGBoost: MAE=${xgb_metrics['price_mae']:,.2f} | RMSE=${xgb_metrics['price_rmse']:,.2f} | MAPE={xgb_metrics['price_mape']:.2f}% | DirAcc={xgb_metrics['directional_accuracy']:.1f}%")

    return comparison_summary, test_pred_df


if __name__ == "__main__":
    run_xgboost_price_pipeline()
