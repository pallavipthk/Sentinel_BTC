"""Chronological Walk-Forward Backtester for BTC Sentinel.
Enforces expanding-window chronological cross-validation with zero lookahead leakage.
Generates comprehensive comparative metrics, error analyses, and visual reporting.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import calculate_regression_metrics
from src.features.feature_engineer import calculate_features, get_feature_columns
from src.forecasting.arima_model import ARIMAForecaster
from src.forecasting.naive_baseline import NaivePersistenceModel
from src.forecasting.xgboost_forecaster import XGBoostPriceForecaster
from src.utils.logger import setup_logger

logger = setup_logger("backtester")

DEFAULT_RESULTS_DIR = Path("reports/results")
DEFAULT_FIGURES_DIR = Path("reports/figures")


class WalkForwardBacktester:
    """Chronological expanding-window walk-forward backtesting engine."""

    def __init__(
        self,
        n_splits: int = 5,
        test_window_size: int = 90,
    ):
        """
        Args:
            n_splits: Number of sequential walk-forward evaluation folds.
            test_window_size: Number of out-of-sample days per fold (e.g. 90 days = quarterly).
        """
        self.n_splits = n_splits
        self.test_window_size = test_window_size

    def run_backtest(
        self,
        df_features: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target_return_next",
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Runs expanding window walk-forward backtest for Naive, ARIMA, and XGBoost.

        Args:
            df_features: Feature DataFrame containing engineered indicators and targets.
            feature_cols: List of predictor feature column names.
            target_col: Name of the stationary target column (e.g. 'target_return_next').

        Returns:
            Tuple of (overall_results_summary, detailed_predictions_dataframe).
        """
        total_rows = len(df_features)
        total_eval_days = self.n_splits * self.test_window_size

        if total_eval_days >= total_rows:
            raise ValueError(
                f"Backtest configuration invalid: {self.n_splits} folds of {self.test_window_size} days = "
                f"{total_eval_days} days, but total available rows is only {total_rows}."
            )

        initial_train_size = total_rows - total_eval_days
        logger.info(
            f"Configured Walk-Forward Backtest: {self.n_splits} folds x {self.test_window_size} days = "
            f"{total_eval_days} total evaluation days. Initial training set: {initial_train_size} days."
        )

        all_fold_records = []
        fold_summaries = []

        # Iterate sequentially through chronological folds
        for fold in range(self.n_splits):
            train_end_idx = initial_train_size + (fold * self.test_window_size)
            test_end_idx = train_end_idx + self.test_window_size

            train_df = df_features.iloc[:train_end_idx].copy()
            test_df = df_features.iloc[train_end_idx:test_end_idx].copy()

            fold_start_date = test_df["Date"].iloc[0]
            fold_end_date = test_df["Date"].iloc[-1]
            logger.info(
                f"--- Executing Walk-Forward Fold {fold + 1}/{self.n_splits}: "
                f"Train=[{train_df['Date'].iloc[0]} to {train_df['Date'].iloc[-1]}] ({len(train_df)} rows), "
                f"Test=[{fold_start_date} to {fold_end_date}] ({len(test_df)} rows) ---"
            )

            current_prices = test_df["Close"].values
            actual_next_prices = test_df["target_close_next"].values
            actual_next_returns = test_df[target_col].values

            # 1. Naive Persistence Baseline
            naive_pred_prices = current_prices.copy()
            naive_pred_returns = np.zeros_like(actual_next_returns)

            # 2. ARIMA Model (Fit on training, forecast 1-step rolling on test)
            arima_forecaster = ARIMAForecaster(order=(1, 1, 0))
            arima_forecaster.fit(train_df["Close"])
            arima_pred_prices = arima_forecaster.forecast_rolling(train_df["Close"], test_df["Close"])
            arima_pred_returns = (arima_pred_prices - current_prices) / (current_prices + 1e-10)

            # 3. XGBoost Model
            # Split train into 85% train, 15% internal validation for early stopping
            inner_train_size = int(len(train_df) * 0.85)
            inner_train = train_df.iloc[:inner_train_size]
            inner_val = train_df.iloc[inner_train_size:]

            xgb_forecaster = XGBoostPriceForecaster(n_estimators=300, max_depth=3, learning_rate=0.03)
            xgb_forecaster.fit(
                X_train=inner_train[feature_cols],
                y_train=inner_train[target_col],
                X_val=inner_val[feature_cols],
                y_val=inner_val[target_col],
            )
            xgb_pred_returns = xgb_forecaster.predict_returns(test_df[feature_cols])
            xgb_pred_prices = xgb_forecaster.predict_prices(current_prices, test_df[feature_cols])

            # Store fold predictions
            for i in range(len(test_df)):
                all_fold_records.append({
                    "Fold": fold + 1,
                    "Date": test_df["Date"].iloc[i],
                    "Current_Close": current_prices[i],
                    "Actual_Next_Close": actual_next_prices[i],
                    "Actual_Next_Return": actual_next_returns[i],
                    "Naive_Pred_Close": naive_pred_prices[i],
                    "Naive_Error": actual_next_prices[i] - naive_pred_prices[i],
                    "ARIMA_Pred_Close": arima_pred_prices[i],
                    "ARIMA_Error": actual_next_prices[i] - arima_pred_prices[i],
                    "XGB_Pred_Close": xgb_pred_prices[i],
                    "XGB_Pred_Return": xgb_pred_returns[i],
                    "XGB_Error": actual_next_prices[i] - xgb_pred_prices[i],
                })

            # Calculate fold-level metrics
            fold_metrics_naive = calculate_regression_metrics(actual_next_prices, naive_pred_prices, current_prices)
            fold_metrics_arima = calculate_regression_metrics(actual_next_prices, arima_pred_prices, current_prices)
            fold_metrics_xgb = calculate_regression_metrics(
                actual_next_returns, xgb_pred_returns, current_prices, is_return_target=True
            )

            fold_summaries.append({
                "fold": fold + 1,
                "period": [fold_start_date, fold_end_date],
                "naive_mae": fold_metrics_naive["mae"],
                "arima_mae": fold_metrics_arima["mae"],
                "xgb_mae": fold_metrics_xgb["price_mae"],
                "naive_rmse": fold_metrics_naive["rmse"],
                "arima_rmse": fold_metrics_arima["rmse"],
                "xgb_rmse": fold_metrics_xgb["price_rmse"],
                "xgb_directional_acc": fold_metrics_xgb.get("directional_accuracy", 0.0),
            })

        pred_df = pd.DataFrame(all_fold_records)

        # Calculate Overall Out-Of-Sample Metrics across entire backtest
        overall_actual = pred_df["Actual_Next_Close"].values
        overall_current = pred_df["Current_Close"].values
        overall_actual_ret = pred_df["Actual_Next_Return"].values

        overall_naive_metrics = calculate_regression_metrics(
            overall_actual, pred_df["Naive_Pred_Close"].values, overall_current
        )
        overall_arima_metrics = calculate_regression_metrics(
            overall_actual, pred_df["ARIMA_Pred_Close"].values, overall_current
        )
        overall_xgb_metrics = calculate_regression_metrics(
            overall_actual_ret, pred_df["XGB_Pred_Return"].values, overall_current, is_return_target=True
        )

        # Statistical Error Analysis for each model
        error_analysis = {}
        for model_name, err_col in [("Naive", "Naive_Error"), ("ARIMA", "ARIMA_Error"), ("XGBoost", "XGB_Error")]:
            errs = pred_df[err_col].values
            abs_errs = np.abs(errs)
            error_analysis[model_name] = {
                "mean_error": float(np.mean(errs)),
                "std_error": float(np.std(errs)),
                "median_abs_error": float(np.median(abs_errs)),
                "max_abs_error": float(np.max(abs_errs)),
                "95th_percentile_abs_error": float(np.percentile(abs_errs, 95)),
            }

        overall_summary = {
            "n_splits": self.n_splits,
            "test_window_size": self.test_window_size,
            "total_backtest_days": len(pred_df),
            "backtest_period": [str(pred_df["Date"].iloc[0]), str(pred_df["Date"].iloc[-1])],
            "models": {
                "Naive_Persistence": overall_naive_metrics,
                "ARIMA": overall_arima_metrics,
                "XGBoost": {
                    "price_mae": overall_xgb_metrics.get("price_mae"),
                    "price_rmse": overall_xgb_metrics.get("price_rmse"),
                    "price_mape": overall_xgb_metrics.get("price_mape"),
                    "price_smape": overall_xgb_metrics.get("price_smape"),
                    "directional_accuracy": overall_xgb_metrics.get("directional_accuracy"),
                    "return_mae": overall_xgb_metrics.get("mae"),
                    "return_rmse": overall_xgb_metrics.get("rmse"),
                },
            },
            "fold_breakdown": fold_summaries,
            "error_analysis": error_analysis,
        }

        return overall_summary, pred_df


def plot_walk_forward_results(
    pred_df: pd.DataFrame,
    summary: Dict[str, Any],
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> None:
    """Generates charts for walk-forward predictions and cumulative absolute error curves."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    dates = pred_df["Date"]

    # 1. Full Walk-Forward Predictions vs Actual
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(dates, pred_df["Actual_Next_Close"], label="Actual BTC Close", color="#111827", linewidth=1.6)
    ax.plot(dates, pred_df["Naive_Pred_Close"], label="Naive Baseline", color="#9ca3af", linestyle=":", linewidth=1.0)
    ax.plot(dates, pred_df["ARIMA_Pred_Close"], label="ARIMA(1,1,0)", color="#f59e0b", linestyle="--", linewidth=1.1)
    ax.plot(dates, pred_df["XGB_Pred_Close"], label="XGBoost Return-Compounded", color="#10b981", linewidth=1.3)

    # Shade fold boundaries
    folds = pred_df["Fold"].unique()
    for f in folds[:-1]:
        fold_last_date = pred_df[pred_df["Fold"] == f]["Date"].iloc[-1]
        ax.axvline(fold_last_date, color="#6b7280", linestyle="-.", alpha=0.5, linewidth=0.8)

    ax.set_title(
        f"Bitcoin Walk-Forward Backtesting (5 Sequential Folds, {len(pred_df)} Days Out-Of-Sample)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Price (USD)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)

    step = max(1, len(dates) // 8)
    ax.set_xticks(dates.iloc[::step])
    ax.set_xticklabels(dates.iloc[::step], rotation=30, ha="right")

    plt.tight_layout()
    plot_path = figures_dir / "walk_forward_comparison.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    logger.info(f"Walk-forward comparison plot saved to {plot_path}")

    # 2. Cumulative Absolute Error (Tracking drift and model divergence)
    fig, ax = plt.subplots(figsize=(12, 6))
    cum_naive = np.cumsum(np.abs(pred_df["Naive_Error"]))
    cum_arima = np.cumsum(np.abs(pred_df["ARIMA_Error"]))
    cum_xgb = np.cumsum(np.abs(pred_df["XGB_Error"]))

    ax.plot(dates, cum_naive, label="Naive Baseline Cumulative Abs Error", color="#9ca3af", linestyle=":", linewidth=1.5)
    ax.plot(dates, cum_arima, label="ARIMA Cumulative Abs Error", color="#f59e0b", linestyle="--", linewidth=1.5)
    ax.plot(dates, cum_xgb, label="XGBoost Cumulative Abs Error", color="#10b981", linewidth=1.8)

    ax.set_title("Walk-Forward Cumulative Absolute Error Over Time", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Cumulative Absolute Error (USD)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    ax.set_xticks(dates.iloc[::step])
    ax.set_xticklabels(dates.iloc[::step], rotation=30, ha="right")

    plt.tight_layout()
    cum_plot_path = figures_dir / "cumulative_absolute_error.png"
    fig.savefig(cum_plot_path, dpi=200)
    plt.close(fig)
    logger.info(f"Cumulative error plot saved to {cum_plot_path}")


def run_walk_forward_pipeline(
    processed_data_path: Path = Path("data/processed/btc_daily.csv"),
    n_splits: int = 5,
    test_window_size: int = 90,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Runs complete chronological walk-forward backtest and saves results."""
    logger.info("=== STEP 1: Feature Engineering for Walk-Forward Backtesting ===")
    raw_df = pd.read_csv(processed_data_path)
    feature_df = calculate_features(raw_df, include_target=True, drop_na=True)
    feature_cols = get_feature_columns(feature_df)

    # Run backtest
    backtester = WalkForwardBacktester(n_splits=n_splits, test_window_size=test_window_size)
    summary, pred_df = backtester.run_backtest(feature_df, feature_cols)

    # Plots
    plot_walk_forward_results(pred_df, summary, figures_dir=figures_dir)

    # Save artifacts
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "walk_forward_backtest_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pred_path = results_dir / "walk_forward_backtest_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    logger.info("=== Walk-Forward Backtest Aggregate Out-Of-Sample Results ===")
    models = summary["models"]
    logger.info(f"Naive:   MAE=${models['Naive_Persistence']['mae']:,.2f} | RMSE=${models['Naive_Persistence']['rmse']:,.2f} | MAPE={models['Naive_Persistence']['mape']:.2f}%")
    logger.info(f"ARIMA:   MAE=${models['ARIMA']['mae']:,.2f} | RMSE=${models['ARIMA']['rmse']:,.2f} | MAPE={models['ARIMA']['mape']:.2f}% | DirAcc={models['ARIMA']['directional_accuracy']:.1f}%")
    logger.info(f"XGBoost: MAE=${models['XGBoost']['price_mae']:,.2f} | RMSE=${models['XGBoost']['price_rmse']:,.2f} | MAPE={models['XGBoost']['price_mape']:.2f}% | DirAcc={models['XGBoost']['directional_accuracy']:.1f}%")

    return summary, pred_df


if __name__ == "__main__":
    run_walk_forward_pipeline()
