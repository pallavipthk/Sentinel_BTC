"""Naive Persistence Baseline Forecasting Model.
Implements the random walk hypothesis benchmark:
Forecast for time t+1 is the observed close price at time t: P_{t+1}_hat = P_t
Expected return is zero: r_{t+1}_hat = 0.0
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.evaluation.metrics import calculate_regression_metrics
from src.utils.logger import setup_logger

logger = setup_logger("naive_baseline")

DEFAULT_RESULTS_DIR = Path("reports/results")


class NaivePersistenceModel:
    """Benchmark model predicting persistence (no price change from current day)."""

    def __init__(self):
        self.name = "Naive_Persistence"

    def predict_price(self, current_prices: np.ndarray) -> np.ndarray:
        """Forecasts tomorrow's price as today's close."""
        return np.asarray(current_prices, dtype=float).copy()

    def predict_return(self, n_samples: int) -> np.ndarray:
        """Forecasts zero return for persistence."""
        return np.zeros(n_samples, dtype=float)

    def evaluate(
        self,
        test_df: pd.DataFrame,
        save_artifacts: bool = True,
        output_dir: Path = DEFAULT_RESULTS_DIR,
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Evaluates the naive baseline on a chronological test set.

        Args:
            test_df: Test DataFrame containing at least 'Date', 'Close', and 'target_close_next'.
            save_artifacts: Whether to write predictions and metrics to disk.
            output_dir: Destination folder.

        Returns:
            Tuple of (metrics_dict, predictions_dataframe).
        """
        logger.info(f"Evaluating Naive Persistence Baseline on {len(test_df)} test observations...")

        dates = test_df["Date"].values
        current_prices = test_df["Close"].values
        actual_next_prices = test_df["target_close_next"].values

        # Predicted next price is current price
        predicted_next_prices = self.predict_price(current_prices)

        actual_returns = (actual_next_prices - current_prices) / (current_prices + 1e-10)
        predicted_returns = self.predict_return(len(current_prices))

        metrics = calculate_regression_metrics(
            y_true=actual_next_prices,
            y_pred=predicted_next_prices,
            current_prices=current_prices,
            is_return_target=False,
        )

        pred_df = pd.DataFrame({
            "Date": dates,
            "Current_Close": current_prices,
            "Actual_Next_Close": actual_next_prices,
            "Predicted_Next_Close": predicted_next_prices,
            "Actual_Return": actual_returns,
            "Predicted_Return": predicted_returns,
            "Absolute_Error": np.abs(actual_next_prices - predicted_next_prices),
            "Percentage_Error": np.abs((actual_next_prices - predicted_next_prices) / (actual_next_prices + 1e-10)) * 100.0,
        })

        if save_artifacts:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            metrics_payload = {
                "model_name": self.name,
                "n_samples": len(test_df),
                "evaluation_period": [str(dates[0]), str(dates[-1])],
                "metrics": metrics,
            }

            metrics_path = output_dir / "baseline_results.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_payload, f, indent=2)

            pred_path = output_dir / "baseline_predictions.csv"
            pred_df.to_csv(pred_path, index=False)

            logger.info(f"Baseline results saved to {metrics_path}")
            logger.info(f"Baseline predictions saved to {pred_path}")

        logger.info(
            f"Baseline Performance: MAE=${metrics['mae']:,.2f}, RMSE=${metrics['rmse']:,.2f}, "
            f"MAPE={metrics['mape']:.2f}%, sMAPE={metrics['smape']:.2f}%, DirAcc={metrics.get('directional_accuracy', 0.0):.1f}%"
        )

        return metrics, pred_df


def run_baseline_evaluation(
    processed_data_path: Path = Path("data/processed/btc_daily.csv"),
    test_ratio: float = 0.2,
    output_dir: Path = DEFAULT_RESULTS_DIR,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Runs end-to-end baseline evaluation on processed data with chronological split.

    Args:
        processed_data_path: Path to cleaned BTC OHLCV CSV.
        test_ratio: Fraction of latest chronological data reserved for evaluation.
        output_dir: Destination directory for artifacts.

    Returns:
        Tuple of (metrics_dict, predictions_dataframe).
    """
    from src.features.feature_engineer import calculate_features

    raw_data = pd.read_csv(processed_data_path)
    feature_df = calculate_features(raw_data, include_target=True, drop_na=True)

    n_total = len(feature_df)
    split_idx = int(n_total * (1.0 - test_ratio))

    train_df = feature_df.iloc[:split_idx].copy()
    test_df = feature_df.iloc[split_idx:].copy()

    logger.info(f"Chronological split: Train={len(train_df)} rows ({train_df['Date'].iloc[0]} to {train_df['Date'].iloc[-1]}), Test={len(test_df)} rows ({test_df['Date'].iloc[0]} to {test_df['Date'].iloc[-1]})")

    model = NaivePersistenceModel()
    metrics, pred_df = model.evaluate(test_df, save_artifacts=True, output_dir=output_dir)
    return metrics, pred_df
