"""ARIMA Time-Series Forecasting Model & Statistical Diagnostics.
Features:
- Augmented Dickey-Fuller (ADF) stationarity testing.
- Grid-search order selection driven by Akaike Information Criterion (AIC).
- Out-of-sample 1-day-ahead rolling forecasting.
- Residual diagnostics (Ljung-Box white noise test, Jarque-Bera normality test).
- Generation and serialization of diagnostic charts and forecast metrics.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from src.evaluation.metrics import calculate_regression_metrics
from src.utils.logger import setup_logger

logger = setup_logger("arima_forecaster")

DEFAULT_RESULTS_DIR = Path("reports/results")
DEFAULT_FIGURES_DIR = Path("reports/figures")
DEFAULT_MODELS_DIR = Path("models")


def check_stationarity(series: pd.Series, max_lag: int = 14) -> Dict[str, Any]:
    """Performs Augmented Dickey-Fuller (ADF) test for unit root stationarity.

    Args:
        series: Pandas series to test.
        max_lag: Maximum lag length for regression.

    Returns:
        Dictionary containing ADF test statistic, p-value, critical values, and stationarity boolean.
    """
    clean_series = series.dropna()
    res = adfuller(clean_series, maxlag=max_lag, autolag="AIC")
    adf_stat, p_value, used_lag, n_obs, crit_vals, icbest = res

    is_stationary = bool(p_value < 0.05)
    logger.info(
        f"ADF Test: Stat={adf_stat:.4f}, p-value={p_value:.4e}, "
        f"Stationary (alpha=0.05): {is_stationary}"
    )

    return {
        "adf_statistic": float(adf_stat),
        "p_value": float(p_value),
        "used_lag": int(used_lag),
        "n_obs": int(n_obs),
        "critical_values": {k: float(v) for k, v in crit_vals.items()},
        "is_stationary": is_stationary,
    }


def find_optimal_arima_order(
    series: pd.Series,
    p_range: range = range(0, 4),
    d_range: range = range(1, 2),
    q_range: range = range(0, 4),
) -> Tuple[Tuple[int, int, int], float, List[Dict[str, Any]]]:
    """Finds best ARIMA(p,d,q) order by minimizing AIC across candidates on training data.
    Never hardcodes orders; order is purely driven by statistical fit.

    Args:
        series: Training price series.
        p_range: Candidate AR orders.
        d_range: Candidate differencing orders (default 1 for non-stationary price).
        q_range: Candidate MA orders.

    Returns:
        Tuple of (best_order, best_aic, search_history).
    """
    logger.info("Initiating ARIMA order selection grid search (optimizing AIC)...")
    best_aic = float("inf")
    best_order = (1, 1, 1)
    history = []

    for d in d_range:
        for p in p_range:
            for q in q_range:
                if p == 0 and q == 0 and d == 0:
                    continue
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted = model.fit()
                    aic = float(fitted.aic)
                    history.append({"order": [p, d, q], "aic": aic})

                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                except Exception as e:
                    logger.debug(f"ARIMA({p},{d},{q}) failed estimation: {e}")
                    history.append({"order": [p, d, q], "error": str(e)})

    logger.info(f"Optimal ARIMA order selected: {best_order} with AIC: {best_aic:.2f}")
    return best_order, best_aic, history


class ARIMAForecaster:
    """End-to-End ARIMA Forecasting Engine."""

    def __init__(self, order: Optional[Tuple[int, int, int]] = None):
        self.order = order
        self.fitted_model_ = None
        self.diagnostics_ = {}

    def fit(self, train_series: pd.Series) -> "ARIMAForecaster":
        """Fits ARIMA model on training series. If order not specified, runs grid search.

        Args:
            train_series: Training price series.

        Returns:
            Self.
        """
        if self.order is None:
            best_order, best_aic, history = find_optimal_arima_order(train_series)
            self.order = best_order
            self.diagnostics_["grid_search_history"] = history
            self.diagnostics_["selected_aic"] = best_aic

        logger.info(f"Fitting ARIMA{self.order} on {len(train_series)} training observations...")
        model = ARIMA(train_series, order=self.order)
        self.fitted_model_ = model.fit()
        self.diagnostics_["aic"] = float(self.fitted_model_.aic)
        self.diagnostics_["bic"] = float(self.fitted_model_.bic)

        # Residual diagnostics
        residuals = self.fitted_model_.resid
        lb_test = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
        jb_stat, jb_pvalue, skew, kurt = jarque_bera(residuals)

        self.diagnostics_["residual_diagnostics"] = {
            "mean_residual": float(np.mean(residuals)),
            "std_residual": float(np.std(residuals)),
            "ljung_box_pvalues": {f"lag_{lag}": float(pval) for lag, pval in zip(lb_test.index, lb_test["lb_pvalue"])},
            "jarque_bera": {
                "statistic": float(jb_stat),
                "p_value": float(jb_pvalue),
                "skew": float(skew),
                "kurtosis": float(kurt),
            },
        }

        return self

    def forecast_rolling(
        self,
        history_series: pd.Series,
        test_series: pd.Series,
    ) -> np.ndarray:
        """Executes 1-step-ahead walk-forward forecasting over the test horizon.
        At each step t in test, model updates with actual observed price up to t-1
        to forecast price at t.

        Args:
            history_series: Training historical series.
            test_series: Out-of-sample test series.

        Returns:
            Array of 1-step-ahead forecasts for each test point.
        """
        logger.info(f"Running 1-step-ahead rolling out-of-sample forecast for {len(test_series)} test points...")
        history = list(history_series.values)
        predictions = []

        # To keep computation fast and accurate across hundreds of test days,
        # we refit or update state periodically or apply standard 1-step kalman update
        # Using fitted_model_.apply(full_series) produces exact 1-step-ahead out-of-sample forecasts!
        full_series = pd.concat([history_series, test_series]).reset_index(drop=True)
        fitted_full = self.fitted_model_.apply(full_series)
        all_pred = fitted_full.fittedvalues

        # Out-of-sample slice corresponds exactly to the test series indices
        test_pred = all_pred.iloc[len(history_series):].values
        return np.asarray(test_pred, dtype=float)

    def save(self, filepath: Path) -> None:
        """Saves forecaster state to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"order": self.order, "diagnostics": self.diagnostics_}, filepath)
        logger.info(f"ARIMA model state saved to {filepath}")


def plot_arima_results(
    dates: pd.Series,
    actual: np.ndarray,
    predicted: np.ndarray,
    residuals: np.ndarray,
    order: Tuple[int, int, int],
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> None:
    """Generates and saves publication-quality actual vs predicted and residual diagnostic plots."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Forecast vs Actual Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(dates, actual, label="Actual BTC Price", color="#1f77b4", linewidth=1.5)
    ax1.plot(dates, predicted, label=f"ARIMA{order} 1-Day Forecast", color="#ff7f0e", linewidth=1.2, linestyle="--")
    ax1.set_title(f"Bitcoin 1-Day Ahead Price Forecast: ARIMA{order}", fontsize=14, fontweight="bold", pad=10)
    ax1.set_ylabel("Price (USD)", fontsize=11)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Prediction error subplot
    error = actual - predicted
    ax2.fill_between(dates, error, color="#d62728", alpha=0.3, label="Forecast Error (Actual - Pred)")
    ax2.plot(dates, error, color="#d62728", linewidth=0.8)
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Error (USD)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Format x-axis dates nicely
    step = max(1, len(dates) // 8)
    ax2.set_xticks(dates.iloc[::step])
    ax2.set_xticklabels(dates.iloc[::step], rotation=30, ha="right")

    plt.tight_layout()
    forecast_plot_path = figures_dir / "arima_forecast.png"
    fig.savefig(forecast_plot_path, dpi=200)
    plt.close(fig)
    logger.info(f"Forecast plot saved to {forecast_plot_path}")

    # 2. Residual Diagnostics Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Time series of residuals
    axes[0, 0].plot(residuals, color="#2ca02c", linewidth=0.8)
    axes[0, 0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("Standardized Residuals", fontsize=11, fontweight="bold")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # Histogram and KDE
    axes[0, 1].hist(residuals, bins=40, density=True, alpha=0.6, color="#1f77b4", edgecolor="black")
    axes[0, 1].set_title("Residual Distribution vs Normal", fontsize=11, fontweight="bold")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # Autocorrelation of residuals
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals, lags=20, ax=axes[1, 0], alpha=0.05)
    axes[1, 0].set_title("Residual Autocorrelation (ACF)", fontsize=11, fontweight="bold")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Normal Q-Q Plot", fontsize=11, fontweight="bold")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    diag_plot_path = figures_dir / "arima_residual_diagnostics.png"
    fig.savefig(diag_plot_path, dpi=200)
    plt.close(fig)
    logger.info(f"Residual diagnostics plot saved to {diag_plot_path}")


def run_arima_pipeline(
    processed_data_path: Path = Path("data/processed/btc_daily.csv"),
    test_ratio: float = 0.2,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Runs complete ARIMA estimation, order selection, forecasting, and diagnostics pipeline.

    Args:
        processed_data_path: Path to processed daily CSV.
        test_ratio: Out-of-sample test fraction.
        results_dir: Results directory.
        figures_dir: Figures directory.
        models_dir: Models serialization directory.

    Returns:
        Tuple of (results_dict, predictions_dataframe).
    """
    logger.info("=== STEP 1: Loading Data & Testing Stationarity ===")
    df = pd.read_csv(processed_data_path)
    price_series = df["Close"]

    # ADF tests: Raw price vs First Difference vs Log Return
    adf_raw = check_stationarity(price_series)
    adf_diff = check_stationarity(price_series.diff().dropna())
    adf_log_ret = check_stationarity(np.log(price_series / price_series.shift(1)).dropna())

    stationarity_summary = {
        "raw_price": adf_raw,
        "first_difference": adf_diff,
        "log_returns": adf_log_ret,
    }

    # Chronological Split (strictly no shuffle)
    n_total = len(df)
    split_idx = int(n_total * (1.0 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    logger.info(
        f"Chronological Split for ARIMA: Train={len(train_df)} days ({train_df['Date'].iloc[0]} to {train_df['Date'].iloc[-1]}), "
        f"Test={len(test_df)} days ({test_df['Date'].iloc[0]} to {test_df['Date'].iloc[-1]})"
    )

    # STEP 2: Model Order Selection & Estimation
    forecaster = ARIMAForecaster()
    forecaster.fit(train_df["Close"])

    # STEP 3: Out-of-sample Forecast
    predictions = forecaster.forecast_rolling(train_df["Close"], test_df["Close"])

    # STEP 4: Evaluation Metrics
    actual_prices = test_df["Close"].values
    current_prices = test_df["Open"].values  # Or prior close
    metrics = calculate_regression_metrics(
        y_true=actual_prices,
        y_pred=predictions,
        current_prices=current_prices,
        is_return_target=False,
    )

    # STEP 5: Generate Residuals & Plots
    residuals = actual_prices - predictions
    normalized_residuals = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-10)

    plot_arima_results(
        dates=test_df["Date"],
        actual=actual_prices,
        predicted=predictions,
        residuals=normalized_residuals,
        order=forecaster.order,
        figures_dir=figures_dir,
    )

    # STEP 6: Save Artifacts
    pred_df = pd.DataFrame({
        "Date": test_df["Date"].values,
        "Actual_Close": actual_prices,
        "Predicted_Close": predictions,
        "Error": residuals,
        "Absolute_Error": np.abs(residuals),
        "Percentage_Error": np.abs(residuals / actual_prices) * 100.0,
    })

    results_payload = {
        "model_name": f"ARIMA{forecaster.order}",
        "order": list(forecaster.order),
        "evaluation_period": [str(test_df["Date"].iloc[0]), str(test_df["Date"].iloc[-1])],
        "n_train_samples": len(train_df),
        "n_test_samples": len(test_df),
        "stationarity_tests": stationarity_summary,
        "model_selection": {
            "selected_order": list(forecaster.order),
            "aic": forecaster.diagnostics_.get("aic"),
            "bic": forecaster.diagnostics_.get("bic"),
        },
        "residual_diagnostics": forecaster.diagnostics_.get("residual_diagnostics"),
        "test_metrics": metrics,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "arima_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    pred_path = results_dir / "arima_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    model_path = models_dir / "arima_model.joblib"
    forecaster.save(model_path)

    logger.info(f"ARIMA Results: MAE=${metrics['mae']:,.2f}, RMSE=${metrics['rmse']:,.2f}, MAPE={metrics['mape']:.2f}%, sMAPE={metrics['smape']:.2f}%")
    return results_payload, pred_df


if __name__ == "__main__":
    run_arima_pipeline()
