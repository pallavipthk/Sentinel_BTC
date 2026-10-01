"""GARCH(1,1) and Rolling Volatility Forecasting Engine.
Implements:
1. Maximum Likelihood Estimation (MLE) of GARCH(1,1) with stationarity constraints.
2. Robust convergence monitoring and graceful fallback handling.
3. 1-day and multi-day volatility projections.
4. Statistical quantile-based volatility regime segmentation (LOW, MEDIUM, HIGH).
5. QLIKE and Volatility RMSE/MAE evaluation against realized Parkinson volatility.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.utils.logger import setup_logger

logger = setup_logger("volatility_forecaster")

DEFAULT_RESULTS_DIR = Path("reports/results")
DEFAULT_FIGURES_DIR = Path("reports/figures")
DEFAULT_MODELS_DIR = Path("models")


class GARCHModel:
    """GARCH(1,1) model fitted via Maximum Likelihood Estimation (Gaussian likelihood).
    sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
    """

    def __init__(self):
        self.omega: float = 1e-5
        self.alpha: float = 0.1
        self.beta: float = 0.85
        self.mu: float = 0.0
        self.converged: bool = False
        self.convergence_message: str = "Uninitialized"
        self.log_likelihood_: float = 0.0
        self.aic_: float = 0.0
        self.bic_: float = 0.0

    def _negative_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """Computes negative log-likelihood of GARCH(1,1) given returns."""
        mu, omega, alpha, beta = params
        eps = returns - mu
        n = len(returns)

        # Variance recursion
        variances = np.zeros(n)
        variances[0] = np.var(returns)

        for t in range(1, n):
            variances[t] = omega + (alpha * (eps[t - 1] ** 2)) + (beta * variances[t - 1])

        # Floor variance to prevent log(0) or division by 0
        variances = np.clip(variances, 1e-12, 1e6)

        # Gaussian log-likelihood
        llh = -0.5 * np.sum(np.log(2.0 * np.pi) + np.log(variances) + ((eps ** 2) / variances))
        return -llh  # Return negative for minimization

    def fit(self, returns: np.ndarray) -> "GARCHModel":
        """Fits GARCH(1,1) parameters via constrained optimization."""
        rets = np.asarray(returns, dtype=float)
        sample_var = float(np.var(rets))
        sample_mean = float(np.mean(rets))

        # Initial parameter estimates
        init_params = [sample_mean, sample_var * 0.05, 0.10, 0.85]

        # Parameter bounds: omega > 0, alpha >= 0, beta >= 0
        bounds = [
            (-1.0, 1.0),        # mu
            (1e-8, sample_var), # omega
            (0.001, 0.40),      # alpha
            (0.50, 0.999),      # beta
        ]

        # Stationarity constraint: alpha + beta < 1.0
        def stationarity_constraint(params):
            return 0.9999 - (params[2] + params[3])

        constraints = [{"type": "ineq", "fun": stationarity_constraint}]

        try:
            res = minimize(
                fun=self._negative_log_likelihood,
                x0=init_params,
                args=(rets,),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 300, "ftol": 1e-7, "disp": False},
            )

            if res.success and (res.x[2] + res.x[3] < 1.0):
                self.mu, self.omega, self.alpha, self.beta = res.x
                self.converged = True
                self.convergence_message = "Optimization converged successfully."
                self.log_likelihood_ = -float(res.fun)
                n = len(rets)
                k = 4
                self.aic_ = float(2 * k - 2 * self.log_likelihood_)
                self.bic_ = float(k * np.log(n) - 2 * self.log_likelihood_)
                logger.info(
                    f"GARCH(1,1) MLE Converged: omega={self.omega:.2e}, alpha={self.alpha:.4f}, "
                    f"beta={self.beta:.4f}, persistence(alpha+beta)={self.alpha + self.beta:.4f}, AIC={self.aic_:.2f}"
                )
            else:
                self.converged = False
                self.convergence_message = f"Optimization failed or violated stationarity: {res.message}"
                logger.warning(f"GARCH fitting warning: {self.convergence_message}. Utilizing robust econometric defaults.")
                # Robust fallback to standard financial stylized values
                self.mu = sample_mean
                self.alpha = 0.08
                self.beta = 0.90
                self.omega = sample_var * (1.0 - (self.alpha + self.beta))
        except Exception as e:
            self.converged = False
            self.convergence_message = f"Exception during GARCH estimation: {e}"
            logger.error(f"GARCH estimation exception: {e}")
            self.mu = sample_mean
            self.alpha = 0.08
            self.beta = 0.90
            self.omega = sample_var * (1.0 - (self.alpha + self.beta))

        return self

    def filter_conditional_volatility(self, returns: np.ndarray) -> np.ndarray:
        """Filters historical conditional volatility sequence."""
        rets = np.asarray(returns, dtype=float)
        eps = rets - self.mu
        n = len(rets)
        variances = np.zeros(n)
        variances[0] = np.var(rets)

        for t in range(1, n):
            variances[t] = self.omega + (self.alpha * (eps[t - 1] ** 2)) + (self.beta * variances[t - 1])

        # Daily standard deviations converted to annualized volatility (* sqrt(365))
        annualized_vols = np.sqrt(np.clip(variances, 1e-12, 1e6)) * np.sqrt(365.0)
        return annualized_vols

    def forecast_1step_ahead(self, last_return: float, last_variance: float) -> float:
        """Forecasts annualized conditional volatility for t+1."""
        eps = last_return - self.mu
        next_var = self.omega + (self.alpha * (eps ** 2)) + (self.beta * last_variance)
        return float(np.sqrt(max(1e-12, next_var)) * np.sqrt(365.0))


def compute_volatility_regimes(
    historical_vol: pd.Series,
    low_percentile: float = 33.3,
    high_percentile: float = 66.7,
) -> Tuple[pd.Series, Dict[str, float]]:
    """Segments historical volatility into statistical regimes based on empirical quantiles.

    Threshold methodology:
    - LOW: volatility < 33.3rd percentile of historical distribution.
    - MEDIUM: 33.3rd percentile <= volatility <= 66.7th percentile.
    - HIGH: volatility > 66.7th percentile.

    Args:
        historical_vol: Series of annualized volatility values.
        low_percentile: Percentile cutoff for Low regime.
        high_percentile: Percentile cutoff for High regime.

    Returns:
        Tuple of (regime_labels_series, threshold_dictionary).
    """
    clean_vol = historical_vol.dropna()
    low_thresh = float(np.percentile(clean_vol, low_percentile))
    high_thresh = float(np.percentile(clean_vol, high_percentile))

    thresholds = {
        "low_cutoff": low_thresh,
        "high_cutoff": high_thresh,
        "low_percentile": low_percentile,
        "high_percentile": high_percentile,
        "historical_median": float(np.median(clean_vol)),
        "historical_mean": float(np.mean(clean_vol)),
    }

    def assign_regime(val: float) -> str:
        if np.isnan(val):
            return "UNKNOWN"
        if val < low_thresh:
            return "LOW"
        elif val <= high_thresh:
            return "MEDIUM"
        else:
            return "HIGH"

    regimes = historical_vol.apply(assign_regime)
    return regimes, thresholds


def evaluate_volatility_forecasts(
    actual_realized_vol: np.ndarray,
    forecast_vol: np.ndarray,
) -> Dict[str, float]:
    """Computes volatility evaluation metrics including QLIKE, RMSE, and MAE.
    QLIKE = mean( ln(sigma_hat^2) + sigma_actual^2 / sigma_hat^2 )
    """
    y_true = np.asarray(actual_realized_vol, dtype=float)
    y_pred = np.asarray(forecast_vol, dtype=float)

    # Convert to variance
    var_true = np.clip(y_true ** 2, 1e-8, 1e8)
    var_pred = np.clip(y_pred ** 2, 1e-8, 1e8)

    qlike = float(np.mean(np.log(var_pred) + (var_true / var_pred)))
    vol_mae = float(np.mean(np.abs(y_true - y_pred)))
    vol_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return {
        "qlike": qlike,
        "volatility_mae": vol_mae,
        "volatility_rmse": vol_rmse,
    }


def plot_volatility_forecasts(
    dates: pd.Series,
    realized_vol: np.ndarray,
    rolling_vol: np.ndarray,
    garch_vol: np.ndarray,
    thresholds: Dict[str, float],
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> None:
    """Generates visualization of historical, rolling, and GARCH volatility with regime boundaries."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(dates, realized_vol * 100.0, label="Realized Parkinson Volatility", color="#9ca3af", alpha=0.6, linewidth=1.0)
    ax.plot(dates, rolling_vol * 100.0, label="30-Day Rolling Volatility", color="#3b82f6", linewidth=1.5)
    ax.plot(dates, garch_vol * 100.0, label="GARCH(1,1) Conditional Volatility", color="#dc2626", linewidth=1.6)

    # Regime bands
    low_c = thresholds["low_cutoff"] * 100.0
    high_c = thresholds["high_cutoff"] * 100.0
    ax.axhline(low_c, color="#10b981", linestyle="--", linewidth=1.2, label=f"Low/Med Threshold ({low_c:.1f}%)")
    ax.axhline(high_c, color="#f59e0b", linestyle="--", linewidth=1.2, label=f"Med/High Threshold ({high_c:.1f}%)")

    ax.set_title("Bitcoin Volatility Dynamics & GARCH(1,1) Conditional Forecasts (Annualized %)", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Annualized Volatility (%)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)

    step = max(1, len(dates) // 8)
    ax.set_xticks(dates.iloc[::step])
    ax.set_xticklabels(dates.iloc[::step], rotation=30, ha="right")

    plt.tight_layout()
    plot_path = figures_dir / "volatility_forecast.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    logger.info(f"Volatility forecast plot saved to {plot_path}")


def run_volatility_pipeline(
    processed_data_path: Path = Path("data/processed/btc_daily.csv"),
    results_dir: Path = DEFAULT_RESULTS_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Runs complete volatility estimation, regime segmentation, and evaluation."""
    logger.info("=== STEP 1: Preparing Volatility Measures ===")
    df = pd.read_csv(processed_data_path)

    # Compute daily log returns
    close = df["Close"]
    log_returns = np.log(close / close.shift(1)).fillna(0.0)

    # Realized Parkinson Volatility (proxy for true volatility)
    sqrt_365 = np.sqrt(365.0)
    log_hl = np.log(df["High"] / df["Low"])
    parkinson_factor = 1.0 / (4.0 * np.log(2.0))
    realized_vol_daily = np.sqrt(parkinson_factor * (log_hl ** 2).rolling(window=14, min_periods=14).mean()) * sqrt_365
    realized_vol_daily = realized_vol_daily.bfill()

    # 30-day Rolling Volatility Benchmark
    rolling_vol_30d = (log_returns.rolling(window=30, min_periods=30).std() * sqrt_365).bfill()

    # STEP 2: Fit GARCH(1,1) Model
    logger.info("Fitting GARCH(1,1) model on historical log returns...")
    garch = GARCHModel()
    garch.fit(log_returns.values)

    # Filter full sequence conditional volatility
    garch_conditional_vol = garch.filter_conditional_volatility(log_returns.values)

    # STEP 3: Regime Segmentation
    regimes, thresholds = compute_volatility_regimes(pd.Series(garch_conditional_vol))

    # STEP 4: Evaluation Metrics against Realized Volatility
    eval_slice = slice(30, None)  # exclude initial warmup
    rolling_metrics = evaluate_volatility_forecasts(realized_vol_daily.iloc[eval_slice].values, rolling_vol_30d.iloc[eval_slice].values)
    garch_metrics = evaluate_volatility_forecasts(realized_vol_daily.iloc[eval_slice].values, garch_conditional_vol[eval_slice])

    # Latest forecast
    latest_return = float(log_returns.iloc[-1])
    latest_variance = float((garch_conditional_vol[-1] / sqrt_365) ** 2)
    next_day_forecast_vol = garch.forecast_1step_ahead(latest_return, latest_variance)
    current_regime = "HIGH" if next_day_forecast_vol > thresholds["high_cutoff"] else ("LOW" if next_day_forecast_vol < thresholds["low_cutoff"] else "MEDIUM")

    logger.info(f"Current Bitcoin Volatility Forecast: {next_day_forecast_vol * 100.0:.2f}% | Current Regime: {current_regime}")

    # STEP 5: Plotting
    plot_volatility_forecasts(
        dates=df["Date"],
        realized_vol=realized_vol_daily.values,
        rolling_vol=rolling_vol_30d.values,
        garch_vol=garch_conditional_vol,
        thresholds=thresholds,
        figures_dir=figures_dir,
    )

    # STEP 6: Artifact Persistence
    vol_df = pd.DataFrame({
        "Date": df["Date"].values,
        "Close": df["Close"].values,
        "Log_Return": log_returns.values,
        "Realized_Parkinson_Vol": realized_vol_daily.values,
        "Rolling_30d_Vol": rolling_vol_30d.values,
        "GARCH_Conditional_Vol": garch_conditional_vol,
        "Volatility_Regime": regimes.values,
    })

    results_dir.mkdir(parents=True, exist_ok=True)
    vol_csv_path = results_dir / "volatility_forecasts.csv"
    vol_df.to_csv(vol_csv_path, index=False)

    summary = {
        "garch_parameters": {
            "omega": garch.omega,
            "alpha": garch.alpha,
            "beta": garch.beta,
            "persistence": garch.alpha + garch.beta,
            "converged": garch.converged,
            "message": garch.convergence_message,
            "aic": garch.aic_,
            "bic": garch.bic_,
        },
        "regime_thresholds_annualized": thresholds,
        "current_status": {
            "latest_date": str(df["Date"].iloc[-1]),
            "latest_price": float(df["Close"].iloc[-1]),
            "next_day_forecast_volatility_annualized": float(next_day_forecast_vol),
            "current_volatility_regime": current_regime,
        },
        "model_evaluation": {
            "Rolling_30d": rolling_metrics,
            "GARCH_1_1": garch_metrics,
        },
    }

    summary_path = results_dir / "volatility_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(garch, models_dir / "garch_model.joblib")

    logger.info(f"Volatility Pipeline Summary: GARCH QLIKE={garch_metrics['qlike']:.4f}, Vol MAE={garch_metrics['volatility_mae']*100:.2f}%")
    return summary, vol_df


if __name__ == "__main__":
    run_volatility_pipeline()
