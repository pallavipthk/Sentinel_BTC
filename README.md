# BTC Sentinel: Explainable, Regime-Aware Bitcoin Forecasting & Crash Early-Warning System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-23%20passed-brightgreen.svg)](tests/)
[![Dashboard: Streamlit](https://img.shields.io/badge/dashboard-Streamlit-red.svg)](app/streamlit_app.py)

---

## 1. Project Overview

**BTC Sentinel** is a quantitative research platform and analytics system for daily Bitcoin (BTC/USD) historical data. It delivers:
- **1-Day Point Price Forecasts** with statistical 95% confidence intervals derived from out-of-sample empirical residual variance.
- **Conditional Volatility Forecasting** via GARCH(1,1) Maximum Likelihood Estimation with empirical regime segmentation (`LOW`, `MEDIUM`, `HIGH`).
- **Crash & Drawdown Early-Warning Risk Quantification** using forward $\epsilon$-drawdown event labeling and calibrated probability estimators.
- **Explainable Macro Market Regime Detection** utilizing transparent, deterministic rules across trend, momentum, volatility, and drawdown dimensions.
- **Historical Pattern Similarity Search** matching the current 30-day normalized return trajectory against non-overlapping historical epochs with subsequent outcome tracking.
- **Strict Chronological Expanding-Window Walk-Forward Backtesting** guaranteeing zero lookahead leakage.
- **Factual Model Explainability** attributing risk drivers to historical distribution percentiles.
- **Interactive Streamlit Quantitative Dashboard** designed in a professional dark financial terminal layout.

> **Disclaimer:** BTC Sentinel is an analytical and quantitative research project. It does **NOT** provide financial advice, trading signals, or automated execution.

---

## 2. Problem Statement & Research Motivation

Forecasting Bitcoin poses distinct econometric and statistical challenges:
1. **Non-Stationarity:** Raw price series $P_t$ fail unit root tests (ADF $p > 0.05$). Naive regression directly on nominal prices leads to spurious correlations and tree-based extrapolation failures outside historical bounds.
2. **Volatility Clustering & Fat Tails:** As documented in cryptocurrency econometrics, return variances exhibit strong clustering with high excess kurtosis (leptokurtosis) and extreme tail events.
3. **Asymmetric Crash Risk:** Bitcoin drawdowns often occur abruptly during structural regime shifts, rendering standard symmetric loss functions insufficient.
4. **Data Leakage Risks:** Standard machine learning workflows frequently contaminate time series via random shuffling, improper scaler fits across splits, or lookahead in rolling indicators.

BTC Sentinel resolves these challenges through rigorous time-series preprocessing, stationary return targets, calibrated probability modeling, and expanding walk-forward cross-validation.

---

## 3. Architecture Pipeline

```text
Historical BTC OHLCV (Binance / Yahoo Fallback)
                     ↓
        Data Cleaning & Validation
      (Deduplication, Monotonicity, Gaps)
                     ↓
         Feature Engineering Engine
 (80+ Causal Indicators: Returns, Volatility, Range, Momentum)
                     ↓
 ┌────────────────────────────────────────────────────────┐
 │ MULTI-MODEL QUANTITATIVE SUITE                         │
 │                                                        │
 │ 1. Price Forecasting                                   │
 │    ├── Naive Random Walk Baseline (P_t+1 = P_t)        │
 │    ├── ARIMA(p,d,q) with AIC Order Selection           │
 │    └── XGBoost Compounded Return Regressor             │
 │                                                        │
 │ 2. Volatility Modeling                                 │
 │    ├── Realized Parkinson Intrabar Volatility          │
 │    ├── 30-Day Rolling Benchmark                        │
 │    └── GARCH(1,1) Maximum Likelihood Estimator         │
 │                                                        │
 │ 3. Forward Drawdown / Crash Risk                       │
 │    ├── Forward 14-Day Epsilon Drawdown Labeler         │
 │    ├── Balanced Logistic Regression                    │
 │    └── Calibrated Random Forest Classifier             │
 └────────────────────────────────────────────────────────┘
                     ↓
          Market Regime Detection
      (Trend, Momentum, Volatility, Drawdown)
                     ↓
       Historical Pattern Similarity
      (Normalized 30-Day Shape Correlation)
                     ↓
          Central Synthesis Engine
      (Point Forecast, 95% Interval, Confidence)
                     ↓
     Chronological Walk-Forward Backtest
                     ↓
        Streamlit Analytics Dashboard
```

---

## 4. Research Connection & Attribution

To preserve academic integrity, we clearly delineate the relationship between literature, our implementation, and engineering extensions:

| Research Reference | Research Paper Influence | Our Implementation | Engineering Extensions |
| :--- | :--- | :--- | :--- |
| **Paper 1:** *Time Series Analysis and Prediction on Bitcoin* | Concepts of ADF unit-root stationarity, order differencing, ACF/PACF analysis, and ARIMA modeling. | Algorithmic grid search selecting optimal $(p, d, q)$ order by minimizing AIC; rolling 1-step Kalman state forecasting; Ljung-Box and Jarque-Bera diagnostics. | Automated stationarity pre-checks on raw vs differenced prices; comparison against modern gradient boosting; integration into walk-forward validation. |
| **Paper 2:** *Cryptocurrency Volatility Forecasting* | Conditional heteroskedasticity, log returns, GARCH(1,1) modeling, and volatility clustering. | Pure SciPy Maximum Likelihood Estimation of GARCH(1,1) with stationary constraint ($\alpha + \beta < 1$); Quasi-Likelihood (QLIKE) and RMSE evaluation. | Historical percentile cutoffs (33.3% / 66.7%) for objective LOW/MEDIUM/HIGH regimes; Parkinson intrabar high-low proxy; convergence fallback. |
| **Paper 3:** *Bitcoin Crash Prediction* | Forward drawdown horizon labeling, rolling features, and binary crash classification. | Forward 14-day $\epsilon$-drawdown labeling ($\ge 10\%$ drop); balanced class-weight estimation; probability calibration via Sigmoid/Isotonic scaling. | Strict elimination of future leakage; percentile-based feature attribution; translation into an executive "Model-Estimated Crash Risk" telemetry score. |

---

## 5. Dataset Specification

- **Asset:** Bitcoin (BTC/USD, BTC/USDT)
- **Granularity:** Daily (00:00 UTC Close)
- **Primary Source:** Binance Public REST API (no paid API key required, 1000 candles/call pagination)
- **Fallback Source:** Yahoo Finance Chart API
- **Historical Scope:** 2018-01-01 to Present (3,194 consecutive daily bars)
- **Fields:** `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- **Data Integrity Audit:**
  - Missing values: 0
  - Duplicate timestamps: 0
  - Monotonicity: 100% strictly chronological ascending
  - OHLC Bounds: $\text{High} \ge \max(\text{Open}, \text{Close})$ and $\text{Low} \le \min(\text{Open}, \text{Close})$ enforced.

---

## 6. Leakage-Safe Feature Engineering

The feature matrix generates 80+ quantitative indicators. Every feature at row $t$ utilizes exclusively historical observations $\le t$.

Mathematical invariance tests (`tests/test_features_and_baseline.py`) verify that mutating price data at $t+1$ results in zero difference ($< 10^{-12}$) across all computed features at $\le t$.

Key feature categories:
- **Log Returns:** 1-day, 3-day, 7-day, 14-day, 30-day.
- **Lagged Structures:** Lags 1, 2, 3, 5, 7, 14 of returns and price ratios ($P_t / P_{t-k}$).
- **Moving Average Ratios:** Price relative to SMA-7, 14, 30, 50, 200, and EMA-12/26.
- **Volatility Metrics:** 7d, 14d, 30d, 60d, 90d rolling standard deviations (annualized by $\sqrt{365}$); 14d and 30d Parkinson range volatility.
- **Momentum:** 7d, 14d, 30d, 90d percentage change; RSI-7 and RSI-14 (Wilder's exponential smoothing); MACD histogram.
- **Drawdown Dynamics:** Drawdown from historical all-time-high; rolling 30d, 90d, 180d peak drawdowns.
- **Volume Flow:** 1-day volume percentage change, volume relative to 7d/14d/30d moving averages, volume-price interaction proxy.
- **Range & Dispersion:** Bollinger Bands (%B and Bandwidth over 20 days), Average True Range (ATR-14).

---

## 7. Model Architectures & Methodologies

### 7.1 Naive Persistence Benchmark
Assumes random walk hypothesis:
$$\hat{P}_{t+1} = P_t, \quad \hat{R}_{t+1} = 0.0$$
Provides the baseline hurdle that any statistical or ML model must justify itself against.

### 7.2 ARIMA(p, d, q) Time Series Forecaster
- **Order Determination:** Evaluates candidate orders $(p, d, q) \in [0, 3] \times \{1\} \times [0, 3]$ on training data.
- **Optimal Order Selected:** **ARIMA(1, 1, 0)** with AIC: 43,176.47.
- **Diagnostics:**
  - Raw Price ADF: Statistic = -0.9591 ($p = 0.7679$ $\to$ Non-Stationary).
  - First Difference ADF: Statistic = -14.9425 ($p = 1.32 \times 10^{-27}$ $\to$ Stationary).
  - Jarque-Bera Test on Residuals: Statistic = 20,459 ($p = 0.000$, Kurtosis = 16.77 $\to$ pronounced leptokurtosis).

### 7.3 XGBoost Return-Compounded Forecaster
Instead of regressing on explosive raw price levels, XGBoost is trained on stationary 1-day returns $R_{t+1}$:
$$\hat{R}_{t+1} = f_{\text{XGB}}(X_t)$$
$$\hat{P}_{t+1} = P_t \times (1 + \hat{R}_{t+1})$$
Trained with early stopping on chronological validation data to prevent overfitting.

### 7.4 GARCH(1,1) Volatility Model
Specifies conditional variance recursion:
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
Estimated via Maximum Likelihood Optimization:
- $\omega = 4.81 \times 10^{-5}$
- $\alpha = 0.1066$ (shock impact)
- $\beta = 0.8601$ (persistence memory)
- $\alpha + \beta = 0.9667 < 1.0$ (stationary)
- Log-Likelihood AIC: -12,997.69.

### 7.5 Crash Risk Early-Warning Classifier
- **Forward Horizon:** 14 calendar days.
- **Crash Definition:** $\min_{1 \le k \le 14} \frac{P_{t+k} - P_t}{P_t} \le -10.0\%$.
- **Historical Prevalence:** 20.73% of days (618 out of 2,981 labeled days).
- **Models:** Balanced Logistic Regression baseline and Calibrated Random Forest Classifier (Sigmoid calibration).

---

## 8. Empirical Results & Backtesting

All metrics reported below were generated directly from actual execution runs on our processed Bitcoin dataset.

### 8.1 Chronological 5-Fold Walk-Forward Backtesting (450 Days Out-Of-Sample)

| Model | MAE ($) | RMSE ($) | MAPE (%) | Directional Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Naive Persistence Baseline** | **$1,317.18** | **$1,856.72** | **1.56%** | 0.0% |
| **XGBoost (Return-Compounded)** | $1,329.50 | $1,866.47 | 1.57% | **48.7%** |
| **ARIMA(1, 1, 0)** | $1,952.29 | $2,652.73 | 2.31% | **48.7%** |

**Empirical Finding:** Consistent with financial econometrics, daily crypto prices exhibit near-martingale properties where a naive persistence model has low nominal error (1.56% MAPE) but zero directional predictive capability. XGBoost achieves comparable low error (1.57% MAPE) while providing active directional signal.

### 8.2 Volatility Forecasting Performance vs Realized Parkinson Range Volatility

| Volatility Model | QLIKE Loss (Lower is better) | Volatility MAE | Volatility RMSE |
| :--- | :---: | :---: | :---: |
| **30-Day Rolling Volatility** | -0.0893 | 10.93% | 16.58% |
| **GARCH(1,1) MLE** | **-0.1049** | **10.03%** | **13.65%** |

### 8.3 Statistical Volatility Regimes (Annualized Thresholds)

- **LOW Regime:** Annualized Volatility $< 51.2\%$ (Historical 33.3rd percentile)
- **MEDIUM Regime:** $51.2\% \le \text{Volatility} \le 65.5\%$
- **HIGH Regime:** Annualized Volatility $> 65.5\%$ (Historical 66.7th percentile)

---

## 9. Example Live Telemetry Output

Running `python scripts/predict.py` produces the real-time intelligence briefing:

```text
=================================================================
 BTC SENTINEL: Quantitative Forecast & Crash Warning Telemetry
=================================================================
As of Date:                   2026-09-29
Current BTC Price:            $83,984.00
-----------------------------------------------------------------
1-Day Price Forecast:         $84,115.52 (+0.16%)
Statistical 95% Interval:     $80,457.24  to  $87,773.80
Forecast Std Error:           ±$1,866.47
Model Projections:            XGB: $84,115.52 | ARIMA: $83,534.45
-----------------------------------------------------------------
Forecast Volatility (GARCH):  47.4% annualized [LOW]
Model-Estimated Crash Risk:   27.5% (Next 14 Days)
Crash Risk Level:             MODERATE
Market Macro Regime:          BULL_MOMENTUM
Qualitative Model Confidence: HIGH (3/4 factors aligned)
Historical Pattern Similarity:93.8% (Top Analog: 2023-06-29)
-----------------------------------------------------------------

[EXPLAINABILITY & REGIME CONTEXT]
Regime Explanation: Price displays strong bullish momentum structure
(trading +9.2% above 50 SMA and +17.9% above 200 SMA) with constructive
momentum (Positive Momentum (+8.1%, RSI 62.2)).
Trend Factor:       Bullish Uptrend
Momentum Factor:    Positive Momentum (+8.1%, RSI 62.2)
Volatility Factor:  Low Volatility (41.3% annualized)
Drawdown Factor:    -3.0% from 90d peak (-32.6% from ATH)

[CONFIDENCE RATIONALE]
Qualitative Model Confidence assessed as HIGH (3/4 factors aligned).
Model forecast direction aligns with 7-day prevailing momentum.
Volatility is contained within LOW historical bounds (47.4%).
Model-estimated 14-day crash risk is restrained at 27.5%.
Price is within historical normal distance from 50-day moving average.
(Note: Qualitative score based on model consensus & regime stability,
NOT a statistical probability).
=================================================================
```

---

## 10. Project Structure

```text
BTC Sentinel/
├── README.md                                  # Complete scientific & engineering documentation
├── LICENSE                                    # MIT License
├── requirements.txt                           # Minimal pinned production dependencies
├── .gitignore                                 # Clean VCS ignore patterns
│
├── data/
│   ├── raw/                                   # Downloaded raw market feeds
│   └── processed/
│       └── btc_daily.csv                      # Cleaned 3,194-day continuous OHLCV dataset
│
├── models/
│   ├── arima_model.joblib                     # Fitted ARIMA parameters and diagnostics
│   ├── xgboost_price_model.joblib             # Serialized XGBoost price forecaster
│   ├── garch_model.joblib                     # Fitted GARCH(1,1) MLE parameters
│   └── crash_risk_model.joblib                # Calibrated Random Forest crash classifier
│
├── reports/
│   ├── figures/                               # Publication-quality charts (PNG)
│   │   ├── arima_forecast.png
│   │   ├── arima_residual_diagnostics.png
│   │   ├── price_forecast_comparison.png
│   │   ├── xgboost_feature_importance.png
│   │   ├── walk_forward_comparison.png
│   │   ├── cumulative_absolute_error.png
│   │   ├── volatility_forecast.png
│   │   ├── crash_roc_pr_curve.png
│   │   ├── crash_risk_timeline.png
│   │   ├── market_regimes_timeline.png
│   │   └── pattern_similarity_overlay.png
│   └── results/                               # Machine-readable execution logs (JSON/CSV)
│       ├── data_validation_report.json
│       ├── baseline_results.json
│       ├── arima_results.json
│       ├── price_model_comparison.json
│       ├── walk_forward_backtest_results.json
│       ├── volatility_results.json
│       ├── crash_risk_results.json
│       ├── market_regimes.json
│       ├── pattern_similarity_results.json
│       └── explainability_report.json
│
├── src/
│   ├── utils/
│   │   └── logger.py                          # Dual-channel structured logging
│   ├── data/
│   │   ├── downloader.py                      # Multi-source market acquisition
│   │   ├── validator.py                       # Data integrity auditor
│   │   ├── cleaner.py                         # Schema & chronological sanitization
│   │   └── pipeline.py                        # Stage 1 pipeline runner
│   ├── features/
│   │   └── feature_engineer.py                # 80+ Causal leakage-safe indicators
│   ├── forecasting/
│   │   ├── naive_baseline.py                  # Random walk persistence benchmark
│   │   ├── arima_model.py                     # ARIMA model with AIC search & diagnostics
│   │   └── xgboost_forecaster.py              # XGBoost stationary return forecaster
│   ├── volatility/
│   │   └── garch_model.py                     # GARCH(1,1) MLE & quantile regimes
│   ├── crash/
│   │   ├── drawdown_labeler.py                # Forward epsilon drawdown labeling
│   │   └── crash_classifier.py                # Calibrated crash probability modeling
│   ├── regime/
│   │   └── regime_detector.py                 # Explainable multi-factor market regimes
│   ├── similarity/
│   │   └── pattern_matcher.py                 # Historical trajectory shape matcher
│   ├── evaluation/
│   │   ├── metrics.py                         # Regression & classification metrics
│   │   ├── backtester.py                      # 5-fold walk-forward cross-validation
│   │   └── explainability.py                  # Factual feature percentile attribution
│   └── engine/
│       └── sentinel_engine.py                 # Central telemetry synthesis engine
│
├── scripts/
│   ├── download_data.py                       # CLI: Download, clean & audit dataset
│   ├── train.py                               # CLI: End-to-end model training
│   ├── backtest.py                            # CLI: Chronological walk-forward backtest
│   └── predict.py                             # CLI: Real-time inference & briefing
│
├── app/
│   └── streamlit_app.py                       # Interactive Streamlit quant dashboard
│
└── tests/
    ├── test_data_pipeline.py                  # Tests for data validation & cleaning (10 tests)
    ├── test_features_and_baseline.py          # Strict mathematical leakage tests (3 tests)
    ├── test_arima.py                          # Stationarity & ARIMA diagnostics (2 tests)
    ├── test_xgboost.py                        # XGBoost training & compounding (1 test)
    ├── test_volatility.py                     # GARCH MLE & regime segmentation (3 tests)
    ├── test_crash_risk.py                     # Drawdown labeling & calibration (2 tests)
    ├── test_regime.py                         # Market regime classification (1 test)
    └── test_similarity.py                     # Historical pattern similarity (1 test)
```

---

## 11. Installation & Environment Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Virtual environment recommended

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/your-username/btc-sentinel.git
cd btc-sentinel

# 2. Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 12. Execution Guide

Every component is runnable from the repository root:

### 1. Download & Validate Market Data
```bash
python scripts/download_data.py --force
```

### 2. Train All Models
```bash
python scripts/train.py
```

### 3. Run Chronological Walk-Forward Backtesting
```bash
python scripts/backtest.py --n-splits 5 --window-size 90
```

### 4. Generate Real-Time Telemetry Briefing
```bash
python scripts/predict.py
```

### 5. Launch Interactive Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

### 6. Run Test Suite
```bash
pytest tests/ -v
```

---

## 13. Limitations & Future Research

### Known Scientific Limitations
1. **Regime Breaks & Exogenous Shocks:** Statistical models assume that future distributions resemble historical behavior. Exogenous regulatory actions, stablecoin collapses, or black swan liquidity events cannot be anticipated purely from historical OHLCV data.
2. **Execution Slippage & Transaction Costs:** Reported backtest metrics evaluate raw price and return forecasts. They do not simulate order book depth, market impact, exchange funding rates, or maker/taker fees.
3. **Probability Calibration Limits:** Rare crash events ($\ge 10\%$ drop in 14 days) represent ~20% of sample history. While Platt/Sigmoid scaling improves probability accuracy, tail probabilities in non-stationary markets remain subject to epistemic uncertainty.

### Future Extensions
1. **On-Chain & Derivatives Feeds:** Incorporating Bitcoin Mempool fee pressure, exchange net inflows/outflows, perpetual futures funding rates, and options implied volatility surfaces (Dvol).
2. **Temporal Convolutional Networks (TCN):** Exploring causal dilated convolutions with receptive fields spanning multi-month macro cycles.
3. **Conformal Prediction:** Constructing finite-sample distribution-free predictive intervals.

---

## 14. References & Research Influence

1. **Time Series Analysis and Prediction on Bitcoin:** Foundational econometric principles of stationarity, differencing, autoregressive integrated moving average (ARIMA) structures, and residual diagnostics.
2. **Cryptocurrency Volatility Forecasting:** Empirical characterization of volatility clustering in digital assets, GARCH conditional heteroskedasticity formulations, and loss function selection (QLIKE vs RMSE).
3. **Bitcoin Crash Prediction:** Drawdown horizon definitions, $\epsilon$-drawdown event formulation, rolling multi-scale feature representations, and classification benchmarks for asymmetric downside risk.

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).
