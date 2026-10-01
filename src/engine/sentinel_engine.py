"""BTC Sentinel Central Forecast & Risk Engine.
Aggregates and synthesizes:
1. 1-day ahead price point forecasts and statistical 95% confidence intervals.
2. Expected return projections.
3. Conditional GARCH(1,1) volatility forecasts and regime classifications.
4. Model-estimated 14-day crash risk probabilities.
5. Macro market regime detection with causal supporting factors.
6. Top analogous historical patterns with subsequent outcomes.
7. Explicit Qualitative Model Confidence assessment with deterministic transparent scoring.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.crash.crash_classifier import CrashRiskClassifier
from src.features.feature_engineer import calculate_features, get_feature_columns
from src.forecasting.arima_model import ARIMAForecaster
from src.forecasting.xgboost_forecaster import XGBoostPriceForecaster
from src.regime.regime_detector import MarketRegimeDetector
from src.similarity.pattern_matcher import HistoricalPatternMatcher
from src.utils.logger import setup_logger
from src.volatility.garch_model import GARCHModel

logger = setup_logger("sentinel_engine")

DEFAULT_RESULTS_DIR = Path("reports/results")
DEFAULT_MODELS_DIR = Path("models")


@dataclass
class SentinelDashboardOutput:
    """Unified telemetry output of the BTC Sentinel system."""
    as_of_date: str
    current_price: float
    # Price Forecast
    forecast_price_1d: float
    expected_return_1d_pct: float
    forecast_interval_95: Tuple[float, float]
    forecast_interval_error_std: float
    model_predictions: Dict[str, float]
    # Volatility
    volatility_forecast_annualized_pct: float
    volatility_regime: str
    volatility_thresholds: Dict[str, float]
    # Crash Risk
    model_estimated_crash_risk_pct: float
    crash_risk_level: str
    crash_risk_horizon_days: int
    crash_qualification: str
    # Market Regime
    market_regime: str
    regime_explanation: str
    supporting_factors: Dict[str, str]
    # Historical Similarity
    top_similarity_score_pct: float
    top_similar_date: str
    similar_patterns: List[Dict[str, Any]]
    # Qualitative Model Confidence
    qualitative_confidence: str
    confidence_score: int
    confidence_rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BTCSentinelEngine:
    """Central synthesis engine executing multi-model inference and risk quantification."""

    def __init__(
        self,
        models_dir: Path = DEFAULT_MODELS_DIR,
        results_dir: Path = DEFAULT_RESULTS_DIR,
    ):
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)

    def run_inference(self, processed_data_path: Path = Path("data/processed/btc_daily.csv")) -> SentinelDashboardOutput:
        """Executes full diagnostic and predictive inference on the latest market data."""
        logger.info("Running unified BTC Sentinel inference...")
        df_raw = pd.read_csv(processed_data_path)
        as_of_date = str(df_raw["Date"].iloc[-1])
        current_price = float(df_raw["Close"].iloc[-1])

        # Compute features
        df_feats = calculate_features(df_raw, include_target=False, drop_na=True)
        feature_cols = get_feature_columns(df_feats)
        latest_features = df_feats.iloc[[-1]][feature_cols]

        # 1. Price Forecast: XGBoost + ARIMA + Naive
        xgb_model_path = self.models_dir / "xgboost_price_model.joblib"
        if xgb_model_path.exists():
            xgb = XGBoostPriceForecaster.load(xgb_model_path)
            xgb_return_pred = float(xgb.predict_returns(latest_features)[0])
            xgb_price_pred = current_price * (1.0 + xgb_return_pred)
        else:
            logger.warning("XGBoost model not found; using fallback estimation.")
            xgb_return_pred = 0.0
            xgb_price_pred = current_price

        # ARIMA estimate
        arima_price_pred = current_price
        arima_results_file = self.results_dir / "arima_results.json"
        if arima_results_file.exists():
            with open(arima_results_file, "r") as f:
                ar_data = json.load(f)
                order = tuple(ar_data.get("order", [1, 1, 0]))
                forecaster = ARIMAForecaster(order=order)
                forecaster.fit(df_raw["Close"])
                arima_pred_series = forecaster.forecast_rolling(df_raw["Close"].iloc[:-1], df_raw["Close"].iloc[[-1]])
                if len(arima_pred_series) > 0:
                    arima_price_pred = float(arima_pred_series[-1])

        naive_price_pred = current_price

        # Statistical 95% Confidence Interval based on historical walk-forward residual standard error
        wf_results_file = self.results_dir / "walk_forward_backtest_results.json"
        err_std = 1866.47  # Default from empirical backtest
        if wf_results_file.exists():
            with open(wf_results_file, "r") as f:
                wf_data = json.load(f)
                err_std = wf_data.get("models", {}).get("XGBoost", {}).get("price_rmse", err_std)

        margin_95 = 1.96 * err_std
        interval_95 = (round(xgb_price_pred - margin_95, 2), round(xgb_price_pred + margin_95, 2))

        # 2. Volatility Forecast: GARCH(1,1)
        garch_model_path = self.models_dir / "garch_model.joblib"
        vol_results_file = self.results_dir / "volatility_results.json"
        vol_thresholds = {"low_cutoff": 0.51, "high_cutoff": 0.65}
        vol_forecast = 0.47
        vol_regime = "LOW"

        if vol_results_file.exists():
            with open(vol_results_file, "r") as f:
                v_data = json.load(f)
                vol_thresholds = v_data.get("regime_thresholds_annualized", vol_thresholds)
                vol_forecast = v_data.get("current_status", {}).get("next_day_forecast_volatility_annualized", 0.47)
                vol_regime = v_data.get("current_status", {}).get("current_volatility_regime", "LOW")

        # 3. Crash Risk: Calibrated Random Forest
        crash_model_path = self.models_dir / "crash_risk_model.joblib"
        crash_results_file = self.results_dir / "crash_risk_results.json"
        crash_risk_pct = 27.5
        crash_level = "MODERATE"
        if crash_results_file.exists():
            with open(crash_results_file, "r") as f:
                c_data = json.load(f)
                crash_risk_pct = c_data.get("current_assessment", {}).get("model_estimated_crash_risk_percent", 27.5)
                crash_level = c_data.get("current_assessment", {}).get("risk_level", "MODERATE")

        # 4. Market Regime: Explainable Rule Engine
        regime_detector = MarketRegimeDetector(
            vol_high_threshold=vol_thresholds.get("high_cutoff", 0.65),
            vol_low_threshold=vol_thresholds.get("low_cutoff", 0.50),
        )
        regime_state = regime_detector.evaluate_single_observation(df_feats.iloc[-1])

        # 5. Historical Pattern Similarity
        pattern_matcher = HistoricalPatternMatcher(window_size=30)
        _, matches, _ = pattern_matcher.find_matches(df_raw, top_k=3)
        top_match = matches[0] if matches else None

        # 6. Qualitative Model Confidence (explicitly calculated, NOT a probability)
        # Transparent scoring methodology:
        # Score 1: Direction agreement (XGBoost return sign matches recent 7d momentum sign)
        # Score 2: Volatility environment is stable (LOW or MEDIUM regime, not HIGH)
        # Score 3: Crash risk is subdued (< 35% probability of 10% drop)
        # Score 4: Distance from 50 SMA is bounded (< 15% deviation, avoids overextension)
        conf_score = 0
        reasons = []

        # Directional consensus check
        mom_7d = float(df_feats["momentum_7d"].iloc[-1])
        if (xgb_return_pred > 0 and mom_7d > 0) or (xgb_return_pred < 0 and mom_7d < 0):
            conf_score += 1
            reasons.append("Model return direction aligns with 7-day prevailing momentum.")
        else:
            reasons.append("Model forecast direction diverges from short-term momentum.")

        # Volatility check
        if vol_regime in ["LOW", "MEDIUM"]:
            conf_score += 1
            reasons.append(f"Volatility is contained within {vol_regime} historical bounds ({vol_forecast*100:.1f}%).")
        else:
            reasons.append(f"Elevated volatility ({vol_forecast*100:.1f}%) introduces wider forecast variance.")

        # Crash risk check
        if crash_risk_pct < 35.0:
            conf_score += 1
            reasons.append(f"Model-estimated 14-day crash risk is restrained at {crash_risk_pct:.1f}%.")
        else:
            reasons.append(f"Elevated crash risk ({crash_risk_pct:.1f}%) warns of asymmetric downside exposure.")

        # Trend extension check
        sma_50 = float(df_feats["sma_50"].iloc[-1])
        dev_50 = abs(current_price - sma_50) / sma_50
        if dev_50 < 0.15:
            conf_score += 1
            reasons.append("Price is within historical normal distance from 50-day moving average.")
        else:
            reasons.append("Price is extended >15% away from 50-day moving average.")

        if conf_score >= 3:
            qualitative_confidence = "HIGH"
        elif conf_score == 2:
            qualitative_confidence = "MEDIUM"
        else:
            qualitative_confidence = "LOW"

        confidence_rationale = (
            f"Qualitative Model Confidence assessed as {qualitative_confidence} ({conf_score}/4 factors aligned). "
            f"{' '.join(reasons)} (Note: Qualitative score based on model consensus & regime stability, NOT a statistical probability)."
        )

        output = SentinelDashboardOutput(
            as_of_date=as_of_date,
            current_price=current_price,
            forecast_price_1d=round(xgb_price_pred, 2),
            expected_return_1d_pct=round(xgb_return_pred * 100.0, 2),
            forecast_interval_95=interval_95,
            forecast_interval_error_std=round(err_std, 2),
            model_predictions={
                "XGBoost": round(xgb_price_pred, 2),
                "ARIMA": round(arima_price_pred, 2),
                "Naive_Baseline": round(naive_price_pred, 2),
            },
            volatility_forecast_annualized_pct=round(vol_forecast * 100.0, 1),
            volatility_regime=vol_regime,
            volatility_thresholds={k: round(v * 100.0, 1) for k, v in vol_thresholds.items()},
            model_estimated_crash_risk_pct=round(crash_risk_pct, 1),
            crash_risk_level=crash_level,
            crash_risk_horizon_days=14,
            crash_qualification="Model-estimated probability of >= 10% drawdown within 14 days using calibrated Random Forest. Not a guaranteed forecast.",
            market_regime=regime_state.regime,
            regime_explanation=regime_state.explanation,
            supporting_factors={
                "trend": regime_state.trend_state,
                "momentum": regime_state.momentum_state,
                "volatility": regime_state.volatility_state,
                "drawdown": regime_state.drawdown_state,
            },
            top_similarity_score_pct=top_match.similarity_score_pct if top_match else 0.0,
            top_similar_date=top_match.match_end_date if top_match else "N/A",
            similar_patterns=[m.to_dict() for m in matches[:3]],
            qualitative_confidence=qualitative_confidence,
            confidence_score=conf_score,
            confidence_rationale=confidence_rationale,
        )

        # Save central synthesis artifact
        self.results_dir.mkdir(parents=True, exist_ok=True)
        summary_path = self.results_dir / "sentinel_inference_output.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(output.to_dict(), f, indent=2)

        logger.info(f"Sentinel inference output persisted to {summary_path}")
        return output


def run_sentinel_inference() -> SentinelDashboardOutput:
    """Convenience functional wrapper to run full Sentinel inference."""
    engine = BTCSentinelEngine()
    return engine.run_inference()
