"""Explainability and Feature Attribution Engine for BTC Sentinel.
Generates factual local and global explanations for tree-based price forecasting
and crash risk models based strictly on verifiable mathematical model weights.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.features.feature_engineer import calculate_features, get_feature_columns
from src.forecasting.xgboost_forecaster import XGBoostPriceForecaster
from src.utils.logger import setup_logger

logger = setup_logger("explainability")

DEFAULT_RESULTS_DIR = Path("reports/results")
DEFAULT_MODELS_DIR = Path("models")


def generate_factual_risk_explanation(
    current_features: pd.Series,
    historical_features: pd.DataFrame,
    crash_importances: Dict[str, float],
) -> Dict[str, Any]:
    """Generates factual, evidence-based explanations for model-estimated crash risk.

    Compares current market indicators against historical distributions for the top risk drivers.
    """
    driver_explanations = []

    # Analyze top 5 crash risk drivers
    top_drivers = list(crash_importances.items())[:5]

    for feat_name, imp_weight in top_drivers:
        if feat_name not in current_features:
            continue

        curr_val = float(current_features[feat_name])
        hist_col = historical_features[feat_name].dropna()
        percentile = float((hist_col < curr_val).mean() * 100.0)

        # Build precise, factual statement
        if "volatility" in feat_name or "std" in feat_name:
            impact_desc = "elevating" if percentile > 65 else "dampening"
            statement = (
                f"{feat_name}: Current value is {curr_val * 100:.1f}% annualized "
                f"({percentile:.1f}th historical percentile), {impact_desc} forward crash risk."
            )
        elif "drawdown" in feat_name:
            impact_desc = "elevating" if percentile < 35 else "contained"
            statement = (
                f"{feat_name}: Current drawdown is {curr_val * 100:.1f}% "
                f"({percentile:.1f}th historical percentile), risk pressure is {impact_desc}."
            )
        elif "momentum" in feat_name or "return" in feat_name:
            trend_dir = "positive" if curr_val > 0 else "negative"
            statement = (
                f"{feat_name}: Momentum is {trend_dir} at {curr_val * 100:+.1f}% "
                f"({percentile:.1f}th percentile)."
            )
        elif "rsi" in feat_name:
            condition = "overbought (>70)" if curr_val > 70 else ("oversold (<30)" if curr_val < 30 else "neutral (30-70)")
            statement = (
                f"{feat_name}: RSI stands at {curr_val:.1f} ({condition}), at the {percentile:.1f}th historical percentile."
            )
        else:
            statement = f"{feat_name}: Valued at {curr_val:.3f} (historical percentile: {percentile:.1f}%)."

        driver_explanations.append({
            "feature": feat_name,
            "importance_weight": round(imp_weight, 4),
            "current_value": round(curr_val, 4),
            "historical_percentile": round(percentile, 1),
            "factual_statement": statement,
        })

    return {
        "methodology": "Model-based global Gini/Gain feature attribution cross-referenced with empirical percentile ranking.",
        "top_drivers": driver_explanations,
    }


def run_explainability_pipeline(
    processed_data_path: Path = Path("data/processed/btc_daily.csv"),
    models_dir: Path = DEFAULT_MODELS_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> Dict[str, Any]:
    """Extracts and serializes verifiable model explanations."""
    logger.info("Generating factual model explainability reports...")
    df = pd.read_csv(processed_data_path)
    df_feats = calculate_features(df, include_target=False, drop_na=True)
    feature_cols = get_feature_columns(df_feats)

    # 1. Load price model feature importances
    price_importances = {}
    xgb_path = models_dir / "xgboost_price_model.joblib"
    if xgb_path.exists():
        xgb = XGBoostPriceForecaster.load(xgb_path)
        price_importances = xgb.feature_importances_

    # 2. Load crash model feature importances
    crash_importances = {}
    crash_results_file = results_dir / "crash_risk_results.json"
    if crash_results_file.exists():
        with open(crash_results_file, "r") as f:
            c_data = json.load(f)
            crash_importances = dict(c_data.get("top_crash_drivers", []))

    # 3. Generate factual risk explanations for latest day
    latest_row = df_feats.iloc[-1]
    risk_explanation = generate_factual_risk_explanation(
        current_features=latest_row,
        historical_features=df_feats[feature_cols],
        crash_importances=crash_importances,
    )

    report = {
        "as_of_date": str(df["Date"].iloc[-1]),
        "current_price": float(df["Close"].iloc[-1]),
        "price_forecasting_top_features": list(price_importances.items())[:15],
        "crash_risk_top_features": list(crash_importances.items())[:15],
        "current_risk_driver_breakdown": risk_explanation,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "explainability_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Explainability report saved to {report_path}")
    return report
