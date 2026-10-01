"""CLI Entry point for BTC Sentinel Forecast & Risk Engine.
Outputs the complete executive intelligence briefing table:
    python scripts/predict.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.engine.sentinel_engine import run_sentinel_inference
from src.utils.logger import setup_logger

logger = setup_logger("script_predict")


def main() -> None:
    print("\n" + "=" * 65)
    print(" BTC SENTINEL: Quantitative Forecast & Crash Warning Telemetry")
    print("=" * 65)

    try:
        out = run_sentinel_inference()

        print("\n" + "*" * 65)
        print("                  EXECUTIVE BRIEFING REPORT")
        print("*" * 65)
        print(f"As of Date:                   {out.as_of_date}")
        print(f"Current BTC Price:            ${out.current_price:,.2f}")
        print("-" * 65)
        print(f"1-Day Price Forecast:         ${out.forecast_price_1d:,.2f} ({out.expected_return_1d_pct:+.2f}%)")
        print(f"Statistical 95% Interval:     ${out.forecast_interval_95[0]:,.2f}  to  ${out.forecast_interval_95[1]:,.2f}")
        print(f"Forecast Std Error:           ±${out.forecast_interval_error_std:,.2f}")
        print(f"Model Projections:            XGB: ${out.model_predictions['XGBoost']:,.2f} | ARIMA: ${out.model_predictions['ARIMA']:,.2f}")
        print("-" * 65)
        print(f"Forecast Volatility (GARCH):  {out.volatility_forecast_annualized_pct:.1f}% annualized [{out.volatility_regime}]")
        print(f"Model-Estimated Crash Risk:   {out.model_estimated_crash_risk_pct:.1f}% (Next 14 Days)")
        print(f"Crash Risk Level:             {out.crash_risk_level}")
        print(f"Market Macro Regime:          {out.market_regime}")
        print(f"Qualitative Model Confidence: {out.qualitative_confidence} ({out.confidence_score}/4)")
        print(f"Historical Pattern Similarity:{out.top_similarity_score_pct:.1f}% (Analog: {out.top_similar_date})")
        print("-" * 65)
        print("\n[EXPLAINABILITY & REGIME CONTEXT]")
        print(f"Regime Explanation: {out.regime_explanation}")
        print(f"Trend Factor:       {out.supporting_factors['trend']}")
        print(f"Momentum Factor:    {out.supporting_factors['momentum']}")
        print(f"Volatility Factor:  {out.supporting_factors['volatility']}")
        print(f"Drawdown Factor:    {out.supporting_factors['drawdown']}")
        print(f"\n[CONFIDENCE RATIONALE]")
        print(f"{out.confidence_rationale}")
        print(f"\n[RISK QUALIFICATION]")
        print(f"{out.crash_qualification}")
        print("=" * 65 + "\n")

    except Exception as e:
        logger.exception(f"Inference execution failed: {e}")
        print(f"\n[ERROR] Inference failed: {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
