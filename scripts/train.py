"""CLI entry point to train all quantitative and machine learning models:
    python scripts/train.py [--force-download]
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.crash.crash_classifier import run_crash_risk_pipeline
from src.data.pipeline import run_data_pipeline
from src.forecasting.arima_model import run_arima_pipeline
from src.forecasting.naive_baseline import run_baseline_evaluation
from src.forecasting.xgboost_forecaster import run_xgboost_price_pipeline
from src.regime.regime_detector import run_regime_detection_pipeline
from src.similarity.pattern_matcher import run_pattern_similarity_pipeline
from src.utils.logger import setup_logger
from src.volatility.garch_model import run_volatility_pipeline

logger = setup_logger("script_train")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train all quantitative, statistical, and ML models for BTC Sentinel."
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force redownloading latest market data prior to training.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print(" BTC SENTINEL: End-to-End Model Training Pipeline")
    print("=" * 65)

    try:
        # Step 1: Ensure processed data is present
        data_path = Path("data/processed/btc_daily.csv")
        if not data_path.exists() or args.force_download:
            print("\n>>> [1/7] Ingesting & Validating Market Data...")
            run_data_pipeline(force_download=args.force_download)
        else:
            print(f"\n>>> [1/7] Verified processed market data at {data_path}")

        # Step 2: Naive Persistence Baseline
        print("\n>>> [2/7] Evaluating Naive Persistence Benchmark...")
        run_baseline_evaluation()

        # Step 3: ARIMA Price Forecaster
        print("\n>>> [3/7] Estimating ARIMA Time-Series Model & Order Search...")
        run_arima_pipeline()

        # Step 4: XGBoost Price Forecaster
        print("\n>>> [4/7] Training XGBoost Return-Compounded Forecaster...")
        run_xgboost_price_pipeline()

        # Step 5: GARCH(1,1) Volatility Model
        print("\n>>> [5/7] Fitting GARCH(1,1) Maximum Likelihood Estimator...")
        run_volatility_pipeline()

        # Step 6: Crash Risk Classifier
        print("\n>>> [6/7] Training Calibrated Forward Drawdown Classifier...")
        run_crash_risk_pipeline()

        # Step 7: Regime Detection & Pattern Similarity
        print("\n>>> [7/7] Running Market Regime Profiling & Historical Similarity...")
        run_regime_detection_pipeline()
        run_pattern_similarity_pipeline()

        print("\n" + "=" * 65)
        print(" ALL MODELS TRAINED AND ARTIFACTS SERIALIZED SUCCESSFULLY!")
        print(" Models saved in:  models/")
        print(" Reports saved in: reports/results/")
        print(" Figures saved in: reports/figures/")
        print("=" * 65 + "\n")

    except Exception as e:
        logger.exception(f"Training pipeline encountered fatal error: {e}")
        print(f"\n[ERROR] Model training failed: {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
