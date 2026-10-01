"""CLI entry point for running chronological walk-forward backtesting.
Usage:
    python scripts/backtest.py [--n-splits 5] [--window-size 90]
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.backtester import run_walk_forward_pipeline
from src.utils.logger import setup_logger

logger = setup_logger("script_backtest")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run chronological expanding-window walk-forward backtest for BTC Sentinel."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/btc_daily.csv",
        help="Path to processed daily BTC CSV.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of chronological walk-forward folds (default: 5).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=90,
        help="Days per evaluation fold (default: 90).",
    )

    args = parser.parse_args()

    print("\n" + "=" * 65)
    print(" BTC SENTINEL: Chronological Walk-Forward Backtesting")
    print("=" * 65)

    try:
        summary, pred_df = run_walk_forward_pipeline(
            processed_data_path=Path(args.data_path),
            n_splits=args.n_splits,
            test_window_size=args.window_size,
        )

        print("\n" + "-" * 65)
        print(" WALK-FORWARD BACKTEST RESULTS SUMMARY")
        print("-" * 65)
        print(f"Total Out-Of-Sample Days: {summary['total_backtest_days']} days ({summary['backtest_period'][0]} to {summary['backtest_period'][1]})")
        print(f"Evaluation Folds:         {summary['n_splits']} sequential blocks of {summary['test_window_size']} days\n")

        print(f"{'Model':<18} {'MAE ($)':<12} {'RMSE ($)':<12} {'MAPE (%)':<10} {'Dir. Acc (%)':<12}")
        print("-" * 65)
        models = summary["models"]
        naive = models["Naive_Persistence"]
        arima = models["ARIMA"]
        xgb = models["XGBoost"]

        print(f"{'Naive Baseline':<18} {naive['mae']:<12.2f} {naive['rmse']:<12.2f} {naive['mape']:<10.2f} {naive.get('directional_accuracy', 0.0):<12.1f}")
        print(f"{'ARIMA(1,1,0)':<18} {arima['mae']:<12.2f} {arima['rmse']:<12.2f} {arima['mape']:<10.2f} {arima.get('directional_accuracy', 0.0):<12.1f}")
        print(f"{'XGBoost':<18} {xgb['price_mae']:<12.2f} {xgb['price_rmse']:<12.2f} {xgb['price_mape']:<10.2f} {xgb.get('directional_accuracy', 0.0):<12.1f}")

        print("-" * 65)
        print("Results persisted to reports/results/walk_forward_backtest_results.json")
        print("Predictions persisted to reports/results/walk_forward_backtest_predictions.csv")
        print("Charts saved to reports/figures/\n")

    except Exception as e:
        logger.exception(f"Backtesting execution failed: {e}")
        print(f"\n[ERROR] Backtesting failed: {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
