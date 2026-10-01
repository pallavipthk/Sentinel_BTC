"""CLI Entry point to download, validate, and clean BTC historical data.
Runnable from the project root:
    python scripts/download_data.py [--force] [--start-date YYYY-MM-DD]
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so scripts can run smoothly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.pipeline import run_data_pipeline
from src.utils.logger import setup_logger

logger = setup_logger("script_download_data")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and preprocess historical Bitcoin daily OHLCV data for BTC Sentinel."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-downloading even if local raw data file exists.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2018-01-01",
        help="Start date for historical data in YYYY-MM-DD format (default: 2018-01-01).",
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default="data/raw/btc_daily_raw.csv",
        help="Destination path for raw CSV data.",
    )
    parser.add_argument(
        "--processed-path",
        type=str,
        default="data/processed/btc_daily.csv",
        help="Destination path for cleaned processed CSV data.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="reports/results/data_validation_report.json",
        help="Destination path for JSON data validation report.",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" BTC SENTINEL: Stage 1 Data Acquisition & Preprocessing")
    print("=" * 60)

    try:
        cleaned_df, report = run_data_pipeline(
            raw_path=Path(args.raw_path),
            processed_path=Path(args.processed_path),
            report_path=Path(args.report_path),
            force_download=args.force,
            start_date=args.start_date,
        )

        print("\n" + "-" * 60)
        print(" DATA PIPELINE EXECUTION SUMMARY")
        print("-" * 60)
        print(f"Validation Status:     {'PASSED (VALID)' if report.is_valid else 'FAILED (INVALID)'}")
        print(f"Total Processed Days:  {report.total_rows:,}")
        print(f"Date Range:            {report.date_range[0]} to {report.date_range[1]}")
        print(f"Missing Values:        {sum(report.missing_values.values())}")
        print(f"Duplicate Timestamps:  {report.duplicate_timestamps}")
        print(f"Unsorted Timestamps:   {report.unsorted_timestamps}")
        print(f"Non-positive Prices:   {report.non_positive_prices}")
        print(f"OHLC Violations:       {report.invalid_ohlc_relationships}")
        print(f"Earliest Close Price:  ${cleaned_df['Close'].iloc[0]:,.2f}")
        print(f"Latest Close Price:    ${cleaned_df['Close'].iloc[-1]:,.2f}")
        print(f"Min / Max Close:       ${report.metrics['min_close']:,.2f} / ${report.metrics['max_close']:,.2f}")
        print(f"Processed File Saved:  {args.processed_path}")
        print(f"Validation Report:     {args.report_path}")
        print("-" * 60)
        print("Stage 1 verification completed successfully!\n")

    except Exception as e:
        logger.exception(f"Stage 1 Pipeline failed: {e}")
        print(f"\n[ERROR] Pipeline execution terminated with failure: {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
