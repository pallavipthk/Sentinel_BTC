"""End-to-End Data Pipeline for BTC Sentinel.
Coordinates data acquisition, raw validation, cleaning, processed validation,
and persistence of structured data and reports.
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.data.cleaner import clean_btc_ohlcv
from src.data.downloader import DEFAULT_RAW_DATA_PATH, download_btc_ohlcv
from src.data.validator import ValidationReport, validate_btc_ohlcv
from src.utils.logger import setup_logger

logger = setup_logger("data_pipeline")

DEFAULT_PROCESSED_DATA_PATH = Path("data/processed/btc_daily.csv")
DEFAULT_VALIDATION_REPORT_PATH = Path("reports/results/data_validation_report.json")


def run_data_pipeline(
    raw_path: Path = DEFAULT_RAW_DATA_PATH,
    processed_path: Path = DEFAULT_PROCESSED_DATA_PATH,
    report_path: Path = DEFAULT_VALIDATION_REPORT_PATH,
    force_download: bool = False,
    start_date: str = "2018-01-01",
) -> Tuple[pd.DataFrame, ValidationReport]:
    """Executes the full Stage 1 Data Pipeline:
    1. Downloads historical BTC OHLCV records.
    2. Validates raw dataset integrity.
    3. Cleans, sanitizes, deduplicates, and chronologically sorts records.
    4. Validates final processed dataset integrity.
    5. Saves processed dataset and generates validation report.

    Args:
        raw_path: Path to raw CSV file.
        processed_path: Path where cleaned processed dataset will be stored.
        report_path: Path where JSON validation report will be saved.
        force_download: If True, forces redownloading even if raw cache exists.
        start_date: Historical start date ('YYYY-MM-DD').

    Returns:
        Tuple of (processed_df, final_validation_report).
    """
    logger.info("=== STEP 1: Acquiring Historical BTC OHLCV Data ===")
    raw_df = download_btc_ohlcv(output_path=raw_path, start_date=start_date, force_download=force_download)

    logger.info(f"Loaded raw dataset with {len(raw_df)} rows. Performing pre-cleaning validation...")
    raw_report = validate_btc_ohlcv(raw_df)
    if not raw_report.is_valid:
        logger.warning(f"Raw data has integrity issues that will be addressed in cleaning: {raw_report.errors}")

    logger.info("=== STEP 2: Cleaning & Preprocessing Dataset ===")
    cleaned_df, cleaning_summary = clean_btc_ohlcv(raw_df)

    logger.info("=== STEP 3: Post-Cleaning Verification & Validation ===")
    processed_report = validate_btc_ohlcv(cleaned_df)
    processed_report.metrics["cleaning_summary"] = cleaning_summary

    if not processed_report.is_valid:
        error_msg = f"FATAL: Processed dataset failed final validation: {processed_report.errors}"
        logger.critical(error_msg)
        raise ValueError(error_msg)

    logger.info("=== STEP 4: Persisting Processed Artifacts ===")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(processed_path, index=False)
    logger.info(f"Cleaned dataset successfully saved to: {processed_path} ({len(cleaned_df)} rows)")

    processed_report.save_json(report_path)
    logger.info(f"Validation report saved to: {report_path}")

    logger.info("=== Stage 1 Data Pipeline Execution Completed Successfully ===")
    return cleaned_df, processed_report


if __name__ == "__main__":
    run_data_pipeline(force_download=True)
