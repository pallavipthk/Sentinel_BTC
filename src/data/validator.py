"""Data Validation Module for BTC Sentinel.
Strictly checks for missing values, duplicates, out-of-order timestamps,
invalid/non-positive prices, violated OHLC relationships, and missing periods.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("validator")

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


@dataclass
class ValidationReport:
    """Structured report detailing validation findings."""
    is_valid: bool = True
    total_rows: int = 0
    date_range: Optional[Tuple[str, str]] = None
    missing_values: Dict[str, int] = field(default_factory=dict)
    duplicate_timestamps: int = 0
    duplicate_rows: int = 0
    unsorted_timestamps: int = 0
    non_positive_prices: int = 0
    invalid_ohlc_relationships: int = 0
    missing_calendar_days: int = 0
    malformed_dates: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts report to dictionary representation."""
        return asdict(self)

    def save_json(self, file_path: Path) -> None:
        """Saves validation report as a formatted JSON document."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Validation report saved to {file_path}")


def validate_btc_ohlcv(df: pd.DataFrame) -> ValidationReport:
    """Performs rigorous statistical and relational integrity validation on BTC OHLCV data.

    Args:
        df: Input DataFrame containing daily OHLCV records.

    Returns:
        ValidationReport instance containing detailed validation statistics and pass/fail state.
    """
    report = ValidationReport()
    report.total_rows = len(df)

    if df.empty:
        report.is_valid = False
        report.errors.append("Dataset is completely empty.")
        logger.error("Validation failed: DataFrame has 0 rows.")
        return report

    # 1. Required column presence
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        report.is_valid = False
        report.errors.append(f"Missing required columns: {missing_cols}")
        logger.error(f"Validation failed: Missing columns {missing_cols}")
        return report

    # 2. Missing values check
    missing_counts = df[REQUIRED_COLUMNS].isnull().sum().to_dict()
    report.missing_values = {col: int(cnt) for col, cnt in missing_counts.items() if cnt > 0}
    if report.missing_values:
        total_missing = sum(report.missing_values.values())
        report.is_valid = False
        report.errors.append(f"Missing values found across required columns: {report.missing_values}")
        logger.error(f"Found {total_missing} missing values: {report.missing_values}")

    # 3. Malformed dates check & conversion
    try:
        parsed_dates = pd.to_datetime(df["Date"], errors="coerce")
        malformed = int(parsed_dates.isnull().sum())
        report.malformed_dates = malformed
        if malformed > 0:
            report.is_valid = False
            report.errors.append(f"Found {malformed} malformed / unparseable dates in 'Date' column.")
            logger.error(f"Malformed dates count: {malformed}")
    except Exception as e:
        report.is_valid = False
        report.errors.append(f"Error parsing date column: {e}")
        return report

    # 4. Duplicate rows & duplicate timestamps
    dup_rows = int(df.duplicated().sum())
    report.duplicate_rows = dup_rows
    if dup_rows > 0:
        report.warnings.append(f"Found {dup_rows} exact duplicate rows.")
        logger.warning(f"Detected {dup_rows} exact duplicate rows.")

    dup_dates = int(df["Date"].duplicated().sum())
    report.duplicate_timestamps = dup_dates
    if dup_dates > 0:
        report.is_valid = False
        report.errors.append(f"Found {dup_dates} duplicate timestamps in 'Date' column.")
        logger.error(f"Detected {dup_dates} duplicate timestamps.")

    # 5. Chronological sorting check
    if not parsed_dates.is_monotonic_increasing:
        report.is_valid = False
        # Calculate how many dates are out of chronological order
        diffs = parsed_dates.diff()
        unsorted_count = int((diffs.dt.total_seconds() < 0).sum())
        report.unsorted_timestamps = max(unsorted_count, 1)
        report.errors.append(f"Timestamps are NOT strictly chronological ({report.unsorted_timestamps} reversals).")
        logger.error("Validation failed: Timestamps are not in ascending chronological order.")

    # 6. Invalid / Non-positive prices & negative volume
    price_cols = ["Open", "High", "Low", "Close"]
    non_pos_mask = (df[price_cols] <= 0).any(axis=1) | (df["Volume"] < 0)
    non_pos_count = int(non_pos_mask.sum())
    report.non_positive_prices = non_pos_count
    if non_pos_count > 0:
        report.is_valid = False
        report.errors.append(f"Found {non_pos_count} records with non-positive prices or negative volume.")
        logger.error(f"Detected {non_pos_count} non-positive prices.")

    # 7. Invalid OHLC Relationships
    # Rule 1: High must be >= Low
    # Rule 2: High must be >= Open and High >= Close
    # Rule 3: Low must be <= Open and Low <= Close
    invalid_high_low = (df["High"] < df["Low"])
    invalid_high_open_close = (df["High"] < df["Open"]) | (df["High"] < df["Close"])
    invalid_low_open_close = (df["Low"] > df["Open"]) | (df["Low"] > df["Close"])

    ohlc_violations = invalid_high_low | invalid_high_open_close | invalid_low_open_close
    ohlc_violation_count = int(ohlc_violations.sum())
    report.invalid_ohlc_relationships = ohlc_violation_count
    if ohlc_violation_count > 0:
        report.is_valid = False
        report.errors.append(f"Found {ohlc_violation_count} records violating OHLC relationship constraints (High >= Low, High >= Open/Close, Low <= Open/Close).")
        logger.error(f"Detected {ohlc_violation_count} OHLC relationship violations.")

    # 8. Missing periods / calendar gaps
    valid_dates = parsed_dates.dropna().sort_values()
    if len(valid_dates) > 1:
        date_deltas = valid_dates.diff().iloc[1:]
        gap_mask = date_deltas > pd.Timedelta(days=1)
        missing_days_count = int(gap_mask.sum())
        report.missing_calendar_days = missing_days_count
        if missing_days_count > 0:
            report.warnings.append(f"Detected {missing_days_count} calendar day gaps (>1 day) in daily series.")
            logger.warning(f"Detected {missing_days_count} calendar gaps in daily frequency.")

        report.date_range = (
            valid_dates.iloc[0].strftime("%Y-%m-%d"),
            valid_dates.iloc[-1].strftime("%Y-%m-%d"),
        )

    # 9. Summary metrics
    report.metrics = {
        "start_date": report.date_range[0] if report.date_range else None,
        "end_date": report.date_range[1] if report.date_range else None,
        "total_days": len(df),
        "mean_close": float(df["Close"].mean()) if "Close" in df.columns and not df.empty else None,
        "min_close": float(df["Close"].min()) if "Close" in df.columns and not df.empty else None,
        "max_close": float(df["Close"].max()) if "Close" in df.columns and not df.empty else None,
        "mean_volume": float(df["Volume"].mean()) if "Volume" in df.columns and not df.empty else None,
    }

    if report.is_valid:
        logger.info(f"Data validation PASSED cleanly for {report.total_rows} records ({report.date_range[0]} to {report.date_range[1]}).")
    else:
        logger.warning(f"Data validation completed with {len(report.errors)} errors and {len(report.warnings)} warnings.")

    return report
