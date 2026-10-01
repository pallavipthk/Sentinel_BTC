"""Data Cleaning and Preprocessing Module for BTC Sentinel.
Enforces strict chronological ordering, standard column schema, deduplication,
and calendar continuity for Bitcoin daily OHLCV series.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("cleaner")

COLUMN_MAPPING = {
    "date": "Date",
    "timestamp": "Date",
    "time": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "vol": "Volume",
}


def clean_btc_ohlcv(
    df: pd.DataFrame,
    fill_missing_calendar_days: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Cleans, deduplicates, chronologically sorts, and verifies continuity of BTC OHLCV data.

    Args:
        df: Raw DataFrame.
        fill_missing_calendar_days: If True, reindexes across full daily date range
                                    and safely forward-fills missing calendar days.

    Returns:
        Tuple of (cleaned_df, cleaning_summary_dict).
    """
    logger.info("Starting data cleaning pipeline...")
    cleaned = df.copy()

    # 1. Standardize column names
    col_rename = {}
    for col in cleaned.columns:
        clean_col = str(col).strip().lower()
        if clean_col in COLUMN_MAPPING:
            col_rename[col] = COLUMN_MAPPING[clean_col]
        else:
            col_rename[col] = str(col).strip()
    cleaned = cleaned.rename(columns=col_rename)

    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in cleaned.columns:
            raise ValueError(f"Clean pipeline requires column '{col}', but it was not found in: {list(cleaned.columns)}")

    cleaned = cleaned[required_cols]
    initial_rows = len(cleaned)

    # 2. Parse and format Date
    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="coerce")
    malformed_count = int(cleaned["Date"].isnull().sum())
    if malformed_count > 0:
        logger.warning(f"Dropping {malformed_count} rows with invalid/malformed dates.")
        cleaned = cleaned.dropna(subset=["Date"])

    # Normalize to YYYY-MM-DD (remove sub-day time if any)
    cleaned["Date"] = cleaned["Date"].dt.strftime("%Y-%m-%d")

    # 3. Drop exact duplicate rows
    exact_dups = int(cleaned.duplicated().sum())
    if exact_dups > 0:
        logger.info(f"Removing {exact_dups} exact duplicate rows.")
        cleaned = cleaned.drop_duplicates()

    # 4. Handle multiple records for the same date
    # In case of duplicates, keep the entry with highest volume or latest valid prices
    date_dups = int(cleaned["Date"].duplicated().sum())
    if date_dups > 0:
        logger.warning(f"Resolving {date_dups} duplicate date entries by retaining record with highest volume.")
        cleaned = cleaned.sort_values(by=["Date", "Volume"], ascending=[True, False])
        cleaned = cleaned.drop_duplicates(subset=["Date"], keep="first")

    # 5. Convert numeric columns to float64 and sanitize
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").astype(float)

    # Drop any remaining rows with NaN in price columns
    price_nan_count = int(cleaned[["Open", "High", "Low", "Close"]].isnull().any(axis=1).sum())
    if price_nan_count > 0:
        logger.warning(f"Dropping {price_nan_count} rows with NaN in price columns.")
        cleaned = cleaned.dropna(subset=["Open", "High", "Low", "Close"])

    # Ensure Volume is non-negative and fill NaN volume with 0.0
    cleaned["Volume"] = cleaned["Volume"].fillna(0.0).clip(lower=0.0)

    # 6. Sort chronologically
    cleaned["DatetimeIndex"] = pd.to_datetime(cleaned["Date"])
    cleaned = cleaned.sort_values(by="DatetimeIndex").reset_index(drop=True)

    # 7. Check and fill missing calendar days if requested (24/7/365 crypto market continuity)
    days_filled = 0
    if fill_missing_calendar_days and len(cleaned) > 1:
        min_date = cleaned["DatetimeIndex"].min()
        max_date = cleaned["DatetimeIndex"].max()
        full_date_range = pd.date_range(start=min_date, end=max_date, freq="D")

        if len(full_date_range) > len(cleaned):
            days_filled = len(full_date_range) - len(cleaned)
            logger.info(f"Detected {days_filled} missing calendar dates in 24/7 series. Reindexing and forward-filling...")

            cleaned = cleaned.set_index("DatetimeIndex").reindex(full_date_range)
            cleaned["Close"] = cleaned["Close"].ffill()
            cleaned["Open"] = cleaned["Open"].fillna(cleaned["Close"])
            cleaned["High"] = cleaned["High"].fillna(cleaned["Close"])
            cleaned["Low"] = cleaned["Low"].fillna(cleaned["Close"])
            cleaned["Volume"] = cleaned["Volume"].fillna(0.0)
            cleaned["Date"] = cleaned.index.strftime("%Y-%m-%d")
            cleaned = cleaned.reset_index(drop=True)
        else:
            cleaned = cleaned.drop(columns=["DatetimeIndex"])
    else:
        if "DatetimeIndex" in cleaned.columns:
            cleaned = cleaned.drop(columns=["DatetimeIndex"])

    # 8. Sanity check OHLC boundaries after cleaning:
    # High = max(High, Open, Close)
    # Low = min(Low, Open, Close)
    cleaned["High"] = cleaned[["High", "Open", "Close"]].max(axis=1)
    cleaned["Low"] = cleaned[["Low", "Open", "Close"]].min(axis=1)

    cleaned = cleaned.reset_index(drop=True)
    final_rows = len(cleaned)

    summary = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "dropped_malformed_dates": malformed_count,
        "dropped_exact_duplicates": exact_dups,
        "dropped_duplicate_dates": date_dups,
        "dropped_price_nans": price_nan_count,
        "filled_missing_calendar_days": days_filled,
        "start_date": cleaned["Date"].iloc[0] if not cleaned.empty else None,
        "end_date": cleaned["Date"].iloc[-1] if not cleaned.empty else None,
    }

    logger.info(f"Data cleaning completed: {initial_rows} -> {final_rows} rows ({summary['start_date']} to {summary['end_date']}).")
    return cleaned, summary
