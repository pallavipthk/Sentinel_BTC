"""Unit and integration tests for BTC Sentinel Stage 1 Data Pipeline.
Validates detection of data anomalies, cleaning integrity, and chronological ordering.
"""

from pathlib import Path
import tempfile
import pandas as pd
import pytest

from src.data.cleaner import clean_btc_ohlcv
from src.data.pipeline import run_data_pipeline
from src.data.validator import validate_btc_ohlcv


@pytest.fixture
def valid_ohlcv_df() -> pd.DataFrame:
    """Creates a small, valid daily OHLCV DataFrame."""
    return pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
        "Open": [16500.0, 16600.0, 16700.0, 16650.0, 16800.0],
        "High": [16700.0, 16800.0, 16900.0, 16850.0, 17000.0],
        "Low": [16400.0, 16500.0, 16600.0, 16550.0, 16750.0],
        "Close": [16600.0, 16700.0, 16650.0, 16800.0, 16950.0],
        "Volume": [1000.5, 1200.0, 1150.3, 1300.2, 1400.0],
    })


def test_validator_accepts_valid_data(valid_ohlcv_df):
    """Ensures validator passes on clean, valid OHLCV series."""
    report = validate_btc_ohlcv(valid_ohlcv_df)
    assert report.is_valid is True
    assert len(report.errors) == 0
    assert report.total_rows == 5
    assert report.non_positive_prices == 0
    assert report.duplicate_timestamps == 0
    assert report.unsorted_timestamps == 0


def test_validator_detects_empty_df():
    """Ensures validator fails on empty DataFrame."""
    empty_df = pd.DataFrame()
    report = validate_btc_ohlcv(empty_df)
    assert report.is_valid is False
    assert any("empty" in err.lower() for err in report.errors)


def test_validator_detects_missing_columns():
    """Ensures validator fails if any required column is missing."""
    bad_df = pd.DataFrame({"Date": ["2023-01-01"], "Close": [16000.0]})
    report = validate_btc_ohlcv(bad_df)
    assert report.is_valid is False
    assert any("missing required columns" in err.lower() for err in report.errors)


def test_validator_detects_non_positive_prices(valid_ohlcv_df):
    """Ensures validator flags non-positive or zero prices."""
    bad_df = valid_ohlcv_df.copy()
    bad_df.loc[2, "Close"] = -50.0
    report = validate_btc_ohlcv(bad_df)
    assert report.is_valid is False
    assert report.non_positive_prices > 0
    assert any("non-positive" in err.lower() for err in report.errors)


def test_validator_detects_invalid_ohlc_relationships(valid_ohlcv_df):
    """Ensures validator flags when High < Low or High < Close."""
    bad_df = valid_ohlcv_df.copy()
    bad_df.loc[1, "High"] = 16000.0  # High is less than Low (16500) and Close (16700)
    report = validate_btc_ohlcv(bad_df)
    assert report.is_valid is False
    assert report.invalid_ohlc_relationships > 0
    assert any("ohlc relationship" in err.lower() for err in report.errors)


def test_validator_detects_duplicate_timestamps(valid_ohlcv_df):
    """Ensures validator flags duplicate timestamps."""
    bad_df = valid_ohlcv_df.copy()
    bad_df.loc[2, "Date"] = "2023-01-02"  # Duplicate of row 1
    report = validate_btc_ohlcv(bad_df)
    assert report.is_valid is False
    assert report.duplicate_timestamps > 0
    assert any("duplicate timestamps" in err.lower() for err in report.errors)


def test_validator_detects_unsorted_timestamps(valid_ohlcv_df):
    """Ensures validator flags out-of-order timestamps."""
    bad_df = valid_ohlcv_df.copy()
    bad_df.loc[3, "Date"] = "2022-12-31"  # Out of order
    report = validate_btc_ohlcv(bad_df)
    assert report.is_valid is False
    assert report.unsorted_timestamps > 0
    assert any("chronological" in err.lower() for err in report.errors)


def test_cleaner_standardizes_columns_and_deduplicates():
    """Tests cleaner column normalization, duplicate removal, and chronological sorting."""
    dirty_df = pd.DataFrame({
        "date": ["2023-01-03", "2023-01-01", "2023-01-01", "2023-01-02"],
        "open": ["16700", "16500", "16500", "16600"],
        "high": ["16900", "16700", "16700", "16800"],
        "low": ["16600", "16400", "16400", "16500"],
        "close": ["16800", "16600", "16600", "16700"],
        "volume": ["1100", "1000", "1050", "1200"],
    })

    clean_df, summary = clean_btc_ohlcv(dirty_df, fill_missing_calendar_days=False)

    # Columns standardized
    assert list(clean_df.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    # Duplicates removed (was 4 rows, row 1 & 2 share same date '2023-01-01', so now 3 unique dates)
    assert len(clean_df) == 3
    # Sorted chronologically
    assert list(clean_df["Date"]) == ["2023-01-01", "2023-01-02", "2023-01-03"]
    # Proper float dtypes
    assert clean_df["Close"].dtype == float


def test_cleaner_fills_missing_calendar_days():
    """Tests that missing days in 24/7 market are continuously forward-filled."""
    gap_df = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-04"],  # missing Jan 2 and Jan 3
        "Open": [16000.0, 17000.0],
        "High": [16500.0, 17500.0],
        "Low": [15900.0, 16900.0],
        "Close": [16200.0, 17200.0],
        "Volume": [100.0, 200.0],
    })

    clean_df, summary = clean_btc_ohlcv(gap_df, fill_missing_calendar_days=True)
    assert len(clean_df) == 4
    assert list(clean_df["Date"]) == ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
    # The filled days (Jan 2 and Jan 3) should have Close equal to Jan 1 Close (16200.0)
    assert clean_df.loc[1, "Close"] == 16200.0
    assert clean_df.loc[2, "Close"] == 16200.0
    assert summary["filled_missing_calendar_days"] == 2


def test_cleaner_enforces_ohlc_boundary_constraints():
    """Tests that cleaner corrects slight boundary inversions."""
    boundary_df = pd.DataFrame({
        "Date": ["2023-01-01"],
        "Open": [16500.0],
        "High": [16200.0],  # Faulty: High < Open
        "Low": [16800.0],   # Faulty: Low > Open
        "Close": [16600.0],
        "Volume": [50.0],
    })

    clean_df, _ = clean_btc_ohlcv(boundary_df, fill_missing_calendar_days=False)
    # High must be >= max(High, Open, Close) = 16600
    assert clean_df["High"].iloc[0] >= clean_df["Close"].iloc[0]
    assert clean_df["High"].iloc[0] >= clean_df["Open"].iloc[0]
    # Low must be <= min(Low, Open, Close) = 16500
    assert clean_df["Low"].iloc[0] <= clean_df["Open"].iloc[0]
    assert clean_df["Low"].iloc[0] <= clean_df["Close"].iloc[0]
