"""BTC Historical Data Downloader.
Fetches daily Bitcoin OHLCV data from free, reliable public market endpoints
without requiring paid API keys, with automatic fallback support.
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.utils.logger import setup_logger

logger = setup_logger("downloader")

DEFAULT_RAW_DATA_PATH = Path("data/raw/btc_daily_raw.csv")
REQUEST_TIMEOUT = 15
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def fetch_from_binance(
    symbol: str = "BTCUSDT",
    start_date: str = "2018-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetches historical daily klines from Binance public REST API with pagination.

    Args:
        symbol: Trading pair symbol (default: 'BTCUSDT').
        start_date: Start date string 'YYYY-MM-DD'.
        end_date: Optional end date string 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame with columns: Date, Open, High, Low, Close, Volume.
    """
    logger.info(f"Initiating fetch from Binance API for {symbol} starting {start_date}...")
    base_urls = [
        "https://api.binance.com/api/v3/klines",
        "https://data-api.binance.vision/api/v3/klines",
    ]

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)

    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_ms = int(end_dt.timestamp() * 1000)
    else:
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    all_klines = []
    current_start = start_ms
    limit = 1000
    headers = {"User-Agent": USER_AGENT}

    active_url = base_urls[0]

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": current_start,
            "endTime": end_ms,
            "limit": limit,
        }

        success = False
        for url in base_urls:
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list):
                        active_url = url
                        success = True
                        break
            except Exception as e:
                logger.warning(f"Connection error to {url}: {e}")

        if not success or not data:
            logger.warning(f"No further records returned by Binance or request failed at timestamp {current_start}.")
            break

        all_klines.extend(data)
        logger.debug(f"Fetched {len(data)} klines, total so far: {len(all_klines)}")

        # Next start time is last candle's close time + 1ms
        last_close_time = data[-1][6]
        if last_close_time <= current_start:
            break
        current_start = last_close_time + 1

        # Small delay to respect rate limits
        time.sleep(0.1)

        if len(data) < limit:
            break

    if not all_klines:
        raise RuntimeError("No records received from Binance API.")

    records = []
    for k in all_klines:
        # kline format: [open_time, open, high, low, close, volume, close_time, ...]
        dt = datetime.fromtimestamp(k[0] / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
        records.append({
            "Date": dt,
            "Open": float(k[1]),
            "High": float(k[2]),
            "Low": float(k[3]),
            "Close": float(k[4]),
            "Volume": float(k[5]),
        })

    df = pd.DataFrame(records)
    logger.info(f"Successfully downloaded {len(df)} daily records from Binance.")
    return df


def fetch_from_yahoo(symbol: str = "BTC-USD", range_str: str = "10y") -> pd.DataFrame:
    """Fallback fetcher querying Yahoo Finance public chart API directly.

    Args:
        symbol: Asset ticker (default: 'BTC-USD').
        range_str: Historical span (default: '10y').

    Returns:
        pd.DataFrame with columns: Date, Open, High, Low, Close, Volume.
    """
    logger.info(f"Attempting fallback fetch from Yahoo Finance Chart API for {symbol} ({range_str})...")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range={range_str}"
    headers = {"User-Agent": USER_AGENT}

    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"Yahoo Finance returned HTTP {resp.status_code}: {resp.text[:200]}")

    payload = resp.json()
    result = payload.get("chart", {}).get("result", [])
    if not result:
        raise RuntimeError("Empty response payload from Yahoo Finance.")

    data = result[0]
    timestamps = data.get("timestamp", [])
    quotes = data.get("indicators", {}).get("quote", [{}])[0]

    opens = quotes.get("open", [])
    highs = quotes.get("high", [])
    lows = quotes.get("low", [])
    closes = quotes.get("close", [])
    volumes = quotes.get("volume", [])

    records = []
    for t, o, h, l, c, v in zip(timestamps, opens, highs, lows, closes, volumes):
        if any(x is None for x in (o, h, l, c)):
            continue
        dt_str = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")
        records.append({
            "Date": dt_str,
            "Open": float(o),
            "High": float(h),
            "Low": float(l),
            "Close": float(c),
            "Volume": float(v) if v is not None else 0.0,
        })

    df = pd.DataFrame(records)
    logger.info(f"Successfully retrieved {len(df)} daily records from Yahoo Finance.")
    return df


def download_btc_ohlcv(
    output_path: Optional[Path] = None,
    start_date: str = "2018-01-01",
    force_download: bool = False,
) -> pd.DataFrame:
    """Downloads BTC daily OHLCV dataset from primary source (Binance) with fallback (Yahoo).

    Args:
        output_path: Destination path for saving raw CSV (default: data/raw/btc_daily_raw.csv).
        start_date: Historical start date ('YYYY-MM-DD').
        force_download: If True, redownload even if raw file exists.

    Returns:
        pd.DataFrame containing downloaded raw records.
    """
    save_path = output_path or DEFAULT_RAW_DATA_PATH
    save_path = Path(save_path)

    if save_path.exists() and not force_download:
        logger.info(f"Existing raw data found at {save_path}. Loading cached file.")
        try:
            df = pd.read_csv(save_path)
            if not df.empty and "Close" in df.columns:
                return df
        except Exception as e:
            logger.warning(f"Could not load existing file {save_path}: {e}. Proceeding to redownload.")

    df = None
    errors = []

    # Attempt 1: Binance
    try:
        df = fetch_from_binance(symbol="BTCUSDT", start_date=start_date)
    except Exception as e:
        logger.warning(f"Binance fetch failed: {e}. Switching to fallback provider...")
        errors.append(f"Binance error: {e}")

    # Attempt 2: Yahoo Finance Fallback
    if df is None or df.empty:
        try:
            df = fetch_from_yahoo(symbol="BTC-USD", range_str="10y")
        except Exception as e:
            logger.error(f"Yahoo Finance fetch also failed: {e}")
            errors.append(f"Yahoo error: {e}")

    if df is None or df.empty:
        error_msg = (
            "FATAL: All data acquisition sources failed.\n"
            f"Details: {'; '.join(errors)}\n"
            "Troubleshooting: Please check your internet connection, or place a manual "
            f"CSV file with columns [Date, Open, High, Low, Close, Volume] at {save_path.resolve()}."
        )
        logger.critical(error_msg)
        raise ConnectionError(error_msg)

    # Save raw data
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Raw BTC data successfully persisted to {save_path} ({len(df)} rows).")

    return df


if __name__ == "__main__":
    download_btc_ohlcv(force_download=True)
