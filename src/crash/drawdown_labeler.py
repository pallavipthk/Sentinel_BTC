"""Forward Drawdown Labeling Engine.
Implements forward-looking epsilon-drawdown labeling over a fixed prediction horizon H.
Strictly separates training targets from feature calculations to avoid data leakage.
"""

from typing import Tuple
import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("drawdown_labeler")


def label_forward_drawdowns(
    df: pd.DataFrame,
    horizon_days: int = 14,
    drawdown_threshold: float = 0.10,
    price_col: str = "Close",
) -> Tuple[pd.DataFrame, dict]:
    """Labels crash/drawdown events over an out-of-sample forward window [t+1, t+H].

    Exact Definition:
    At day t, we observe current close price P_t.
    Over the next H days {t+1, ..., t+H}, the lowest observed price is P_min.
    Future Maximum Drawdown = (P_min - P_t) / P_t.
    Crash Label y_t = 1 if Future Maximum Drawdown <= -drawdown_threshold, else 0.
    For the final H rows of the dataset, future observations are unobserved, so y_t = NaN.

    Args:
        df: Input DataFrame containing chronological records.
        horizon_days: Forward lookahead horizon (default: 14 days).
        drawdown_threshold: Minimum drop from current price to trigger crash label (default: 0.10 = -10%).
        price_col: Price column name.

    Returns:
        Tuple of (enriched_dataframe, metadata_summary_dict).
    """
    logger.info(
        f"Computing forward drawdown labels: Horizon={horizon_days} days, "
        f"Threshold={drawdown_threshold * 100:.1f}% drop..."
    )
    result = df.copy()
    prices = result[price_col].values
    n = len(prices)

    future_min_price = np.full(n, np.nan)
    future_max_drawdown = np.full(n, np.nan)
    crash_label = np.full(n, np.nan)

    # For each index t, examine window t+1 to t+horizon_days
    for t in range(n - horizon_days):
        forward_slice = prices[t + 1 : t + horizon_days + 1]
        min_p = np.min(forward_slice)
        dd = (min_p - prices[t]) / prices[t]

        future_min_price[t] = min_p
        future_max_drawdown[t] = dd
        crash_label[t] = 1.0 if dd <= -drawdown_threshold else 0.0

    result["future_min_price"] = future_min_price
    result["future_max_drawdown"] = future_max_drawdown
    result["target_crash"] = crash_label

    valid_labels = crash_label[~np.isnan(crash_label)]
    n_crashes = int(np.sum(valid_labels == 1.0))
    n_non_crashes = int(np.sum(valid_labels == 0.0))
    crash_rate = (n_crashes / len(valid_labels)) * 100.0 if len(valid_labels) > 0 else 0.0

    summary = {
        "horizon_days": horizon_days,
        "drawdown_threshold": drawdown_threshold,
        "total_labeled_days": len(valid_labels),
        "unlabeled_lead_days": horizon_days,
        "crash_events_count": n_crashes,
        "non_crash_count": n_non_crashes,
        "crash_prevalence_pct": crash_rate,
    }

    logger.info(
        f"Forward drawdown labeling completed: {n_crashes} crash periods ({crash_rate:.2f}% prevalence) "
        f"out of {len(valid_labels)} labeled days."
    )
    return result, summary
