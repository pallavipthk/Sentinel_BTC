"""Transparent, Explainable Market Regime Detector.
Classifies Bitcoin market conditions at time t into discrete macro regimes:
- BULL_MOMENTUM (sustained uptrend with positive momentum)
- BEAR_DOWNTREND (sustained downtrend with negative momentum and severe drawdown)
- HIGH_VOLATILITY_EXPANSION (turbulent regimes with elevated annualized volatility)
- LOW_VOLATILITY_CONSOLIDATION (quiet compression / rangebound trading)
- NEUTRAL_TRANSITION (intermediate / rotational phases)

Calculated strictly causally using historical trend, momentum, volatility, and drawdown indicators.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.feature_engineer import calculate_features
from src.utils.logger import setup_logger

logger = setup_logger("regime_detector")

DEFAULT_RESULTS_DIR = Path("reports/results")
DEFAULT_FIGURES_DIR = Path("reports/figures")


@dataclass
class RegimeState:
    """Detailed summary of the detected market regime and supporting quantitative factors."""
    regime: str
    date: str
    price: float
    trend_state: str
    momentum_state: str
    volatility_state: str
    drawdown_state: str
    metrics: Dict[str, float]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MarketRegimeDetector:
    """Rule-based, explainable market regime classifier."""

    def __init__(
        self,
        vol_high_threshold: float = 0.65,
        vol_low_threshold: float = 0.50,
        drawdown_bear_threshold: float = -0.20,
    ):
        self.vol_high = vol_high_threshold
        self.vol_low = vol_low_threshold
        self.dd_bear = drawdown_bear_threshold

    def evaluate_single_observation(self, row: pd.Series) -> RegimeState:
        """Classifies a single point in time and builds explainable supporting factors."""
        date_str = str(row.get("Date", "Unknown"))
        price = float(row.get("Close", 0.0))

        # 1. Trend Factor (Close vs SMA-50 and SMA-200)
        sma_50 = float(row.get("sma_50", price))
        sma_200 = float(row.get("sma_200", price))
        pct_above_50 = (price - sma_50) / (sma_50 + 1e-10) * 100.0
        pct_above_200 = (price - sma_200) / (sma_200 + 1e-10) * 100.0

        is_uptrend = (price > sma_50) and (sma_50 >= sma_200)
        is_downtrend = (price < sma_50) and (sma_50 <= sma_200)

        trend_state = "Bullish Uptrend" if is_uptrend else ("Bearish Downtrend" if is_downtrend else "Mixed / Neutral Trend")

        # 2. Momentum Factor (30d momentum and RSI-14)
        mom_30d = float(row.get("momentum_30d", 0.0)) * 100.0
        rsi_14 = float(row.get("rsi_14", 50.0))

        if mom_30d > 5.0 and rsi_14 > 50.0:
            momentum_state = f"Positive Momentum (+{mom_30d:.1f}%, RSI {rsi_14:.1f})"
        elif mom_30d < -5.0 and rsi_14 < 50.0:
            momentum_state = f"Negative Momentum ({mom_30d:.1f}%, RSI {rsi_14:.1f})"
        else:
            momentum_state = f"Neutral Momentum ({mom_30d:+.1f}%, RSI {rsi_14:.1f})"

        # 3. Volatility Factor (Annualized 30d rolling volatility)
        vol_30d = float(row.get("rolling_volatility_30d", 0.55))
        if vol_30d >= self.vol_high:
            vol_state = f"High Volatility ({vol_30d * 100:.1f}% annualized)"
        elif vol_30d <= self.vol_low:
            vol_state = f"Low Volatility ({vol_30d * 100:.1f}% annualized)"
        else:
            vol_state = f"Medium Volatility ({vol_30d * 100:.1f}% annualized)"

        # 4. Drawdown Factor (Rolling 90d drawdown & ATH drawdown)
        dd_90d = float(row.get("rolling_drawdown_90d", 0.0)) * 100.0
        dd_ath = float(row.get("drawdown_from_ath", 0.0)) * 100.0
        drawdown_state = f"{dd_90d:.1f}% from 90d peak ({dd_ath:.1f}% from ATH)"

        # Transparent Hierarchical Decision Tree
        if vol_30d >= self.vol_high and abs(mom_30d) > 15.0:
            regime = "HIGH_VOLATILITY_EXPANSION"
            explanation = (
                f"Market is undergoing high volatility expansion ({vol_30d*100:.1f}% annualized) "
                f"with rapid price velocity ({mom_30d:+.1f}% 30-day shift)."
            )
        elif is_downtrend and (dd_90d <= self.dd_bear * 100.0 or mom_30d < -10.0):
            regime = "BEAR_DOWNTREND"
            explanation = (
                f"Price is in a confirmed bear downtrend (below 50 & 200 SMAs), "
                f"with {momentum_state} and severe drawdown ({drawdown_state})."
            )
        elif is_uptrend and mom_30d > 0.0 and dd_90d > -15.0:
            regime = "BULL_MOMENTUM"
            explanation = (
                f"Price displays strong bullish momentum structure (trading +{pct_above_50:.1f}% above 50 SMA "
                f"and +{pct_above_200:.1f}% above 200 SMA) with constructive momentum ({momentum_state})."
            )
        elif vol_30d <= self.vol_low and abs(mom_30d) <= 10.0:
            regime = "LOW_VOLATILITY_CONSOLIDATION"
            explanation = (
                f"Market is consolidating in a low-volatility compression band ({vol_30d*100:.1f}% annualized) "
                f"with tight price range and flat 30d momentum ({mom_30d:+.1f}%)."
            )
        else:
            regime = "NEUTRAL_TRANSITION"
            explanation = (
                f"Indicators reflect a transitional or rotational market state: "
                f"{trend_state}, {momentum_state}, and {vol_state}."
            )

        return RegimeState(
            regime=regime,
            date=date_str,
            price=price,
            trend_state=trend_state,
            momentum_state=momentum_state,
            volatility_state=vol_state,
            drawdown_state=drawdown_state,
            metrics={
                "close": price,
                "sma_50": sma_50,
                "sma_200": sma_200,
                "momentum_30d_pct": mom_30d,
                "rsi_14": rsi_14,
                "volatility_30d_annualized": vol_30d,
                "drawdown_90d_pct": dd_90d,
                "drawdown_ath_pct": dd_ath,
            },
            explanation=explanation,
        )


def detect_market_regimes(df_features: pd.DataFrame) -> Tuple[pd.Series, List[RegimeState]]:
    """Calculates market regimes across all historical observations strictly causally."""
    detector = MarketRegimeDetector()
    regime_labels = []
    regime_states = []

    for _, row in df_features.iterrows():
        state = detector.evaluate_single_observation(row)
        regime_labels.append(state.regime)
        regime_states.append(state)

    return pd.Series(regime_labels, index=df_features.index), regime_states


def plot_market_regimes(
    dates: pd.Series,
    prices: np.ndarray,
    regimes: pd.Series,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> None:
    """Plots historical Bitcoin price chart color-coded by detected market regimes."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7))

    regime_colors = {
        "BULL_MOMENTUM": "#10b981",              # Green
        "BEAR_DOWNTREND": "#ef4444",             # Red
        "HIGH_VOLATILITY_EXPANSION": "#f59e0b",   # Amber
        "LOW_VOLATILITY_CONSOLIDATION": "#3b82f6",# Blue
        "NEUTRAL_TRANSITION": "#9ca3af",          # Gray
    }

    ax.plot(dates, prices, color="#111827", linewidth=1.5, zorder=2, label="BTC Price (USD)")

    # Group consecutive regime stretches to draw vertical colored spans
    current_regime = regimes.iloc[0]
    start_idx = 0

    for i in range(1, len(regimes)):
        if regimes.iloc[i] != current_regime or i == len(regimes) - 1:
            color = regime_colors.get(current_regime, "#d1d5db")
            ax.axvspan(dates.iloc[start_idx], dates.iloc[i], color=color, alpha=0.22, zorder=1)
            current_regime = regimes.iloc[i]
            start_idx = i

    # Custom legend for regimes
    from matplotlib.patches import Patch
    legend_elements = [plt.Line2D([0], [0], color="#111827", linewidth=1.5, label="BTC Price")]
    for name, col in regime_colors.items():
        clean_name = name.replace("_", " ").title()
        legend_elements.append(Patch(facecolor=col, alpha=0.35, label=clean_name))

    ax.set_title("Bitcoin Historical Market Regimes & Structural Phases", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Price (USD, Log Scale)", fontsize=11)
    ax.set_yscale("log")
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    step = max(1, len(dates) // 8)
    ax.set_xticks(dates.iloc[::step])
    ax.set_xticklabels(dates.iloc[::step], rotation=30, ha="right")

    plt.tight_layout()
    plot_path = figures_dir / "market_regimes_timeline.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    logger.info(f"Market regimes plot saved to {plot_path}")


def run_regime_detection_pipeline(
    processed_data_path: Path = Path("data/processed/btc_daily.csv"),
    results_dir: Path = DEFAULT_RESULTS_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Runs complete market regime classification and persists summary reports."""
    logger.info("=== STEP 1: Feature Extraction for Market Regime Detection ===")
    raw_df = pd.read_csv(processed_data_path)
    feature_df = calculate_features(raw_df, include_target=False, drop_na=True)

    # Classify all rows
    regimes_series, states = detect_market_regimes(feature_df)
    latest_state = states[-1]

    # Plot
    plot_market_regimes(
        dates=feature_df["Date"],
        prices=feature_df["Close"].values,
        regimes=regimes_series,
        figures_dir=figures_dir,
    )

    # Historical distribution of regimes
    dist = regimes_series.value_counts(normalize=True) * 100.0
    dist_dict = {k: float(v) for k, v in dist.items()}

    regime_df = pd.DataFrame({
        "Date": feature_df["Date"].values,
        "Close": feature_df["Close"].values,
        "Regime": regimes_series.values,
    })

    results_dir.mkdir(parents=True, exist_ok=True)
    regime_csv_path = results_dir / "market_regimes_series.csv"
    regime_df.to_csv(regime_csv_path, index=False)

    summary = {
        "as_of_date": latest_state.date,
        "current_price": latest_state.price,
        "current_regime": latest_state.regime,
        "supporting_factors": {
            "trend": latest_state.trend_state,
            "momentum": latest_state.momentum_state,
            "volatility": latest_state.volatility_state,
            "drawdown": latest_state.drawdown_state,
        },
        "metrics": latest_state.metrics,
        "explanation": latest_state.explanation,
        "historical_regime_distribution_pct": dist_dict,
    }

    summary_path = results_dir / "market_regimes.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Current Market Regime ({latest_state.date}): {latest_state.regime} | Price=${latest_state.price:,.2f}")
    logger.info(f"Explanation: {latest_state.explanation}")

    return summary, regime_df


if __name__ == "__main__":
    run_regime_detection_pipeline()
