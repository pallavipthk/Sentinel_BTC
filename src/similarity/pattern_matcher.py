"""Historical Pattern Similarity Search Engine.
Finds top similar historical Bitcoin market regimes and trajectories:
1. Normalizes the latest 30-day window to shape representation.
2. Extracts non-overlapping candidate historical windows.
3. Computes shape correlation and distance metrics.
4. Retrieves top matches and displays subsequent returns (7d, 14d, 30d).
5. Visualizes pattern overlay and post-pattern trajectories.
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

from src.utils.logger import setup_logger

logger = setup_logger("pattern_matcher")

DEFAULT_RESULTS_DIR = Path("reports/results")
DEFAULT_FIGURES_DIR = Path("reports/figures")


@dataclass
class PatternMatch:
    """Historical pattern match details."""
    rank: int
    match_start_date: str
    match_end_date: str
    end_price: float
    similarity_score_pct: float
    correlation: float
    normalized_distance: float
    subsequent_return_7d_pct: Optional[float]
    subsequent_return_14d_pct: Optional[float]
    subsequent_return_30d_pct: Optional[float]
    subsequent_max_drawdown_pct: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HistoricalPatternMatcher:
    """Finds analogous historical price regimes and analyzes forward outcomes."""

    def __init__(self, window_size: int = 30, min_separation_days: int = 45):
        self.window_size = window_size
        self.min_separation = min_separation_days

    def _normalize_window(self, prices: np.ndarray) -> np.ndarray:
        """Normalizes window to percentage return trajectory anchored at start: (P_i / P_0) - 1.0"""
        p0 = prices[0]
        return (prices / (p0 + 1e-10)) - 1.0

    def find_matches(
        self,
        df: pd.DataFrame,
        top_k: int = 4,
        query_end_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[PatternMatch], Dict[str, Any]]:
        """Searches historical dataset for the most similar windows to the query pattern."""
        prices = df["Close"].values
        dates = df["Date"].values
        n = len(prices)

        if query_end_idx is None:
            query_end_idx = n - 1

        query_start_idx = query_end_idx - self.window_size + 1
        if query_start_idx < 0:
            raise ValueError(f"Insufficient data for window size {self.window_size}")

        query_prices = prices[query_start_idx : query_end_idx + 1]
        query_norm = self._normalize_window(query_prices)
        query_dates = [str(dates[query_start_idx]), str(dates[query_end_idx])]

        # Exclude candidates that overlap with query window (buffer = min_separation)
        latest_valid_candidate_end = query_start_idx - self.min_separation
        if latest_valid_candidate_end < self.window_size:
            raise ValueError("Insufficient history outside query window to perform similarity search.")

        candidates = []
        for i in range(self.window_size - 1, latest_valid_candidate_end + 1):
            start_i = i - self.window_size + 1
            cand_prices = prices[start_i : i + 1]
            cand_norm = self._normalize_window(cand_prices)

            # Pearson correlation
            std_q = np.std(query_norm)
            std_c = np.std(cand_norm)
            if std_q > 1e-6 and std_c > 1e-6:
                corr = float(np.corrcoef(query_norm, cand_norm)[0, 1])
            else:
                corr = 0.0

            # Euclidean distance on normalized shapes
            dist = float(np.linalg.norm(query_norm - cand_norm))

            # Composite similarity score (scaled 0-100%)
            # High correlation + small shape distance
            sim_score = max(0.0, min(100.0, ((corr + 1.0) / 2.0) * 100.0 - (dist * 5.0)))

            candidates.append({
                "end_idx": i,
                "start_idx": start_i,
                "correlation": corr,
                "distance": dist,
                "sim_score": sim_score,
            })

        # Sort candidates by similarity score descending
        candidates.sort(key=lambda x: x["sim_score"], reverse=True)

        # Select top non-overlapping matches
        selected = []
        for cand in candidates:
            # Check minimum separation from already selected candidates
            overlap = False
            for s in selected:
                if abs(cand["end_idx"] - s["end_idx"]) < self.min_separation:
                    overlap = True
                    break
            if not overlap:
                selected.append(cand)
            if len(selected) >= top_k:
                break

        # Calculate subsequent returns for each selected match
        matches: List[PatternMatch] = []
        for rank, match in enumerate(selected, start=1):
            e_idx = match["end_idx"]
            s_idx = match["start_idx"]
            match_p = prices[e_idx]

            # Forward returns
            r7 = float((prices[e_idx + 7] - match_p) / match_p * 100.0) if e_idx + 7 < n else None
            r14 = float((prices[e_idx + 14] - match_p) / match_p * 100.0) if e_idx + 14 < n else None
            r30 = float((prices[e_idx + 30] - match_p) / match_p * 100.0) if e_idx + 30 < n else None

            # Max drawdown over next 30 days
            if e_idx + 30 < n:
                fwd_slice = prices[e_idx + 1 : e_idx + 31]
                min_fwd = np.min(fwd_slice)
                mdd = float((min_fwd - match_p) / match_p * 100.0)
            else:
                mdd = None

            matches.append(PatternMatch(
                rank=rank,
                match_start_date=str(dates[s_idx]),
                match_end_date=str(dates[e_idx]),
                end_price=float(match_p),
                similarity_score_pct=round(match["sim_score"], 1),
                correlation=round(match["correlation"], 3),
                normalized_distance=round(match["distance"], 3),
                subsequent_return_7d_pct=round(r7, 2) if r7 is not None else None,
                subsequent_return_14d_pct=round(r14, 2) if r14 is not None else None,
                subsequent_return_30d_pct=round(r30, 2) if r30 is not None else None,
                subsequent_max_drawdown_pct=round(mdd, 2) if mdd is not None else None,
            ))

        metadata = {
            "window_size_days": self.window_size,
            "query_period": query_dates,
            "query_start_price": float(query_prices[0]),
            "query_end_price": float(query_prices[-1]),
            "query_total_return_pct": float((query_prices[-1] - query_prices[0]) / query_prices[0] * 100.0),
            "disclaimer": "Historical similarity is an analytical research aid and NOT a guaranteed prediction of future market behavior.",
        }

        return query_norm, matches, metadata


def plot_pattern_similarity(
    df: pd.DataFrame,
    query_norm: np.ndarray,
    matches: List[PatternMatch],
    window_size: int = 30,
    fwd_days: int = 30,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> None:
    """Plots the query pattern alongside top historical patterns aligned at day 0 and forward trajectories."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7))

    day_axis = np.arange(window_size)
    ax.plot(day_axis, query_norm * 100.0, color="#111827", linewidth=2.8, label="Current Pattern (Latest 30 Days)", zorder=10)

    colors = ["#2563eb", "#d97706", "#7c3aed", "#059669"]

    dates = list(df["Date"])
    prices = df["Close"].values

    for i, match in enumerate(matches):
        c = colors[i % len(colors)]
        try:
            e_idx = dates.index(match.match_end_date)
            s_idx = e_idx - window_size + 1
            hist_p = prices[s_idx : e_idx + 1]
            p0 = hist_p[0]
            hist_norm = (hist_p / p0 - 1.0) * 100.0

            lbl = f"#{match.rank}: {match.match_end_date} (Sim: {match.similarity_score_pct}%, +30d: {match.subsequent_return_30d_pct:+.1f}%)"
            ax.plot(day_axis, hist_norm, color=c, linewidth=1.8, linestyle="--", label=lbl, alpha=0.9)

            # Subsequent forward trajectory
            if e_idx + fwd_days < len(prices):
                fwd_p = prices[e_idx : e_idx + fwd_days + 1]
                fwd_axis = np.arange(window_size - 1, window_size + fwd_days)
                # Align to the match's normalized endpoint
                fwd_norm = (fwd_p / p0 - 1.0) * 100.0
                ax.plot(fwd_axis, fwd_norm, color=c, linewidth=1.2, linestyle=":", alpha=0.6)
        except Exception as e:
            logger.warning(f"Could not render match {match.match_end_date}: {e}")

    ax.axvline(window_size - 1, color="#6b7280", linestyle="-.", label="Pattern End / Forward Horizon Boundary")
    ax.set_title("Bitcoin Historical Pattern Similarity & Forward Subsequent Outcomes", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Cumulative Normalized Return (%)", fontsize=11)
    ax.set_xlabel("Relative Timeline (Days from Window Start)", fontsize=11)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig_path = figures_dir / "pattern_similarity_overlay.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    logger.info(f"Pattern similarity overlay saved to {fig_path}")


def run_pattern_similarity_pipeline(
    processed_data_path: Path = Path("data/processed/btc_daily.csv"),
    window_size: int = 30,
    top_k: int = 4,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Runs complete pattern similarity search and outcome profiling."""
    logger.info("=== STEP 1: Running Historical Pattern Similarity Search ===")
    df = pd.read_csv(processed_data_path)

    matcher = HistoricalPatternMatcher(window_size=window_size)
    query_norm, matches, metadata = matcher.find_matches(df, top_k=top_k)

    # Plot
    plot_pattern_similarity(df, query_norm, matches, window_size=window_size, figures_dir=figures_dir)

    # Save artifacts
    matches_records = [m.to_dict() for m in matches]
    matches_df = pd.DataFrame(matches_records)

    results_dir.mkdir(parents=True, exist_ok=True)
    matches_csv_path = results_dir / "pattern_similarity_matches.csv"
    matches_df.to_csv(matches_csv_path, index=False)

    summary = {
        "metadata": metadata,
        "top_matches": matches_records,
    }

    summary_path = results_dir / "pattern_similarity_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Top 3 Similar Historical Periods to Current BTC Pattern ({metadata['query_period'][0]} to {metadata['query_period'][1]}):")
    for m in matches[:3]:
        logger.info(
            f"Rank #{m.rank}: {m.match_end_date} | Sim={m.similarity_score_pct}% | "
            f"7d Return={m.subsequent_return_7d_pct:+.1f}% | 14d Return={m.subsequent_return_14d_pct:+.1f}% | 30d Return={m.subsequent_return_30d_pct:+.1f}%"
        )

    return summary, matches_df


if __name__ == "__main__":
    run_pattern_similarity_pipeline()
