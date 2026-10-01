"""Unit tests for Historical Pattern Similarity Matcher."""

import numpy as np
import pandas as pd
import pytest

from src.similarity.pattern_matcher import HistoricalPatternMatcher


@pytest.fixture
def synthetic_pattern_dataset():
    """Generates synthetic dataset with repeating sinusoidal and trending cycles."""
    np.random.seed(42)
    n = 350
    dates = pd.date_range("2022-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
    t = np.linspace(0, 10 * np.pi, n)
    # Price with oscillation and upward drift
    prices = 20000.0 + (5000.0 * np.sin(t)) + (50.0 * np.arange(n))
    return pd.DataFrame({"Date": dates, "Close": prices})


def test_pattern_matcher_finds_similar_periods(synthetic_pattern_dataset):
    """Verifies that pattern matcher identifies similar cyclical phases."""
    matcher = HistoricalPatternMatcher(window_size=30, min_separation_days=30)
    query_norm, matches, meta = matcher.find_matches(synthetic_pattern_dataset, top_k=3)

    assert len(query_norm) == 30
    assert len(matches) == 3
    assert matches[0].rank == 1
    assert matches[0].similarity_score_pct >= matches[1].similarity_score_pct
    assert matches[1].similarity_score_pct >= matches[2].similarity_score_pct

    # Ensure matches are separated in time (no self-overlap)
    d1 = pd.to_datetime(matches[0].match_end_date)
    d2 = pd.to_datetime(matches[1].match_end_date)
    assert abs((d1 - d2).days) >= 30

    # Ensure subsequent returns are populated
    assert matches[0].subsequent_return_7d_pct is not None
    assert matches[0].subsequent_return_14d_pct is not None
