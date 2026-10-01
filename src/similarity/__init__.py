"""Historical pattern similarity search package for BTC Sentinel."""

from src.similarity.pattern_matcher import (
    HistoricalPatternMatcher,
    run_pattern_similarity_pipeline,
)

__all__ = ["HistoricalPatternMatcher", "run_pattern_similarity_pipeline"]
