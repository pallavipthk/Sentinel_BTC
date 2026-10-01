"""Unit tests for Crash & Forward Drawdown Risk Engine."""

import numpy as np
import pandas as pd
import pytest

from src.crash.crash_classifier import CrashRiskClassifier
from src.crash.drawdown_labeler import label_forward_drawdowns


def test_forward_drawdown_labeling():
    """Verifies forward-looking drawdown labeling and boundary NaNs."""
    # Day 0: 100
    # Day 1: 95
    # Day 2: 85 (15% drop from Day 0, triggers >= 10% crash)
    # Day 3: 110
    # Day 4: 120
    df = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
        "Close": [100.0, 95.0, 85.0, 110.0, 120.0],
    })

    labeled, summary = label_forward_drawdowns(df, horizon_days=2, drawdown_threshold=0.10)

    # Day 0 sees next 2 days: min is 85 -> drawdown is (85-100)/100 = -0.15 <= -0.10 -> label 1.0
    assert labeled.loc[0, "target_crash"] == 1.0
    # Day 2 sees next 2 days (Day 3: 110, Day 4: 120): min is 110 -> drawdown is (110-85)/85 > 0 -> label 0.0
    assert labeled.loc[2, "target_crash"] == 0.0
    # Final horizon (days 3 and 4) have incomplete future window, must be NaN
    assert np.isnan(labeled.loc[3, "target_crash"])
    assert np.isnan(labeled.loc[4, "target_crash"])


def test_crash_classifier_fit_and_calibrate():
    """Tests calibrated probability predictions and metric generation."""
    np.random.seed(42)
    n = 150
    X = pd.DataFrame({
        "vol": np.random.normal(0, 1, size=n),
        "drawdown": np.random.normal(0, 1, size=n),
    })
    # Probabilistic label
    logits = 1.5 * X["vol"] - 1.2 * X["drawdown"]
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = pd.Series((probs > 0.5).astype(int))

    clf = CrashRiskClassifier(model_type="random_forest", calibrate=True)
    clf.fit(X, y)

    preds_prob = clf.predict_proba(X)
    assert len(preds_prob) == n
    assert np.all(preds_prob >= 0.0)
    assert np.all(preds_prob <= 1.0)
    assert len(clf.feature_importances_) == 2
