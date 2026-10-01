"""Standard evaluation metrics for time-series forecasting and risk classification.
All calculations are deterministic and purely derived from actual model outputs.
"""

from typing import Dict, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Symmetric Mean Absolute Percentage Error (sMAPE) as a percentage (0-100%).
    Formula: 100% / n * sum( 2 * |y_pred - y_true| / (|y_true| + |y_pred| + eps) )
    """
    denominator = np.abs(y_true) + np.abs(y_pred) + 1e-10
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denominator) * 100.0)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Mean Absolute Percentage Error (MAPE) as a percentage (0-100%)."""
    denominator = np.abs(y_true) + 1e-10
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)


def calculate_directional_accuracy(
    actual_returns: np.ndarray,
    predicted_returns: np.ndarray,
) -> float:
    """Calculates directional accuracy (hit rate) as percentage (0-100%).
    Measures how often predicted return direction matches actual return direction.
    """
    actual_dir = np.sign(actual_returns)
    pred_dir = np.sign(predicted_returns)
    # Exclude zero return if any, or treat equal sign as 1
    correct = (actual_dir == pred_dir)
    return float(np.mean(correct) * 100.0)


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    current_prices: Optional[np.ndarray] = None,
    is_return_target: bool = False,
) -> Dict[str, float]:
    """Calculates comprehensive price/return regression metrics.

    Args:
        y_true: Ground truth target values (either prices or returns).
        y_pred: Predicted values (either prices or returns).
        current_prices: Optional array of prices at time t (used if y is returns to compute price metrics).
        is_return_target: Flag indicating if y represents returns rather than price levels.

    Returns:
        Dictionary of computed metric names and values.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))

    metrics = {
        "mae": mae,
        "rmse": rmse,
    }

    if is_return_target:
        # y is returns
        directional_acc = calculate_directional_accuracy(y_t, y_p)
        metrics["directional_accuracy"] = directional_acc

        if current_prices is not None:
            c_p = np.asarray(current_prices, dtype=float)
            price_true = c_p * (1.0 + y_t)
            price_pred = c_p * (1.0 + y_p)

            metrics["price_mae"] = float(mean_absolute_error(price_true, price_pred))
            metrics["price_rmse"] = float(np.sqrt(np.mean((price_true - price_pred) ** 2)))
            metrics["price_mape"] = calculate_mape(price_true, price_pred)
            metrics["price_smape"] = calculate_smape(price_true, price_pred)
    else:
        # y is raw prices
        metrics["mape"] = calculate_mape(y_t, y_p)
        metrics["smape"] = calculate_smape(y_t, y_p)

        if current_prices is not None:
            c_p = np.asarray(current_prices, dtype=float)
            actual_ret = (y_t - c_p) / (c_p + 1e-10)
            pred_ret = (y_p - c_p) / (c_p + 1e-10)
            metrics["directional_accuracy"] = calculate_directional_accuracy(actual_ret, pred_ret)

    return metrics


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculates classification metrics for crash / regime prediction.

    Args:
        y_true: True binary labels (0 or 1).
        y_pred: Predicted discrete labels (0 or 1).
        y_prob: Predicted probabilities for positive class (0.0 to 1.0).

    Returns:
        Dictionary of classification metrics.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

    if y_prob is not None and len(np.unique(y_t)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_t, y_prob))
        except Exception:
            metrics["roc_auc"] = None
        try:
            metrics["pr_auc"] = float(average_precision_score(y_t, y_prob))
        except Exception:
            metrics["pr_auc"] = None

    cm = confusion_matrix(y_t, y_p).tolist()
    metrics["confusion_matrix"] = cm

    return metrics
