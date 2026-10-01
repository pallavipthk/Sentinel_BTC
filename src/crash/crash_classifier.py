"""Crash & Drawdown Risk Early-Warning Classifier.
Implements:
1. Balanced Logistic Regression baseline.
2. Calibrated Random Forest / XGBoost classifier for forward drawdown prediction.
3. Probability calibration using isotonic / sigmoid scaling.
4. Explainable feature attributions for risk drivers.
5. Out-of-sample evaluation (ROC-AUC, PR-AUC, F1, Recall, Precision, Confusion Matrix).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler

from src.crash.drawdown_labeler import label_forward_drawdowns
from src.evaluation.metrics import calculate_classification_metrics
from src.features.feature_engineer import calculate_features, get_feature_columns
from src.utils.logger import setup_logger

logger = setup_logger("crash_classifier")

DEFAULT_RESULTS_DIR = Path("reports/results")
DEFAULT_FIGURES_DIR = Path("reports/figures")
DEFAULT_MODELS_DIR = Path("models")


class CrashRiskClassifier:
    """Predicts forward-looking severe drawdown probability using calibrated classifiers."""

    def __init__(self, model_type: str = "random_forest", calibrate: bool = True):
        self.model_type = model_type
        self.calibrate = calibrate
        self.scaler = StandardScaler()
        self.feature_names_: List[str] = []
        self.feature_importances_: Dict[str, float] = {}

        if model_type == "logistic_regression":
            self.base_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        else:
            self.base_model = RandomForestClassifier(
                n_estimators=250,
                max_depth=4,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=42,
            )

        if calibrate:
            self.model = CalibratedClassifierCV(estimator=self.base_model, method="sigmoid", cv=3)
        else:
            self.model = self.base_model

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "CrashRiskClassifier":
        """Fits scaler and calibrated classifier on training records."""
        self.feature_names_ = list(X_train.columns)
        X_scaled = self.scaler.fit_transform(X_train)

        logger.info(f"Fitting {self.model_type} crash classifier on {len(X_train)} samples ({int(y_train.sum())} positive crashes)...")
        self.model.fit(X_scaled, y_train.astype(int))

        # Extract feature importances
        if self.model_type == "random_forest":
            temp_base = RandomForestClassifier(n_estimators=100, max_depth=4, class_weight="balanced_subsample", random_state=42)
            temp_base.fit(X_scaled, y_train.astype(int))
            imps = temp_base.feature_importances_
            self.feature_importances_ = {
                feat: float(imp)
                for feat, imp in sorted(zip(self.feature_names_, imps), key=lambda x: x[1], reverse=True)
            }
        else:
            temp_base = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
            temp_base.fit(X_scaled, y_train.astype(int))
            coefs = np.abs(temp_base.coef_[0])
            self.feature_importances_ = {
                feat: float(imp)
                for feat, imp in sorted(zip(self.feature_names_, coefs), key=lambda x: x[1], reverse=True)
            }

        top_3 = list(self.feature_importances_.items())[:3]
        logger.info(f"Top 3 crash risk indicator features: {top_3}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts calibrated crash risk probability (class 1)."""
        X_scaled = self.scaler.transform(X[self.feature_names_])
        probs = self.model.predict_proba(X_scaled)[:, 1]
        return np.asarray(probs, dtype=float)

    def predict(self, X: pd.DataFrame, threshold: float = 0.50) -> np.ndarray:
        """Predicts binary crash risk label using probability threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def save(self, filepath: Path) -> None:
        """Serializes classifier state."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names_,
                "feature_importances": self.feature_importances_,
                "model_type": self.model_type,
                "calibrate": self.calibrate,
            },
            filepath,
        )
        logger.info(f"Crash risk classifier saved to {filepath}")


def plot_crash_curves(
    y_test: np.ndarray,
    lr_probs: np.ndarray,
    rf_probs: np.ndarray,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> None:
    """Generates ROC and Precision-Recall curves comparing Logistic Regression and Random Forest."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curves
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)

    ax1.plot(fpr_lr, tpr_lr, label="Logistic Regression", color="#6366f1", linewidth=1.5)
    ax1.plot(fpr_rf, tpr_rf, label="Calibrated Random Forest", color="#dc2626", linewidth=1.8)
    ax1.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af", label="Random Chance")
    ax1.set_title("Crash Early-Warning ROC Curve", fontsize=12, fontweight="bold")
    ax1.set_xlabel("False Positive Rate", fontsize=10)
    ax1.set_ylabel("True Positive Rate (Recall)", fontsize=10)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Precision-Recall Curves
    prec_lr, rec_lr, _ = precision_recall_curve(y_test, lr_probs)
    prec_rf, rec_rf, _ = precision_recall_curve(y_test, rf_probs)
    no_skill = np.sum(y_test) / len(y_test)

    ax2.plot(rec_lr, prec_lr, label="Logistic Regression", color="#6366f1", linewidth=1.5)
    ax2.plot(rec_rf, prec_rf, label="Calibrated Random Forest", color="#dc2626", linewidth=1.8)
    ax2.axhline(no_skill, linestyle="--", color="#9ca3af", label=f"Baseline Prevalence ({no_skill*100:.1f}%)")
    ax2.set_title("Crash Early-Warning Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Recall (Fraction of Crashes Detected)", fontsize=10)
    ax2.set_ylabel("Precision (Accuracy when Warning Issued)", fontsize=10)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    curve_plot_path = figures_dir / "crash_roc_pr_curve.png"
    fig.savefig(curve_plot_path, dpi=200)
    plt.close(fig)
    logger.info(f"Crash ROC/PR curve figure saved to {curve_plot_path}")


def plot_crash_risk_timeline(
    dates: pd.Series,
    prices: np.ndarray,
    estimated_risk: np.ndarray,
    actual_crashes: np.ndarray,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> None:
    """Visualizes model-estimated crash risk against Bitcoin price and actual crash periods."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})

    # Top: Bitcoin Price with shaded actual crash occurrences
    ax1.plot(dates, prices, color="#111827", linewidth=1.5, label="BTC Price (USD)")
    crash_indices = np.where(actual_crashes == 1.0)[0]
    if len(crash_indices) > 0:
        for idx in crash_indices:
            ax1.axvspan(dates.iloc[idx], dates.iloc[min(len(dates) - 1, idx + 1)], color="#ef4444", alpha=0.15)
    ax1.set_title("Bitcoin Price & Model-Estimated 14-Day Crash Risk Timeline", fontsize=14, fontweight="bold", pad=10)
    ax1.set_ylabel("Price (USD)", fontsize=11)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Bottom: Estimated Risk Probability
    ax2.plot(dates, estimated_risk * 100.0, color="#dc2626", linewidth=1.5, label="Model-Estimated Crash Risk (%)")
    ax2.axhline(50.0, color="#f59e0b", linestyle="--", linewidth=1.0, label="High Risk Threshold (50%)")
    ax2.fill_between(dates, estimated_risk * 100.0, color="#dc2626", alpha=0.2)
    ax2.set_ylabel("Estimated Risk (%)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylim(-5, 105)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle=":", alpha=0.6)

    step = max(1, len(dates) // 8)
    ax2.set_xticks(dates.iloc[::step])
    ax2.set_xticklabels(dates.iloc[::step], rotation=30, ha="right")

    plt.tight_layout()
    timeline_plot_path = figures_dir / "crash_risk_timeline.png"
    fig.savefig(timeline_plot_path, dpi=200)
    plt.close(fig)
    logger.info(f"Crash risk timeline saved to {timeline_plot_path}")


def run_crash_risk_pipeline(
    processed_data_path: Path = Path("data/processed/btc_daily.csv"),
    horizon_days: int = 14,
    drawdown_threshold: float = 0.10,
    test_ratio: float = 0.20,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Runs complete crash risk modeling, calibration, and evaluation pipeline."""
    logger.info("=== STEP 1: Feature Engineering & Forward Drawdown Labeling ===")
    raw_df = pd.read_csv(processed_data_path)

    # Compute leakage-safe features
    feature_df = calculate_features(raw_df, include_target=False, drop_na=True)

    # Label forward drawdowns
    labeled_df, label_summary = label_forward_drawdowns(
        feature_df,
        horizon_days=horizon_days,
        drawdown_threshold=drawdown_threshold,
    )

    feature_cols = get_feature_columns(labeled_df)
    # Remove future drawdown columns from features
    feature_cols = [c for c in feature_cols if c not in ["future_min_price", "future_max_drawdown", "target_crash"]]

    # Valid ground truth data (excluding the final horizon_days where future is unobserved)
    labeled_data = labeled_df.dropna(subset=["target_crash"]).reset_index(drop=True)

    # Chronological Split (Train 80% / Test 20%)
    n_labeled = len(labeled_data)
    split_idx = int(n_labeled * (1.0 - test_ratio))

    train_data = labeled_data.iloc[:split_idx].copy()
    test_data = labeled_data.iloc[split_idx:].copy()

    logger.info(
        f"Chronological Split for Crash Risk: Train={len(train_data)} days ({train_data['Date'].iloc[0]} to {train_data['Date'].iloc[-1]}), "
        f"Test={len(test_data)} days ({test_data['Date'].iloc[0]} to {test_data['Date'].iloc[-1]})"
    )

    X_train, y_train = train_data[feature_cols], train_data["target_crash"]
    X_test, y_test = test_data[feature_cols], test_data["target_crash"]

    # STEP 2: Train Model 1 (Logistic Regression Baseline)
    lr_clf = CrashRiskClassifier(model_type="logistic_regression", calibrate=True)
    lr_clf.fit(X_train, y_train)
    lr_probs = lr_clf.predict_proba(X_test)
    lr_preds = (lr_probs >= 0.50).astype(int)
    lr_metrics = calculate_classification_metrics(y_test.values, lr_preds, lr_probs)

    # STEP 3: Train Model 2 (Calibrated Random Forest)
    rf_clf = CrashRiskClassifier(model_type="random_forest", calibrate=True)
    rf_clf.fit(X_train, y_train)
    rf_probs = rf_clf.predict_proba(X_test)
    rf_preds = (rf_probs >= 0.50).astype(int)
    rf_metrics = calculate_classification_metrics(y_test.values, rf_preds, rf_probs)

    # STEP 4: Current Real-Time Crash Risk Score (using latest available row)
    latest_row = feature_df.iloc[[-1]][feature_cols]
    current_estimated_risk = float(rf_clf.predict_proba(latest_row)[0])
    current_date = str(feature_df["Date"].iloc[-1])
    current_price = float(feature_df["Close"].iloc[-1])

    logger.info(
        f"Current Bitcoin Status ({current_date}, ${current_price:,.2f}): "
        f"Model-Estimated 14-Day Crash Risk: {current_estimated_risk * 100.0:.1f}%"
    )

    # STEP 5: Visualizations
    plot_crash_curves(y_test.values, lr_probs, rf_probs, figures_dir=figures_dir)

    # Compute timeline risk over the full dataset
    full_probs = rf_clf.predict_proba(labeled_data[feature_cols])
    plot_crash_risk_timeline(
        dates=labeled_data["Date"],
        prices=labeled_data["Close"].values,
        estimated_risk=full_probs,
        actual_crashes=labeled_data["target_crash"].values,
        figures_dir=figures_dir,
    )

    # STEP 6: Save Artifacts
    test_pred_df = pd.DataFrame({
        "Date": test_data["Date"].values,
        "Close": test_data["Close"].values,
        "Actual_Crash": y_test.values,
        "Future_Max_Drawdown": test_data["future_max_drawdown"].values,
        "Logistic_Regression_Risk": lr_probs,
        "Random_Forest_Risk": rf_probs,
    })

    results_dir.mkdir(parents=True, exist_ok=True)
    test_pred_path = results_dir / "crash_risk_predictions.csv"
    test_pred_df.to_csv(test_pred_path, index=False)

    summary = {
        "definition": {
            "horizon_days": horizon_days,
            "drawdown_threshold_percent": drawdown_threshold * 100.0,
            "label_meaning": f"1 if maximum price drop over next {horizon_days} days exceeds {drawdown_threshold*100:.0f}%, else 0",
        },
        "dataset_summary": label_summary,
        "current_assessment": {
            "as_of_date": current_date,
            "current_price": current_price,
            "model_estimated_crash_risk_percent": float(current_estimated_risk * 100.0),
            "risk_level": "ELEVATED" if current_estimated_risk > 0.50 else ("MODERATE" if current_estimated_risk > 0.25 else "LOW"),
            "qualification": "Model-estimated probability of >= 10% drawdown within 14 days using calibrated Random Forest. Not a guaranteed forecast.",
        },
        "model_comparison": {
            "Logistic_Regression": lr_metrics,
            "Random_Forest": rf_metrics,
        },
        "top_crash_drivers": list(rf_clf.feature_importances_.items())[:10],
    }

    summary_path = results_dir / "crash_risk_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    rf_clf.save(models_dir / "crash_risk_model.joblib")

    logger.info(
        f"Crash Risk Models Evaluated: RF ROC-AUC={rf_metrics.get('roc_auc', 0.0):.3f}, "
        f"PR-AUC={rf_metrics.get('pr_auc', 0.0):.3f}, F1={rf_metrics['f1']:.3f}, Recall={rf_metrics['recall']:.3f}"
    )
    return summary, test_pred_df


if __name__ == "__main__":
    run_crash_risk_pipeline()
