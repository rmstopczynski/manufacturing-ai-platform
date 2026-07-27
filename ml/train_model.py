"""
Trains and evaluates the predictive maintenance classifier.

Two models are trained on purpose, not one:
    1. Logistic Regression - simple, interpretable baseline.
    2. XGBoost             - the model we actually ship.

Why bother with the baseline if we're shipping XGBoost anyway? Because "why
XGBoost over a simpler model" is a near-guaranteed interview question, and the
honest answer needs a number behind it, not just "because it's better." This
script produces that number (the actual precision/recall/F1 delta on this
data) so the README's design-decisions section isn't hand-waving.

Evaluation uses precision/recall/F1, NOT accuracy: failed_within_7d is a
~1% minority class, so a model that always predicts "healthy" would score
~99% accuracy while being completely useless. A time-based train/test split
is used (train on earlier days, test on later days) rather than a random
split, since a random split would leak future information about a machine's
failure into the training set via nearby rolling-window rows.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

FEATURES_PATH = "data/processed/features.csv"
MODEL_OUT_PATH = "ml/models/xgb_predictive_maintenance.joblib"
SCALER_OUT_PATH = "ml/models/logreg_scaler.joblib"

FEATURE_COLS = [
    "temperature_c", "vibration_mm_s", "runtime_hours", "days_since_maintenance",
    "temp_roll7_mean", "vib_roll7_mean", "temp_roll7_std", "vib_roll7_std",
    "temp_rate_of_change", "vib_rate_of_change",
    "type_CNC_MILL", "type_CONVEYOR", "type_PRESS", "type_PUMP", "type_ROBOT_ARM",
]
TARGET_COL = "failed_within_7d"


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("timestamp")
    unique_dates = sorted(df["timestamp"].unique())
    cutoff_idx = int(len(unique_dates) * (1 - test_frac))
    cutoff = unique_dates[cutoff_idx]
    train = df[df["timestamp"] < cutoff]
    test = df[df["timestamp"] >= cutoff]
    return train, test


def evaluate(name: str, y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n--- {name} ---")
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
    print(f"Confusion matrix [[TN FP] [FN TP]]:\n{cm}")
    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    df = pd.read_csv(FEATURES_PATH)
    # convert bool one-hot columns (from get_dummies) to int for model compatibility
    for c in FEATURE_COLS:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)

    train, test = time_based_split(df)
    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

    print(f"Train rows: {len(train)} (positives: {y_train.sum()})")
    print(f"Test rows:  {len(test)} (positives: {y_test.sum()})")

    # --- Baseline: Logistic Regression ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(class_weight="balanced", max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    logreg_preds = logreg.predict(X_test_scaled)
    logreg_metrics = evaluate("Logistic Regression (baseline)", y_test, logreg_preds)

    # --- Shipped model: XGBoost ---
    # scale_pos_weight compensates for class imbalance without oversampling,
    # roughly = (negative count / positive count) in the training set.
    pos = max(y_train.sum(), 1)
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_metrics = evaluate("XGBoost", y_test, xgb_preds)

    print("\n--- XGBoost classification report ---")
    print(classification_report(y_test, xgb_preds, zero_division=0))

    print("\n--- Feature importances (XGBoost) ---")
    importances = pd.Series(xgb.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(importances.to_string())

    # Persist artifacts the API will load at inference time
    joblib.dump(xgb, MODEL_OUT_PATH)
    joblib.dump(scaler, SCALER_OUT_PATH)  # kept for reference/baseline reproducibility
    print(f"\nSaved XGBoost model to {MODEL_OUT_PATH}")

    print("\n--- Summary: baseline vs shipped model ---")
    print(f"Logistic Regression -> precision={logreg_metrics['precision']:.3f}, "
          f"recall={logreg_metrics['recall']:.3f}, f1={logreg_metrics['f1']:.3f}")
    print(f"XGBoost             -> precision={xgb_metrics['precision']:.3f}, "
          f"recall={xgb_metrics['recall']:.3f}, f1={xgb_metrics['f1']:.3f}")


if __name__ == "__main__":
    main()
