"""
Loads the trained XGBoost model and produces a prediction for a specific machine ID,
using its most recent row of engineered features. This is what the orchestration
layer calls for the "(a) ML prediction" branch of routing.

Deliberately imports FEATURE_COLS from train_model.py rather than redefining the list
here — if the feature set ever changes, there is exactly one place to update it, and
prediction can't silently drift out of sync with what the model was trained on.
"""

import pandas as pd
import joblib

from ml.train_model import FEATURE_COLS

FEATURES_PATH = "data/processed/features.csv"
MODEL_PATH = "ml/models/xgb_predictive_maintenance.joblib"

_model = None
_features_df = None


def _load():
    global _model, _features_df
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _features_df is None:
        df = pd.read_csv(FEATURES_PATH)
        for c in FEATURE_COLS:
            if df[c].dtype == bool:
                df[c] = df[c].astype(int)
        _features_df = df
    return _model, _features_df


def predict_machine(machine_id: str) -> dict:
    """Returns a prediction dict for the given machine_id using its latest reading, or
    an error dict if the machine_id doesn't exist in the data."""
    model, df = _load()

    machine_rows = df[df["machine_id"] == machine_id]
    if machine_rows.empty:
        return {
            "machine_id": machine_id,
            "found": False,
            "error": f"No sensor data found for machine ID '{machine_id}'. "
                     f"It may not exist, or may not have been logged in the current dataset.",
        }

    latest = machine_rows.sort_values("timestamp").iloc[[-1]]
    X = latest[FEATURE_COLS].astype(float)

    proba = float(model.predict_proba(X)[0][1])
    pred = int(proba >= 0.5)
    latest_row = latest.iloc[0]

    # Derive machine_type from the one-hot type_* columns (the original categorical
    # column doesn't survive pd.get_dummies) so the caller can use it to filter
    # retrieval even when the question itself never names the machine type.
    type_cols = [c for c in FEATURE_COLS if c.startswith("type_")]
    active_type = next((c.replace("type_", "") for c in type_cols if latest_row[c] == 1), None)

    # Surface which engineered features are furthest from a "typical healthy" value,
    # so the LLM layer has something concrete to explain rather than just a probability.
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    top_features = importances.sort_values(ascending=False).head(3).index.tolist()
    top_feature_values = {f: round(float(latest_row[f]), 3) for f in top_features}

    return {
        "machine_id": machine_id,
        "found": True,
        "machine_type": active_type,
        "failure_predicted_within_7d": bool(pred),
        "failure_probability": round(proba, 3),
        "latest_reading_date": latest_row["timestamp"],
        "days_since_maintenance": int(latest_row["days_since_maintenance"]),
        "top_contributing_features": top_feature_values,
    }


if __name__ == "__main__":
    for mid in ["M001", "M002", "M999"]:
        print(predict_machine(mid))
