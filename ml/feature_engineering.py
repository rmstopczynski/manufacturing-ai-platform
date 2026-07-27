"""
Feature engineering for the predictive maintenance model.

Takes the raw daily sensor log (one row per machine per day) and builds features
that capture TRENDS, not just point-in-time readings, since a single day's
temperature/vibration reading in isolation is a weak signal — what actually
predicts failure is the machine drifting away from its own baseline over time.

Features engineered per (machine_id, day):
    temp_roll7_mean / vib_roll7_mean   - 7-day rolling average (smooths noise)
    temp_roll7_std  / vib_roll7_std    - 7-day rolling volatility
    temp_rate_of_change / vib_rate_of_change - value today minus value 7 days ago
    days_since_maintenance             - already in raw data, carried through
    runtime_hours                      - already in raw data, carried through
    machine_type                       - one-hot encoded

Output: data/processed/features.csv
"""

import pandas as pd

RAW_PATH = "data/raw/sensor_logs.csv"
OUT_PATH = "data/processed/features.csv"
ROLL_WINDOW = 7


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    df["temp_roll7_mean"] = (
        df.groupby("machine_id")["temperature_c"]
        .transform(lambda s: s.rolling(ROLL_WINDOW, min_periods=1).mean())
    )
    df["vib_roll7_mean"] = (
        df.groupby("machine_id")["vibration_mm_s"]
        .transform(lambda s: s.rolling(ROLL_WINDOW, min_periods=1).mean())
    )
    df["temp_roll7_std"] = (
        df.groupby("machine_id")["temperature_c"]
        .transform(lambda s: s.rolling(ROLL_WINDOW, min_periods=1).std())
        .fillna(0)
    )
    df["vib_roll7_std"] = (
        df.groupby("machine_id")["vibration_mm_s"]
        .transform(lambda s: s.rolling(ROLL_WINDOW, min_periods=1).std())
        .fillna(0)
    )

    df["temp_rate_of_change"] = (
        df.groupby("machine_id")["temperature_c"]
        .transform(lambda s: s.diff(ROLL_WINDOW))
        .fillna(0)
    )
    df["vib_rate_of_change"] = (
        df.groupby("machine_id")["vibration_mm_s"]
        .transform(lambda s: s.diff(ROLL_WINDOW))
        .fillna(0)
    )

    # One-hot encode machine_type — different machine types have different
    # baseline operating ranges, so the model needs this as context, not just
    # the raw sensor values.
    df = pd.get_dummies(df, columns=["machine_type"], prefix="type")

    return df


def main():
    df = pd.read_csv(RAW_PATH)
    features_df = build_features(df)
    features_df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(features_df)} rows, {features_df.shape[1]} columns to {OUT_PATH}")
    print(features_df.columns.tolist())


if __name__ == "__main__":
    main()
