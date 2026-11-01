"""
Features for Boston 311, intake only.

Run: python src/features.py
Input:  data/raw/boston311_2025.csv
Output: prints train/test sizes and overdue rates.
        Splits are built in memory by baselines.py, not saved as CSVs,
        because a 1,700-column CSV is too slow to write.

Time split: train Jan-Sep 2025, test Oct-Dec 2025.
Target: is_overdue = 1 if on_time is OVERDUE, else 0.
Only closed requests are used, because open requests have no outcome yet.
No closed_dt, closure_reason, case_status, or on_time is used as a feature.
"""
import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw", "boston311_2025.csv")
PROCESSED = os.path.join(ROOT, "data", "processed")

CAT = ["case_title", "subject", "reason", "type",
       "department", "neighborhood", "source"]
NUM = ["sla_days", "open_hour", "open_weekday", "open_month"]


def build_features(path=RAW):
    df = pd.read_csv(path, low_memory=False)
    closed = df[df["case_status"] == "Closed"].copy()
    for col in ["open_dt", "sla_target_dt"]:
        closed[col] = pd.to_datetime(closed[col], errors="coerce", format="mixed")
    closed["sla_days"] = (
        closed["sla_target_dt"] - closed["open_dt"]
    ).dt.total_seconds() / 86400
    closed["open_hour"] = closed["open_dt"].dt.hour
    closed["open_weekday"] = closed["open_dt"].dt.weekday
    closed["open_month"] = closed["open_dt"].dt.month
    closed["is_overdue"] = (closed["on_time"] == "OVERDUE").astype(int)
    closed = closed.dropna(subset=["sla_days", "open_dt"])

    train = closed[closed["open_dt"] < "2025-10-01"].copy()
    test = closed[closed["open_dt"] >= "2025-10-01"].copy()

    # Group rare categories as "Other": a category seen fewer than
    # 500 times in train becomes one shared column, instead of
    # hundreds of near-empty columns that make logistic regression slow.
    for col in CAT:
        keep = train[col].value_counts()
        keep = keep[keep >= 500].index
        train[col] = train[col].where(train[col].isin(keep), "Other")
        test[col] = test[col].where(test[col].isin(keep), "Other")

    x_train = pd.get_dummies(train[CAT + NUM])
    x_test = pd.get_dummies(test[CAT + NUM])
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)
    return x_train, x_test, train["is_overdue"], test["is_overdue"]


def main():
    x_train, x_test, y_train, y_test = build_features()
    print(f"Train: {len(x_train):,} rows, overdue {y_train.mean():.4f}")
    print(f"Test:  {len(x_test):,} rows, overdue {y_test.mean():.4f}")
    print(f"Features: {x_train.shape[1]}")


if __name__ == "__main__":
    main()
