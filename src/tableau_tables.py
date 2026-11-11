"""
Small tables for the Tableau dashboard: Early Flag, Boston 311.

Run: python src/tableau_tables.py
Input:  data/raw/boston311_2025.csv
Output: four small CSVs in reports/tableau/ — aggregated counts only,
        never the raw 276k rows. Tableau Public gets these four files.

The four tables match the four dashboard views:
1. overall.csv — on-time vs overdue, all closed 2025 requests.
2. overdue_by_type.csv — overdue rate for the 12 most common types.
3. missed_by_neighbourhood.csv — requests the logistic baseline missed
   (actually overdue, predicted on time) by neighbourhood, Oct-Dec test.
4. monthly_trend.csv — requests and overdue rate by open month, 2025.

Tables 1, 2 and 4 use all closed requests, the same as eda.py, so the
numbers match reports/eda_summary.md. Table 3 reuses the logistic
baseline and the Oct-Dec test set from error_analysis.py, so it
matches reports/error_analysis.md.
"""
import os
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features
from error_analysis import test_metadata

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw", "boston311_2025.csv")
TABLEAU = os.path.join(ROOT, "reports", "tableau")

USECOLS = ["open_dt", "on_time", "case_status", "case_title", "neighborhood"]


def closed_requests():
    """Closed 2025 requests, dates parsed with format mixed (see eda.py)."""
    df = pd.read_csv(RAW, usecols=USECOLS, low_memory=False)
    closed = df[df["case_status"] == "Closed"].copy()
    closed["open_dt"] = pd.to_datetime(closed["open_dt"], errors="coerce", format="mixed")
    closed = closed.dropna(subset=["open_dt"])
    closed["is_overdue"] = (closed["on_time"] == "OVERDUE").astype(int)
    closed["open_month"] = closed["open_dt"].dt.month
    return closed


def main():
    os.makedirs(TABLEAU, exist_ok=True)
    closed = closed_requests()

    # 1. Overall: on-time vs overdue.
    overall = (
        closed["is_overdue"]
        .value_counts()
        .rename(index={0: "ONTIME", 1: "OVERDUE"})
        .rename_axis("outcome")
        .reset_index(name="requests")
    )
    overall["share"] = overall["requests"] / overall["requests"].sum()
    overall.to_csv(os.path.join(TABLEAU, "overall.csv"), index=False)

    # 2. Overdue rate for the 12 most common request types.
    by_type = (
        closed.groupby("case_title")
        .agg(requests=("is_overdue", "size"), overdue=("is_overdue", "sum"))
        .sort_values("requests", ascending=False)
        .head(12)
        .reset_index()
    )
    by_type["overdue_rate"] = by_type["overdue"] / by_type["requests"]
    by_type.to_csv(os.path.join(TABLEAU, "overdue_by_type.csv"), index=False)

    # 3. Missed overdue requests by neighbourhood, Oct-Dec test set.
    x_train, x_test, y_train, y_test = build_features()
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    results = pd.DataFrame(
        {"actual": y_test.values, "predicted": pred}, index=y_test.index
    )
    results["missed"] = (results["actual"] == 1) & (results["predicted"] == 0)
    meta = test_metadata(y_test)
    results["neighborhood"] = meta["neighborhood"].values
    missed = []
    for name, group in results.groupby("neighborhood"):
        n_overdue = int((group["actual"] == 1).sum())
        if n_overdue < 20:
            continue
        missed.append({
            "neighborhood": name,
            "test_requests": len(group),
            "overdue": n_overdue,
            "missed": int(group["missed"].sum()),
        })
    missed = pd.DataFrame(missed).sort_values("missed", ascending=False)
    missed.to_csv(os.path.join(TABLEAU, "missed_by_neighbourhood.csv"), index=False)

    # 4. Monthly trend: requests and overdue rate by open month.
    monthly = (
        closed.groupby("open_month")
        .agg(requests=("is_overdue", "size"), overdue=("is_overdue", "sum"))
        .reset_index()
    )
    monthly["overdue_rate"] = monthly["overdue"] / monthly["requests"]
    monthly.to_csv(os.path.join(TABLEAU, "monthly_trend.csv"), index=False)

    print(overall.to_string(index=False))
    print()
    print(by_type.to_string(index=False))
    print()
    print(missed.to_string(index=False))
    print()
    print(monthly.to_string(index=False))
    print()
    print("Wrote 4 tables to reports/tableau/")


if __name__ == "__main__":
    main()
