"""
Error check for Boston 311: which types / neighbourhoods get missed?

Run: python src/error_analysis.py
Input:  data/raw/boston311_2025.csv, through src/features.py
Output: prints a summary and writes reports/error_analysis.md

"Missed" means the request actually went OVERDUE but the model
predicted ONTIME (a false negative). That is the error that matters
most here, because the point of the model is to flag likely-late
requests early.

Uses the logistic regression baseline from baselines.py, the same
intake-only features and the same time-based split: train Jan-Sep
2025, test Oct-Dec 2025. Logistic and the linear SVM were almost
tied on the test set, so logistic is used here as the representative
baseline; a deeper check for both models comes on Nov 7.

Only groups with at least 20 overdue requests in the test set are
shown, so a group with 2 overdue requests does not look like a
big problem just because both were missed.
"""
import os
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw", "boston311_2025.csv")
REPORTS = os.path.join(ROOT, "reports")


def test_metadata(y_test):
    """
    Return case_title / neighborhood for the test rows.

    build_features returns only the feature matrix and the target,
    so the labels are reloaded here from the raw file using exactly
    the same filters (closed only, valid dates, opened Oct-Dec 2025)
    and matched back by row index, which build_features preserves.
    """
    df = pd.read_csv(
        RAW,
        low_memory=False,
        usecols=["case_status", "open_dt", "sla_target_dt",
                 "case_title", "neighborhood"],
    )
    closed = df[df["case_status"] == "Closed"].copy()
    for col in ["open_dt", "sla_target_dt"]:
        closed[col] = pd.to_datetime(closed[col], errors="coerce", format="mixed")
    closed["sla_days"] = (
        closed["sla_target_dt"] - closed["open_dt"]
    ).dt.total_seconds() / 86400
    closed = closed.dropna(subset=["sla_days", "open_dt"])
    test = closed[closed["open_dt"] >= "2025-10-01"]
    meta = test.loc[y_test.index, ["case_title", "neighborhood"]]
    return meta.fillna("Unknown")


def breakdown(results, col):
    """One row per group: test size, overdue count, missed, recall."""
    rows = []
    for name, group in results.groupby(col):
        n_overdue = int((group["actual"] == 1).sum())
        if n_overdue < 20:
            continue
        missed = int(group["missed"].sum())
        rows.append({
            "group": name,
            "n_test": len(group),
            "n_overdue": n_overdue,
            "missed": missed,
            "recall": 1 - missed / n_overdue,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["share_of_missed"] = table["missed"] / table["missed"].sum()
    return table.sort_values("missed", ascending=False)


def table_md(table, title_col):
    lines = [
        f"| {title_col} | Test requests | Overdue | Missed | Overdue recall |",
        "|---|---|---|---|---|",
    ]
    for _, row in table.head(10).iterrows():
        lines.append(
            f"| {row['group']} | {int(row['n_test']):,} "
            f"| {int(row['n_overdue']):,} | {int(row['missed']):,} "
            f"| {row['recall']:.4f} |"
        )
    return "\n".join(lines)


def main():
    x_train, x_test, y_train, y_test = build_features()

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    results = pd.DataFrame(
        {"actual": y_test.values, "predicted": pred}, index=y_test.index
    )
    results["missed"] = (results["actual"] == 1) & (results["predicted"] == 0)
    meta = test_metadata(y_test)
    results["case_title"] = meta["case_title"].values
    results["neighborhood"] = meta["neighborhood"].values

    total_overdue = int((results["actual"] == 1).sum())
    total_missed = int(results["missed"].sum())
    overall_recall = 1 - total_missed / total_overdue

    by_type = breakdown(results, "case_title")
    by_area = breakdown(results, "neighborhood")

    summary = (
        "# Error check — Boston 311, 2025\n\n"
        "Logistic regression baseline, same intake-only features and "
        "same time-based split as baselines.py: train Jan-Sep 2025, "
        f"test Oct-Dec 2025 ({len(results):,} rows).\n\n"
        f"Overdue in test: {total_overdue:,}. Missed (predicted ONTIME): "
        f"{total_missed:,}. Overall overdue recall: {overall_recall:.4f}.\n\n"
        "## Most missed request types\n\n"
        + table_md(by_type, "Request type")
        + "\n\n## Most missed neighbourhoods\n\n"
        + table_md(by_area, "Neighbourhood")
        + "\n\nOnly groups with at least 20 overdue requests in the test "
        "set are shown, sorted by number missed, not by recall alone, "
        "so the biggest groups of missed requests come first.\n"
    )
    print(summary)
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, "error_analysis.md"), "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
