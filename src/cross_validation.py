"""
Cross-validation for Boston 311.

Run: python src/cross_validation.py
Input: data/raw/boston311_2025.csv
Output: prints one row per fold and writes reports/cross_validation.md
        and reports/figures/cv_folds.png

Same logistic regression as baselines.py, same intake-only
features. Expanding windows inside Jan-Sep 2025 only, no
random split, so no later month leaks into training. Oct-Dec
2025 is not used here, it stays a fresh final check.

Fold 1: train Jan-Feb, check Mar-Apr
Fold 2: train Jan-Apr, check May-Jun
Fold 3: train Jan-Jun, check Jul-Sep
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw", "boston311_2025.csv")
REPORTS = os.path.join(ROOT, "reports")
FIGURES = os.path.join(REPORTS, "figures")

CAT = ["case_title", "subject", "reason", "type",
       "department", "neighborhood", "source"]
NUM = ["sla_days", "open_hour", "open_weekday", "open_month"]

FOLDS = [
    ("Jan-Feb", [1, 2], "Mar-Apr", [3, 4]),
    ("Jan-Apr", [1, 2, 3, 4], "May-Jun", [5, 6]),
    ("Jan-Jun", [1, 2, 3, 4, 5, 6], "Jul-Sep", [7, 8, 9]),
]


def load_closed():
    df = pd.read_csv(RAW, low_memory=False)
    closed = df[df["case_status"] == "Closed"].copy()
    for col in ["open_dt", "sla_target_dt"]:
        # Mixed formats, default parse dropped about a third of rows.
        closed[col] = pd.to_datetime(closed[col], errors="coerce", format="mixed")
    closed["sla_days"] = (
        closed["sla_target_dt"] - closed["open_dt"]
    ).dt.total_seconds() / 86400
    closed["open_hour"] = closed["open_dt"].dt.hour
    closed["open_weekday"] = closed["open_dt"].dt.weekday
    closed["open_month"] = closed["open_dt"].dt.month
    closed["is_overdue"] = (closed["on_time"] == "OVERDUE").astype(int)
    closed = closed.dropna(subset=["sla_days", "open_dt"])
    # Jan-Sep only. Oct-Dec is the final test set and is never used here.
    return closed[closed["open_dt"] < "2025-10-01"].copy()


def make_features(train, check):
    train = train.copy()
    check = check.copy()
    for col in CAT:
        keep = train[col].value_counts()
        keep = keep[keep >= 500].index
        train[col] = train[col].where(train[col].isin(keep), "Other")
        check[col] = check[col].where(check[col].isin(keep), "Other")
    x_train = pd.get_dummies(train[CAT + NUM])
    x_check = pd.get_dummies(check[CAT + NUM])
    x_check = x_check.reindex(columns=x_train.columns, fill_value=0)
    return x_train, x_check


def main():
    closed = load_closed()
    print(f"Training months (Jan-Sep 2025): {len(closed):,} closed requests")

    results = []
    for train_label, train_months, check_label, check_months in FOLDS:
        train = closed[closed["open_month"].isin(train_months)]
        check = closed[closed["open_month"].isin(check_months)]
        x_train, x_check = make_features(train, check)
        y_train = train["is_overdue"]
        y_check = check["is_overdue"]

        model = LogisticRegression(max_iter=1000)
        model.fit(x_train, y_train)
        pred = model.predict(x_check)
        row = {
            "fold": f"Train {train_label}, check {check_label}",
            "n_check": len(check),
            "overdue_rate": y_check.mean(),
            "accuracy": accuracy_score(y_check, pred),
            "f1": f1_score(y_check, pred, zero_division=0),
            "recall": recall_score(y_check, pred, zero_division=0),
        }
        results.append(row)
        print(f"{row['fold']}: n={row['n_check']:,}, "
              f"overdue {row['overdue_rate']:.4f}, "
              f"accuracy {row['accuracy']:.4f}, "
              f"F1 {row['f1']:.4f}, overdue recall {row['recall']:.4f}")

    mean_f1 = sum(r["f1"] for r in results) / len(results)
    mean_recall = sum(r["recall"] for r in results) / len(results)
    print(f"Mean across folds: F1 {mean_f1:.4f}, "
          f"overdue recall {mean_recall:.4f}")

    table = (
        "| Fold | Check requests | Overdue rate | Accuracy | "
        "F1 (overdue) | Overdue recall |\n"
        "|---|---|---|---|---|---|\n"
    )
    for r in results:
        table += (f"| {r['fold']} | {r['n_check']:,} "
                  f"| {r['overdue_rate']:.4f} | {r['accuracy']:.4f} "
                  f"| {r['f1']:.4f} | {r['recall']:.4f} |\n")
    table += (f"| Mean |  |  |  | {mean_f1:.4f} | {mean_recall:.4f} |\n")

    summary = (
        "# Cross-validation — Boston 311, 2025\n\n"
        "Logistic regression baseline, same intake-only features as "
        "baselines.py, checked with expanding windows inside the "
        "training months only (Jan-Sep 2025). Each fold trains on "
        "earlier months and is checked on the months right after, "
        "so no later month leaks into training. The Oct-Dec 2025 "
        "test set is not used here; it stays a fresh final check. "
        "For comparison, the single Oct-Dec test result was accuracy "
        "0.9104, F1 0.5676, overdue recall 0.4687.\n\n"
        + table
        + "\nIf the fold F1 and recall sit near the Oct-Dec numbers, "
        "the result is not just one lucky set of months. If a fold "
        "is far lower, that says something about those months, and "
        "it belongs in the write-up, not hidden.\n"
    )
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, "cross_validation.md"), "w") as f:
        f.write(summary)

    os.makedirs(FIGURES, exist_ok=True)
    labels = [r["fold"].replace("Train ", "") for r in results]
    x = range(len(results))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - 0.2 for i in x], [r["f1"] for r in results],
           width=0.4, label="F1 (overdue)")
    ax.bar([i + 0.2 for i in x], [r["recall"] for r in results],
           width=0.4, label="Overdue recall")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Cross-validation: F1 and overdue recall by fold")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "cv_folds.png"))
    print("Wrote reports/cross_validation.md and "
          "reports/figures/cv_folds.png")


if __name__ == "__main__":
    main()
