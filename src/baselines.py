"""
Majority baseline and logistic regression for Boston 311.

Run: python src/baselines.py
Input:  data/raw/boston311_2025.csv, through src/features.py
Output: prints a results table and writes reports/baseline_results.md

Both models use the same intake-only features and the same time-based
split from features.py: train Jan-Sep 2025, test Oct-Dec 2025.
Only closed requests are used, because open requests have no outcome yet.
"""
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features

ROOT = os.path.join(os.path.dirname(__file__), "..")
REPORTS = os.path.join(ROOT, "reports")


def score(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def main():
    x_train, x_test, y_train, y_test = build_features()

    majority_pred = np.zeros(len(y_test), dtype=int)
    majority = score(y_test, majority_pred)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    logistic = score(y_test, model.predict(x_test))

    table = (
    "| Model | Accuracy | F1 (overdue) | Overdue recall |\n"
    "|---|---|---|---|\n"
    f"| Majority (always ONTIME) | {majority['accuracy']:.4f} | {majority['f1']:.4f} | {majority['recall']:.4f} |\n"
    f"| Logistic regression | {logistic['accuracy']:.4f} | {logistic['f1']:.4f} | {logistic['recall']:.4f} |\n"
)

    summary = (
        "# Baseline results — Boston 311, 2025\n\n"
        "Same intake-only features and same time-based split for both models: "
        f"train Jan-Sep 2025 ({len(x_train):,} rows, "
        f"overdue {y_train.mean():.4f}), test Oct-Dec 2025 "
        f"({len(x_test):,} rows, overdue {y_test.mean():.4f}).\n\n"
        + table
    )
    print(summary)
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, "baseline_results.md"), "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
