"""
SVM baseline for Boston 311.

Run: python src/svm.py
Input:  data/raw/boston311_2025.csv, through src/features.py
Output: prints a results table and writes reports/svm_results.md

Uses LinearSVC (a linear SVM), not SVC with an RBF kernel, because an
RBF kernel on about 149,000 training rows is far too slow.
This uses the same intake-only features and the same time-based split as
baselines.py: train Jan-Sep 2025, test Oct-Dec 2025, so its F1 and
overdue recall compare directly with logistic regression.
Only closed requests are used, because open requests have no outcome yet.
"""
import os
import sys

from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.svm import LinearSVC

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

    model = LinearSVC(dual=False, max_iter=2000)
    model.fit(x_train, y_train)
    svm = score(y_test, model.predict(x_test))

    table = (
        "| Model | Accuracy | F1 (overdue) | Overdue recall |\n"
        "|---|---|---|---|\n"
        f"| SVM (linear) | {svm['accuracy']:.4f} "
        f"| {svm['f1']:.4f} | {svm['recall']:.4f} |\n"
    )
    summary = (
        "# SVM results — Boston 311, 2025\n\n"
        "Same intake-only features and same time-based split as baselines.py: "
        f"train Jan-Sep 2025 ({len(x_train):,} rows, "
        f"overdue {y_train.mean():.4f}), test Oct-Dec 2025 "
        f"({len(x_test):,} rows, overdue {y_test.mean():.4f}).\n\n"
        + table
        + "\nCompare this row with reports/baseline_results.md on "
        "F1 (overdue) and overdue recall, not accuracy alone, because "
        "OVERDUE is only about 12% of closed requests.\n"
    )
    print(summary)
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, "svm_results.md"), "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
