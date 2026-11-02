# SVM results — Boston 311, 2025

Same intake-only features and same time-based split as baselines.py: train Jan-Sep 2025 (148,675 rows, overdue 0.1216), test Oct-Dec 2025 (38,869 rows, overdue 0.1254).

| Model | Accuracy | F1 (overdue) | Overdue recall |
|---|---|---|---|
| SVM (linear) | 0.9102 | 0.5648 | 0.4644 |

Compare this row with reports/baseline_results.md on F1 (overdue) and overdue recall, not accuracy alone, because OVERDUE is only about 12% of closed requests.
