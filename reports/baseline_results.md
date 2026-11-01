# Baseline results — Boston 311, 2025

Same intake-only features and same time-based split for both models: train Jan-Sep 2025 (148,675 rows, overdue 0.1216), test Oct-Dec 2025 (38,869 rows, overdue 0.1254).

| Model | Accuracy | F1 (overdue) | Overdue recall |
|---|---|---|---|
| Majority (always ONTIME) | 0.8746 | 0.0000 | 0.0000 |
| Logistic regression | 0.9104 | 0.5676 | 0.4687 |
