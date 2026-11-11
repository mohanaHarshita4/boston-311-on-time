# Cross-validation — Boston 311, 2025

Logistic regression baseline, same intake-only features as baselines.py, checked with expanding windows inside the training months only (Jan-Sep 2025). Each fold trains on earlier months and is checked on the months right after, so no later month leaks into training. The Oct-Dec 2025 test set is not used here; it stays a fresh final check. For comparison, the single Oct-Dec test result was accuracy 0.9104, F1 0.5676, overdue recall 0.4687.

| Fold | Check requests | Overdue rate | Accuracy | F1 (overdue) | Overdue recall |
|---|---|---|---|---|---|
| Train Jan-Feb, check Mar-Apr | 29,275 | 0.1482 | 0.8813 | 0.5203 | 0.4343 |
| Train Jan-Apr, check May-Jun | 32,199 | 0.1190 | 0.9157 | 0.5658 | 0.4614 |
| Train Jan-Jun, check Jul-Sep | 56,237 | 0.1049 | 0.9271 | 0.5550 | 0.4337 |
| Mean |  |  |  | 0.5470 | 0.4431 |

If the fold F1 and recall sit near the Oct-Dec numbers, the result is not just one lucky set of months. If a fold is far lower, that says something about those months, and it belongs in the write-up, not hidden.
