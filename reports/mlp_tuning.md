# MLP tuning — Boston 311, 2025

Same intake-only features and same time-based split as baselines: train Jan-Sep 2025 (148,675 rows, overdue 0.1216), test Oct-Dec 2025 (38,869 rows, overdue 0.1254). Features: 133. Each setting: Adam, batch 512, 5 epochs, BCE loss with pos_weight 7.22, threshold 0.5. Best setting by F1 for the overdue class: 64-32, lr 0.001, no dropout.

| Setting | Accuracy | F1 (overdue) | Overdue recall |
|---|---|---|---|
| 64-32, lr 0.001, no dropout | 0.8533 | 0.5912 | 0.8457 |
| 128-64, lr 0.001, dropout 0.2 | 0.8444 | 0.5779 | 0.8492 |
| 64-32, lr 0.0005, dropout 0.2 | 0.8539 | 0.5891 | 0.8349 |
