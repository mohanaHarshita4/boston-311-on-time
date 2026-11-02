# Error check — Boston 311, 2025

Logistic regression baseline, same intake-only features and same time-based split as baselines.py: train Jan-Sep 2025, test Oct-Dec 2025 (38,869 rows).

Overdue in test: 4,875. Missed (predicted ONTIME): 2,590. Overall overdue recall: 0.4687.

## Most missed request types

| Request type | Test requests | Overdue | Missed | Overdue recall |
|---|---|---|---|---|
| Abandoned Vehicles | 1,061 | 573 | 549 | 0.0419 |
| Request for Pothole Repair | 723 | 321 | 263 | 0.1807 |
| Requests for Street Cleaning | 4,432 | 238 | 235 | 0.0126 |
| Traffic Signal Inspection | 1,014 | 213 | 212 | 0.0047 |
| Parking Enforcement | 2,575 | 190 | 190 | 0.0000 |
| Sidewalk Repair (Make Safe) | 471 | 201 | 148 | 0.2637 |
| Graffiti Removal | 677 | 140 | 140 | 0.0000 |
| Equipment Repair | 300 | 101 | 101 | 0.0000 |
| Request for Recycling Cart | 914 | 47 | 47 | 0.0000 |
| Pick up Dead Animal | 1,678 | 46 | 46 | 0.0000 |

## Most missed neighbourhoods

| Neighbourhood | Test requests | Overdue | Missed | Overdue recall |
|---|---|---|---|---|
| Dorchester | 5,259 | 742 | 392 | 0.4717 |
| Allston / Brighton | 2,999 | 473 | 280 | 0.4080 |
| Roxbury | 3,659 | 438 | 245 | 0.4406 |
| Jamaica Plain | 2,211 | 414 | 175 | 0.5773 |
| Hyde Park | 1,596 | 360 | 140 | 0.6111 |
| Roslindale | 1,404 | 290 | 132 | 0.5448 |
| Boston | 1,503 | 172 | 131 | 0.2384 |
| Downtown / Financial District | 2,232 | 208 | 120 | 0.4231 |
| Back Bay | 2,263 | 184 | 119 | 0.3533 |
| South End | 3,137 | 193 | 116 | 0.3990 |

Only groups with at least 20 overdue requests in the test set are shown, sorted by number missed, not by recall alone, so the biggest groups of missed requests come first.
