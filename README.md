# Boston 311: Will This Request Go Overdue?

## Problem

When a Boston resident reports a pothole, a broken streetlight, or missed trash pickup, the city sets a target date. Some requests close on time, others go overdue, and nobody knows at intake which ones will get stuck.

## Goal

Predict, at the time a request is opened, whether it will close overdue, so likely-late requests can be flagged early.

## Why this project

- Real, public, city operations data — not a toy dataset.
- A clear question a data analyst would actually answer: where should the city act early?
- A clean label that already exists in the data: `on_time` = ONTIME / OVERDUE.
- No leakage if we are careful: only use fields known at intake.

## Data

Source: City of Boston, Analyze Boston — 311 Service Requests
https://data.boston.gov/dataset/311-service-requests

Files used:
- 2024: `lagan_311_open_data_2024_rvsd3.csv`
- 2025: `lagan_311_open_data_2025_rvsd3.csv`

The raw files are downloaded by `src/download_data.py` and stored locally in `data/raw/`. They are not committed to GitHub.

2025 file at a glance:
- 276,093 rows
- 205,871 closed requests
- 88.85% ONTIME, 11.15% OVERDUE among closed requests
- Median time to close: 0.28 days; mean: 8.26 days, showing a long tail
- Median SLA window (target − open): 2.88 days
- Most common closed request types: Improper Storage of Trash (Barrels), Requests for Street Cleaning, CE Collection, Parking Enforcement, Needle Pickup

Full exploratory analysis: see `reports/eda_summary.md` and `reports/figures/`.

## Target and leakage rules

- Target: `is_overdue = 1` if `on_time == OVERDUE`, else `0`. Only closed requests are used for training, because open requests have no outcome yet.
- Allowed at intake: `case_title`, `subject`, `reason`, `type`, `department`, `neighborhood`, `source`, open hour / weekday / month, and SLA window in days (`sla_target_dt − open_dt`).
- Not allowed as features: `closed_dt`, `closure_reason`, `case_status`, and `on_time` itself. Those only exist after closure.

## Models

All models use the same features and the same time-based split (train on earlier months, test on later months), and are compared on accuracy, F1, and recall for the OVERDUE class, not accuracy alone, because OVERDUE is only about 11% of closed requests.

1. Majority-class baseline — always predict ONTIME
2. Logistic regression (scikit-learn)
3. SVM (scikit-learn)
4. Small MLP / DNN (PyTorch)

## Repo structure

```
boston-311-on-time/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   ├── raw/               # local only, git-ignored, large CSVs live here
│   └── processed/
├── src/
│   ├── download_data.py
│   └── eda.py
└── reports/
    ├── eda_summary.md
    └── figures/
```

## How to run

```bash
pip install -r requirements.txt
python src/download_data.py
python src/eda.py
```

`download_data.py` downloads the 2024 and 2025 CSVs to `data/raw/` (about 153 MB for 2025 alone, not committed). `eda.py` writes the EDA summary and figures to `reports/`.

## Author

Mohana Harshita Rangubhotla

MS Computer Science student at Khoury College of Computer Science, Northeastern University. I am taking Machine Learning this semester, and I built this project to put that coursework into practical use on a real city dataset.
