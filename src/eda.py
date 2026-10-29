"""EDA for Boston 311, 2025 file.

Run: python src/eda.py
Input:  data/raw/boston311_2025.csv  (see src/download_data.py)
Output: reports/eda_summary.md, reports/figures/overdue_by_case_title.png,
        reports/figures/on_time_share.png


"""
import os

import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw", "boston311_2025.csv")
REPORTS = os.path.join(ROOT, "reports")
FIGURES = os.path.join(REPORTS, "figures")

USECOLS = [
    "open_dt", "sla_target_dt", "closed_dt", "on_time", "case_status",
    "case_title", "department", "source", "neighborhood",
]


def main():
    os.makedirs(FIGURES, exist_ok=True)
    df = pd.read_csv(RAW, usecols=USECOLS, low_memory=False)
    closed = df[df["case_status"] == "Closed"].copy()
    for col in ["open_dt", "closed_dt", "sla_target_dt"]:
        closed[col] = pd.to_datetime(closed[col], errors="coerce", format="mixed")
    closed["days_to_close"] = (closed["closed_dt"] - closed["open_dt"]).dt.total_seconds() / 86400
    closed["sla_days"] = (closed["sla_target_dt"] - closed["open_dt"]).dt.total_seconds() / 86400
    closed["is_overdue"] = (closed["on_time"] == "OVERDUE").astype(int)

    overdue_rate = closed["is_overdue"].mean()
    top = closed.groupby("case_title").agg(
        n=("is_overdue", "size"), overdue_rate=("is_overdue", "mean")
    ).sort_values("n", ascending=False).head(12)

    # Figure 1: overdue rate for the 12 most common request types
    fig, ax = plt.subplots(figsize=(10, 5))
    top_sorted = top.sort_values("overdue_rate")
    ax.barh(top_sorted.index, top_sorted["overdue_rate"] * 100)
    ax.set_xlabel("Share overdue (%)")
    ax.set_title("Overdue rate by request type — 2025, closed requests only")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "overdue_by_case_title.png"), dpi=150)
    plt.close()

    # Figure 2: overall on-time share
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [1 - overdue_rate, overdue_rate],
        labels=[f"ONTIME {(1-overdue_rate)*100:.1f}%", f"OVERDUE {overdue_rate*100:.1f}%"],
        autopct="",
    )
    ax.set_title("Closed 2025 requests: on-time vs overdue")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "on_time_share.png"), dpi=150)
    plt.close()

    summary = f"""# EDA Summary — Boston 311, 2025

Source: City of Boston, Analyze Boston, 311 Service Requests.
File: `data/raw/boston311_2025.csv` (downloaded Oct 29, 2026).

## Counts
- Rows in 2025 file: {len(df):,}
- Closed requests: {len(closed):,}
- Open requests: {(df['case_status'] == 'Open').sum():,}

## Target
- ONTIME among closed: {(1-overdue_rate)*100:.2f}%
- OVERDUE among closed: {overdue_rate*100:.2f}%
- This is imbalanced, so later models will be judged on F1 and recall for OVERDUE, not accuracy alone.

## Timing
- Median days to close: {closed['days_to_close'].median():.2f} days
- Mean days to close: {closed['days_to_close'].mean():.2f} days (long tail)
- Median SLA window (target - open): {closed['sla_days'].median():.2f} days
- Open date range (closed): {closed['open_dt'].min().date()} to {closed['open_dt'].max().date()}

## Most common closed request types
{closed['case_title'].value_counts().head(10).to_string()}

## Overdue rate by most common request type
{top.to_string()}

Takeaway for modelling: overdue risk is not uniform. For example, in this
snapshot Recycling Cart Return and Sign Repair go overdue far more often
than trash or needle pickup. That is why request type is a core intake feature,
and why a model that learns combinations (type + neighbourhood + SLA window)
can beat a single average.

## Top departments, sources, neighbourhoods (closed)
Departments:
{closed['department'].value_counts().head(8).to_string()}

Sources:
{closed['source'].value_counts().head(8).to_string()}

Neighbourhoods:
{closed['neighborhood'].value_counts().head(8).to_string()}

## Leakage rule
Use only intake-known fields: case_title, subject, reason, type, department,
neighborhood, source, open hour/weekday/month, SLA window.
Never use closed_dt, closure_reason, case_status, or on_time as a feature.
"""
    with open(os.path.join(REPORTS, "eda_summary.md"), "w") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
