"""Download Boston 311 CSVs from Analyze Boston (CKAN).

Run: python src/download_data.py
Writes: data/raw/boston311_2024.csv, data/raw/boston311_2025.csv
These files are git-ignored; only the script and the EDA summary are committed.
"""
import os
import urllib.request

BASE = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource"

FILES = {
    # resource ids verified Oct 29, 2026 via the CKAN package_show API
    "2024": "dff4d804-5031-443a-8409-8344efd0e5c8",
    "2025": "9d7c2214-4709-478a-a2e8-fb2020a5bb94",
}
FILENAMES = {
    "2024": "lagan_311_open_data_2024_rvsd3.csv",
    "2025": "lagan_311_open_data_2025_rvsd3.csv",
}


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    for year, resource_id in FILES.items():
        url = f"{BASE}/{resource_id}/download/{FILENAMES[year]}"
        out = os.path.join(out_dir, f"boston311_{year}.csv")
        if os.path.exists(out):
            print(f"exists, skipping: {out}")
            continue
        print(f"downloading {year} -> {out}")
        urllib.request.urlretrieve(url, out)
        print(f"done: {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
