"""Build a local, report-ready Sri Lanka tourism arrivals dataset.

This script compiles monthly SLTDA arrival counts (2016-2025) into a single CSV
and writes a metadata summary JSON used by the report and Streamlit app.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

ARRIVALS_BY_YEAR = {
    # Monthly arrivals values manually compiled from SLTDA monthly reports.
    2016: [194280, 193507, 188076, 136367, 125044, 118970, 115971, 143918, 113099, 109375, 121976, 224791],
    2017: [219360, 197517, 209960, 180249, 121891, 123351, 150445, 136419, 145077, 153398, 148499, 244536],
    2018: [238924, 235618, 233382, 180429, 129466, 146828, 217829, 200359, 149087, 153123, 195582, 253169],
    2019: [244239, 252033, 244328, 166975, 166869, 123675, 115701, 143587, 108575, 118743, 176984, 241663],
    2020: [228434, 207507, 71457, 0, 0, 1683, 4168, 3935, 3527, 3980, 4493, 9633],
    2021: [19417, 36603, 49995, 24297, 14975, 1248, 2538, 4355, 13186, 22371, 4494, 8933],
    2022: [82327, 96507, 106500, 62980, 30207, 32856, 47293, 37760, 29802, 59759, 59091, 91961],
    2023: [102545, 107639, 125495, 105498, 121913, 100388, 143039, 136405, 111938, 109199, 151496, 210352],
    2024: [208253, 218350, 209181, 148867, 132344, 113470, 147208, 164609, 122140, 135907, 184158, 253309],
    2025: [252761, 240217, 229298, 174608, 132919, 123914, 143039, 165465, 113204, 109275, 127756, 178669],
}

SOURCE_URLS = {
    2016: "https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports-2017",
    2017: "https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports-2017",
    2018: "https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports-2018",
    2019: "https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports-2019",
    2020: "https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports-2020",
    2021: "https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports-2021",
    2022: "https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports-2022",
    2023: "https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports-2023",
    2024: "https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports-2024",
    2025: "https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports-2025",
}


def build_dataset() -> pd.DataFrame:
    """Return a normalized monthly arrivals table sorted by date."""
    rows: list[dict[str, object]] = []
    for year, arrivals in sorted(ARRIVALS_BY_YEAR.items()):
        for month_idx, value in enumerate(arrivals, start=1):
            rows.append(
                {
                    "year": year,
                    "month": month_idx,
                    "month_name": MONTH_ORDER[month_idx - 1],
                    "arrivals": int(value),
                    "source_url": SOURCE_URLS[year],
                }
            )

    df = pd.DataFrame(rows)
    # Normalize every record to first day of month for consistent time-series indexing.
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    return df.sort_values("date").reset_index(drop=True)


def main() -> None:
    """Generate dataset artifacts under `data/raw` and `outputs`."""
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "raw"
    output_dir = root / "outputs"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataset()
    csv_path = data_dir / "sri_lanka_tourism_monthly_arrivals_2016_2025.csv"
    df.to_csv(csv_path, index=False)

    yearly_totals = df.groupby("year")["arrivals"].sum().to_dict()
    # Keep metadata in JSON so report writing and source verification are reproducible.
    summary = {
        "dataset_name": "Sri Lanka Monthly Tourist Arrivals",
        "source_owner": "Sri Lanka Tourism Development Authority (SLTDA)",
        "source_collection": "Monthly Tourist Arrivals Reports",
        "source_base_url": "https://www.sltda.gov.lk/en/statistics",
        "n_rows": int(df.shape[0]),
        "n_features_raw": 6,
        "date_range": {
            "start": df["date"].min().strftime("%Y-%m-%d"),
            "end": df["date"].max().strftime("%Y-%m-%d"),
        },
        "latest_public_data_year": int(df["year"].max()),
        "missing_values_total": int(df.isna().sum().sum()),
        "yearly_totals": {str(k): int(v) for k, v in yearly_totals.items()},
        "source_urls": SOURCE_URLS,
        "collected_on": "2026-02-19",
        "csv_path": str(csv_path.relative_to(root)),
    }
    summary_path = output_dir / "step1_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved dataset to: {csv_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
