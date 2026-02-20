# Step 1: Problem Definition and Dataset Collection (Sri Lanka)

## Problem Statement
Forecast monthly tourist arrivals to Sri Lanka using classical machine learning.  
Accurate short-term demand forecasts can support tourism planning, staffing, infrastructure utilization, and policy decisions.

## Dataset Used
- Name: **Sri Lanka Monthly Tourist Arrivals (2016-2025)**
- Source owner: **Sri Lanka Tourism Development Authority (SLTDA)**
- Public source section: [SLTDA Statistics](https://www.sltda.gov.lk/en/statistics)
- Source pages used:
  - [Monthly Tourist Arrivals Reports 2025](https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports-2025)
  - [Monthly Tourist Arrivals Reports 2024](https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports-2024)
  - [Monthly Tourist Arrivals Reports 2023](https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports-2023)
  - [Monthly Tourist Arrivals Reports 2022](https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports-2022)
  - [Monthly Tourist Arrivals Reports 2021](https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports-2021)
  - [Monthly Tourist Arrivals Reports 2020](https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports-2020)
  - [Monthly Tourist Arrivals Reports 2019](https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports-2019)
  - [Monthly Tourist Arrivals Reports 2018](https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports-2018)
  - [Monthly Tourist Arrivals Reports 2017](https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports-2017)

## Latest Public Availability
- Latest year available in the selected source pages: **2025**.
- Latest month in the compiled dataset: **December 2025**.
- Data compiled on: **February 19, 2026**.

## Local Dataset Artifacts
- Dataset CSV: `data/raw/sri_lanka_tourism_monthly_arrivals_2016_2025.csv`
- Collection script: `src/step1_prepare_dataset.py`
- Summary JSON: `outputs/step1_dataset_summary.json`

## Features and Target
- Raw rows: **120** monthly records.
- Time span: **2016-01-01 to 2025-12-01**.
- Core fields:
  - `year`
  - `month`
  - `month_name`
  - `arrivals` (target for regression)
  - `date`
  - `source_url`

## Planned Preprocessing
- Convert data to chronological monthly time series.
- Engineer lag and rolling features (`lag_1`, `lag_12`, `rolling_3`, etc.).
- Handle seasonal effects using cyclic month encoding (`month_sin`, `month_cos`).
- Split data chronologically into train/validation/test to avoid temporal leakage.

## Ethical and Practical Notes
- Dataset is aggregate public statistics (no personal/sensitive individual records).
- Forecasts should be treated as decision-support estimates, not exact ground truth.
- External shocks (pandemic, policy changes, geopolitics) can create regime shifts and reduce model stability.
