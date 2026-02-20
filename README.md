# Sri Lanka Tourism Arrival Predictor

## Assignment Details
- `214097U - Kalhara J.A.K.`
- `ML Assignment`

This project uses a publicly available Sri Lankan dataset from SLTDA monthly tourist arrival reports and applies traditional ML models for forecasting.

## Setup
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Run Step-by-Step
```bash
. .venv/bin/activate
python src/step1_prepare_dataset.py
python src/train_and_evaluate.py
python src/explain_model.py
```

## Front-End (Streamlit)
```bash
. .venv/bin/activate
streamlit run app.py
```

The app includes:
- Forecast controls (1-12 month horizon)
- Scenario testing by overriding latest known arrivals
- Downloadable forecast table
- Explainability panels (SHAP summary and SHAP global importance)

## Streamlit Cloud Deployment
This repository is structured for Streamlit Cloud:
- Entry file: `app.py`
- Python dependencies: `requirements.txt`
- Python runtime: `runtime.txt`
- Streamlit config: `.streamlit/config.toml`

Deploy steps:
1. Push this project to GitHub.
2. In Streamlit Cloud, create a new app from the repo.
3. Set main file path to `app.py`.
4. Deploy.

First-run behavior on Cloud:
- If model artifacts are missing, the app automatically runs:
  - `src/step1_prepare_dataset.py`
  - `src/train_and_evaluate.py`
  - `src/explain_model.py`
- Then it reloads with ready-to-use predictions and SHAP outputs.

## Main Outputs
- `data/raw/sri_lanka_tourism_monthly_arrivals_2016_2025.csv`
- `outputs/model_comparison.csv`
- `outputs/best_model_metrics.json`
- `outputs/test_predictions.csv`
- `outputs/figures/*.png`
- `outputs/explainability/*.csv`
- `app.py` (front-end)

## Report Draft Sections
- `docs/01_problem_and_dataset.md`
- `docs/02_model_selection_and_evaluation.md`
- `docs/03_explainability_and_critical_discussion.md`
- `docs/04_frontend.md`
- `docs/05_demo_video_script.md`
- `docs/06_final_report_draft.md`
