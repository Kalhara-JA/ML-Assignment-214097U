# Step 7 Front-End Integration

## Implemented Front-End
A Streamlit web application has been added:
- App file: `app.py`

## Features
- Displays source and model overview.
- Lets users set forecast horizon (1 to 12 months).
- Allows scenario-based override of the latest known arrival value.
- Generates future predictions using the trained best model.
- Shows downloadable prediction table.
- Shows explainability artifacts (SHAP summary visuals).

## How to Run
```bash
. .venv/bin/activate
streamlit run app.py
```

## Demo Notes (for your video)
1. Open app and show source details and model comparison table.
2. Run forecast with default settings and explain output chart/table.
3. Change scenario input (latest arrivals override) and rerun.
4. Open `Why This Prediction` tab and discuss top driving features.
