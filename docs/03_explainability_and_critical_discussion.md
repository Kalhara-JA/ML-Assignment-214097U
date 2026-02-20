# Step 4 & 5: Explainability and Critical Discussion

## Explainability Methods Applied
At least one XAI method was required. This project applies:
- SHAP (SHapley Additive exPlanations)

Generated from:
- `src/explain_model.py`

## What the Model Learned
Top features from SHAP mean absolute values (`outputs/explainability/shap_feature_importance.csv`):
1. `lag_1`
2. `lag_12`
3. `rolling_3`
4. `pct_change_12`
5. `month`

Interpretation:
- The most recent month (`lag_1`) strongly influences the next month.
- Annual seasonality (`lag_12`) is important, consistent with tourism cycles.
- Short-term and year-over-year trend changes (`pct_change_1`, `pct_change_12`) also contribute.

This behavior aligns with domain expectations for tourism demand forecasting.

## Explainability Artifacts
- SHAP importance table: `outputs/explainability/shap_feature_importance.csv`
- SHAP beeswarm chart: `outputs/figures/shap_summary_beeswarm.png`
- SHAP bar chart: `outputs/figures/shap_summary_bar.png`
- Summary file: `outputs/explainability/explainability_summary.json`

## Limitations
- Small dataset size (120 months total; 103 usable after lag feature creation).
- Major regime shift during COVID-19 years (2020-2021) makes patterns unstable.
- Forecasts are univariate-style with limited external drivers (no explicit flight, exchange rate, geopolitical, or campaign features).

## Data Quality and Bias Risks
- Data is official aggregate statistics, but revisions and reporting delays can occur.
- Pandemic disruptions may bias trend features and reduce transferability to normal periods.
- Country-level total arrivals hide distributional variations by market segment/country of origin.

## Ethical and Real-World Considerations
- Forecast outputs should support planning, not replace policy judgment.
- Mis-forecasting can cause over/under-allocation of tourism resources.
- The model should be periodically retrained as new monthly data is released.
