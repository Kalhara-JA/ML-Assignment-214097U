# Step 2 & 3: Model Selection, Training, and Evaluation

## Algorithms Chosen (Traditional ML Only)
The following classical regression models were compared:
- Ridge Regression
- Support Vector Regression (RBF kernel)
- Random Forest Regressor

These models were selected to compare:
- a linear baseline (Ridge),
- a non-linear margin-based model (SVR),
- and a non-linear ensemble tree model (Random Forest).

## Feature Engineering
Target variable: `arrivals` (monthly tourist arrivals).

Engineered predictors:
- Calendar features: `year`, `month`, `month_sin`, `month_cos`
- Lag features: `lag_1`, `lag_2`, `lag_3`, `lag_12`
- Rolling statistics: `rolling_3`, `rolling_6`
- Trend features: `pct_change_1`, `pct_change_12`

## Data Split Strategy
Chronological split to avoid leakage:
- Train: 79 rows
- Validation: 12 rows
- Test: 12 rows

Cross-validation for tuning:
- `TimeSeriesSplit(n_splits=5)` on training set only.

## Hyperparameter Tuning
Grid search was applied for each model using validation metric:
- Scoring metric in CV: negative RMSE
- Final model selection metric: validation RMSE

Best tuned model:
- `RandomForestRegressor(max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=200, random_state=42)`

## Performance Summary
From `outputs/model_comparison.csv`:

- Random Forest
  - Validation RMSE: **28,589.68**
  - Test RMSE: **19,255.20**
  - Test MAE: **15,864.72**
  - Test MAPE: **10.55%**
  - Test R2: **0.842**

- Ridge Regression
  - Validation RMSE: 29,663.09
  - Test RMSE: 24,506.43
  - Test R2: 0.744

- SVR (RBF)
  - Validation RMSE: 64,296.32
  - Test RMSE: 65,627.57
  - Test R2: -0.837

## Saved Artifacts
- Model comparison: `outputs/model_comparison.csv`
- Best model metrics: `outputs/best_model_metrics.json`
- Best model binary: `outputs/models/best_model.joblib`
- Test predictions: `outputs/test_predictions.csv`
- Plot (comparison): `outputs/figures/model_comparison_rmse.png`
- Plot (actual vs predicted): `outputs/figures/test_actual_vs_predicted.png`
