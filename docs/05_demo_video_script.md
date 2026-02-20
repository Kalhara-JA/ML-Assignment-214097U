# Demo Video Script (3-5 Minutes)

## 0:00-0:20 Intro
Hello, this is my Machine Learning assignment on forecasting monthly tourist arrivals in Sri Lanka using traditional machine learning models.

## 0:20-0:55 Problem and Dataset
The problem is short-term tourism demand forecasting for Sri Lanka.  
I used publicly available monthly arrival statistics from the Sri Lanka Tourism Development Authority, covering January 2016 to December 2025.  
The local dataset has 120 monthly rows and includes source links for traceability.

## 0:55-1:35 Method
I framed this as a regression task where the target is monthly arrivals.  
I created time-series features such as lag 1 month, lag 12 months, rolling averages, and seasonal month encodings using sine and cosine.  
Then I compared three traditional models:
- Ridge Regression
- SVR with RBF kernel
- Random Forest Regressor

I used chronological train/validation/test splitting and time-series cross-validation for tuning.

## 1:35-2:10 Results
The best model is Random Forest.  
On the test set, RMSE is about 19,255 and R2 is about 0.842.  
This outperformed Ridge and SVR in this dataset.

## 2:10-3:45 Front-End Demo (Streamlit)
Now I will show the app.

1. In the Overview tab:
   - Show source details and model comparison table.
   - Point out key metrics (RMSE, MAE, R2).

2. In the sidebar:
   - Set forecast horizon, for example 6 months.
   - First run baseline scenario.

3. In Forecast Output tab:
   - Show monthly predictions table.
   - Show historical plus forecast chart.
   - Download forecast CSV.

4. Scenario analysis:
   - Switch to custom latest arrivals.
   - Enter a different latest value.
   - Click Generate Forecast and show how predictions change.

5. In Explainability tab:
   - Show SHAP feature-importance results.
   - Explain top features: lag_1 and lag_12 are strongest.
   - Show SHAP summary plots for interpretation.

## 3:45-4:30 Critical Discussion
Main limitations:
- Small sample size for time-series ML.
- COVID shock years create structural breaks.
- Model uses limited external explanatory variables.

Potential improvements:
- Add macroeconomic and flight capacity variables.
- Retrain monthly with latest data updates.
- Build uncertainty intervals and monitoring.

## 4:30-5:00 Closing
In summary, the project meets the assignment requirements with a Sri Lankan public dataset, traditional ML comparison, explainability, and a working front-end integration.
